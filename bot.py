#
# voice-caller — Pipecat outbound dietary-inquiry bot
# Based on pipecat-examples/twilio-chatbot/outbound, customized to:
#   - use ElevenLabs (the "Adam" voice) for TTS
#   - use Anthropic Claude (Sonnet) for the LLM
#   - load our tuned dietary-inquiry system prompt from prompt.txt
#   - inject per-call context (restaurant_name, call_notes, cuisine, ...) that
#     arrives as Twilio <Stream> parameters (see server_utils.generate_twiml)
#

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# The dynamic-variable placeholders our tuned prompt expects (same names we used on Retell).
PROMPT_VARS = ["restaurant_name", "call_notes", "cuisine", "candidate_dishes", "visit_date", "party_size"]

with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "r") as f:
    PROMPT_TEMPLATE = f.read()


def build_system_prompt(call_ctx: dict) -> str:
    """Substitute {{var}} placeholders in the tuned prompt with this call's context."""
    text = PROMPT_TEMPLATE
    for var in PROMPT_VARS:
        text = text.replace("{{" + var + "}}", str(call_ctx.get(var, "") or ""))
    return text


async def run_bot(transport: BaseTransport, call_ctx: dict, handle_sigint: bool):
    system_prompt = build_system_prompt(call_ctx)
    logger.info(f"Calling {call_ctx.get('restaurant_name')!r} — prompt {len(system_prompt)} chars")

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        # Haiku 4.5 = the latency sweet spot for real-time voice (~0.4s to first
        # token). Sonnet's ~2s first-token made calls feel sluggish; Opus is worse.
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        # Cache the ~7.7k-token system prompt so it isn't reprocessed every turn
        # (0.1x input cost after turn 1 + faster TTFT). ~3-min calls fit the 5-min TTL.
        params=AnthropicLLMService.InputParams(enable_prompt_caching=True),
    )

    # STT: default to a telephony-tuned Deepgram model and BOOST the dietary
    # vocabulary so 8kHz phone audio doesn't mangle key words (e.g. hearing
    # "scallions" as "scallops"). Wrapped defensively: if the enhanced options
    # aren't accepted by this SDK build, fall back to the plain service so a
    # call never fails to start over an STT config detail.
    _dg_key = os.getenv("DEEPGRAM_API_KEY")
    # The dietary vocabulary to boost so 8kHz audio doesn't mangle it (plain terms).
    DIET_TERMS = [
        "scallion", "scallions", "allium", "shallot", "leek", "chive",
        "paneer", "ghee", "tofu", "seitan", "lentil", "chickpea",
        "vegan", "Jain", "truffle", "asafoetida", "hing",
    ]
    # nova-3 (default; lower WER incl. 8kHz) boosts via keyterm prompting (plain
    # terms). nova-2* uses keywords with ":N" intensities. Passing the wrong one
    # for the model makes Deepgram reject the connection, so branch on the model.
    _dg_model = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
    if _dg_model.startswith("nova-3"):
        _dg_boost = {"keyterm": DIET_TERMS}
    else:
        _dg_boost = {"keywords": [f"{t}:2" for t in DIET_TERMS]}
    try:
        # pipecat ships its own LiveOptions compat wrapper; the deepgram SDK no
        # longer exports LiveOptions at top level (that import silently fell back
        # to default STT, so the telephony model + keyword boost never loaded).
        from pipecat.services.deepgram.stt import LiveOptions

        stt = DeepgramSTTService(
            api_key=_dg_key,
            live_options=LiveOptions(
                model=_dg_model,
                language="en-US",
                smart_format=True,
                punctuate=True,
                **_dg_boost,
            ),
        )
    except Exception as e:
        logger.warning(f"Deepgram enhanced options unavailable ({e}); using defaults")
        stt = DeepgramSTTService(api_key=_dg_key)

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),  # ElevenLabs voice id (swap from Railway, no code change)
        # turbo sounds noticeably more natural than flash and is still very fast
        # (~0.1s to first audio); we have plenty of latency headroom for it.
        model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        # 1.0 = natural pacing. Tunable from Railway (ELEVENLABS_SPEED) without a code change.
        params=ElevenLabsTTSService.InputParams(speed=float(os.getenv("ELEVENLABS_SPEED", "1.0"))),
    )

    # Turn-taking / barge-in tuning. pipecat's DEFAULT lets ANY 200ms of speech
    # (a breath, a one-word overlap, kitchen noise) instantly interrupt the bot,
    # which produced a greeting-collision loop on real calls. Fix (verified vs
    # pipecat 1.7.0 source):
    #   - MinWordsUserTurnStartStrategy(3): while the bot is speaking, an
    #     interruption needs >=3 real words (interims count, so it stays fast);
    #     while the bot is silent, a single word still starts a turn. Replaces
    #     BOTH default start strategies. Stop strategy is left as the default
    #     smart-turn analyzer (correct for "callee pauses to check the kitchen").
    #   - FirstSpeechUserMuteStrategy: guarantees the opening greeting plays to
    #     completion once, no matter what the caller does over it.
    # Wrapped defensively: if a strategy API ever differs, fall back to defaults
    # so a call still runs (just with the old, twitchier turn-taking).
    user_params_kwargs = dict(vad_analyzer=SileroVADAnalyzer())
    try:
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
        from pipecat.turns.user_mute import FirstSpeechUserMuteStrategy

        user_params_kwargs["user_turn_strategies"] = UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=3)],
        )
        user_params_kwargs["user_mute_strategies"] = [FirstSpeechUserMuteStrategy()]
        logger.info("Barge-in tuning active: MinWords(3) + FirstSpeechMute")
    except Exception as e:
        logger.warning(f"Turn-taking strategies unavailable ({e}); using pipecat defaults")

    # Seed the conversation with our tuned system prompt for THIS restaurant.
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(**user_params_kwargs),
    )

    pipeline = Pipeline(
        [
            transport.input(),   # audio in from Twilio
            stt,                 # speech -> text (Deepgram)
            user_aggregator,
            llm,                 # Anthropic Claude (our tuned prompt)
            tts,                 # text -> speech (ElevenLabs Adam)
            transport.output(),  # audio out to Twilio
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # User-first: we wait for the restaurant to speak; the prompt's OPENING
        # section tells the LLM to greet only after it hears them.
        logger.info("Media stream connected; waiting for the other side to speak")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended")
        # Dump a clean, readable transcript (system prompt excluded) so calls can
        # be reviewed by filtering the logs for "TRANSCRIPT". pipecat 1.7 has no
        # TranscriptProcessor, so we read the final LLM context directly.
        try:
            for m in context.messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "?")
                if role == "system":
                    continue
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                    )
                logger.info(f"TRANSCRIPT | {str(role).upper()}: {content}")
        except Exception as e:
            logger.warning(f"Transcript dump failed ({e})")
        await worker.cancel()

    from pipecat.workers.runner import WorkerRunner

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point invoked per WebSocket connection (from server.py /ws)."""
    transport_params = {
        "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)

    # Per-call context arrives as Twilio <Stream> parameters -> call_data.
    call_data = runner_args.call_data
    call_ctx = {}
    if call_data is not None:
        # call_data exposes to_number/from_number plus any custom stream params.
        raw = getattr(call_data, "__dict__", {}) or {}
        call_ctx.update(raw)
        # custom params may be nested under a dict-like attribute depending on version:
        for attr in ("body", "custom_parameters", "parameters"):
            extra = getattr(call_data, attr, None)
            if isinstance(extra, dict):
                call_ctx.update(extra)
    logger.info(f"call_ctx keys: {list(call_ctx.keys())}")

    await run_bot(transport, call_ctx, runner_args.handle_sigint)
