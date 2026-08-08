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
        # Sonnet = best balance for real-time voice (smart enough for the nuance,
        # fast enough to avoid dead air). Drop to claude-haiku-4-5 if latency bites;
        # avoid Opus here — it's the slowest and the extra reasoning is wasted.
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
    )

    # STT: default to a telephony-tuned Deepgram model and BOOST the dietary
    # vocabulary so 8kHz phone audio doesn't mangle key words (e.g. hearing
    # "scallions" as "scallops"). Wrapped defensively: if the enhanced options
    # aren't accepted by this SDK build, fall back to the plain service so a
    # call never fails to start over an STT config detail.
    _dg_key = os.getenv("DEEPGRAM_API_KEY")
    # ":N" is Deepgram keyword-boost intensity — higher = more likely to be heard.
    DIET_KEYWORDS = [
        "scallion:3", "scallions:3", "allium:2", "shallot:2", "leek:2", "chive:2",
        "paneer:2", "ghee:2", "tofu:2", "seitan:2", "lentil:2", "chickpea:2",
        "vegan:2", "Jain:3", "truffle:3", "asafoetida:2", "hing:2",
    ]
    try:
        from deepgram import LiveOptions

        stt = DeepgramSTTService(
            api_key=_dg_key,
            live_options=LiveOptions(
                model=os.getenv("DEEPGRAM_MODEL", "nova-2-phonecall"),
                language="en-US",
                smart_format=True,
                punctuate=True,
                keywords=DIET_KEYWORDS,
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

    # Seed the conversation with our tuned system prompt for THIS restaurant.
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Clean transcript logging: emit readable "TRANSCRIPT | USER: ..." /
    # "TRANSCRIPT | ASSISTANT: ..." lines to the deploy logs so we can review
    # exactly what each side said (the raw DEBUG frames don't give a usable
    # transcript). Wrapped defensively — if this pipecat build's API differs,
    # transcript logging simply turns off and the call still runs normally.
    transcript_user = transcript_assistant = None
    try:
        from pipecat.processors.transcript_processor import TranscriptProcessor

        transcript = TranscriptProcessor()
        transcript_user = transcript.user()
        transcript_assistant = transcript.assistant()

        @transcript.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for msg in frame.messages:
                logger.info(f"TRANSCRIPT | {msg.role.upper()}: {msg.content}")
    except Exception as e:
        logger.warning(f"Transcript logging disabled ({e})")
        transcript_user = transcript_assistant = None

    stages = [
        transport.input(),   # audio in from Twilio
        stt,                 # speech -> text (Deepgram)
    ]
    if transcript_user is not None:
        stages.append(transcript_user)      # capture the caller's (restaurant's) words
    stages += [
        user_aggregator,
        llm,                 # Anthropic Claude (our tuned prompt)
        tts,                 # text -> speech (ElevenLabs Adam)
        transport.output(),  # audio out to Twilio
    ]
    if transcript_assistant is not None:
        stages.append(transcript_assistant)  # capture what the agent said
    stages.append(assistant_aggregator)

    pipeline = Pipeline(stages)

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
