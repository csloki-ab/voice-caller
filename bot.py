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

import re

# STT keyterm boosting is built PER CALL: base dietary vocabulary + a cuisine-
# specific dish pack + this restaurant's name + the expected dish names. Without
# the cuisine pack, 8kHz audio mangled "dal"->"doll", "idli"->"Italy",
# "Tamarind"->"Cameron" — the LLM never received the real words.
BASE_DIET_TERMS = [
    "scallion", "scallions", "allium", "shallot", "leek", "chive",
    "paneer", "ghee", "tofu", "seitan", "lentil", "chickpea",
    "vegan", "Jain", "truffle", "asafoetida", "hing",
]

CUISINE_TERMS = {
    "indian": [
        "dal", "daal", "chana", "chana masala", "chole", "rajma",
        "idli", "dosa", "sambar", "uttapam", "vada", "upma", "poha",
        "roti", "naan", "paratha", "puri", "thali", "sabzi", "subzi",
        "aloo", "gobi", "bhindi", "baingan", "biryani", "pulao",
        "korma", "tadka", "masala", "pakora", "papad",
    ],
    "mexican": ["frijoles", "nopales", "rajas", "tlacoyo", "huitlacoche",
                "quesadilla", "sope", "tamal", "mole", "salsa verde"],
    "thai": ["pad thai", "pad see ew", "tom yum", "massaman", "larb", "som tum"],
    "italian": ["marinara", "pomodoro", "arrabbiata", "aglio", "focaccia", "bruschetta"],
    "chinese": ["mapo tofu", "chow fun", "lo mein", "congee", "bok choy", "doubanjiang"],
    "japanese": ["inari", "kappa maki", "agedashi", "shojin", "dashi", "edamame"],
    "middle eastern": ["falafel", "hummus", "mujadara", "fattoush", "tabbouleh", "shawarma"],
    "mediterranean": ["falafel", "hummus", "tabbouleh", "baba ganoush", "dolma"],
    "ethiopian": ["injera", "misir wot", "shiro", "gomen", "atkilt", "berbere"],
}


def build_keyterms(call_ctx: dict, cap: int = 50) -> list:
    """Build this call's Deepgram keyterm list from the per-call context."""
    terms = list(BASE_DIET_TERMS)
    cuisine = (call_ctx.get("cuisine") or "").lower()
    for key, pack in CUISINE_TERMS.items():
        if key in cuisine:
            terms += pack
    if call_ctx.get("restaurant_name"):
        terms.append(call_ctx["restaurant_name"])  # the "Tamarind"->"Cameron" fix
    for dish in re.split(r"[,;/]", call_ctx.get("candidate_dishes") or ""):
        if dish.strip():
            terms.append(dish.strip())
    for w in re.findall(r"[A-Za-z]{4,}", call_ctx.get("call_notes") or ""):
        if w[0].isupper() and w.lower() not in {"they", "their", "there", "then", "this", "that", "with"}:
            terms.append(w)  # proper nouns from the note (dish/restaurant names)
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:cap]


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
        # Haiku 4.5 = the latency sweet spot for real-time voice (~0.4s TTFB).
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        params=AnthropicLLMService.InputParams(enable_prompt_caching=True),
        # If a request hangs, retry once at 5s instead of leaving the bot silent
        # forever (silent-bot dead air is a real cause of the restaurant hanging up).
        retry_on_timeout=True,
    )

    @llm.event_handler("on_completion_timeout")
    async def on_completion_timeout(service):
        logger.warning("LLM completion timed out — retried")

    # STT: default to a telephony-tuned Deepgram model and BOOST the dietary
    # vocabulary so 8kHz phone audio doesn't mangle key words (e.g. hearing
    # "scallions" as "scallops"). Wrapped defensively: if the enhanced options
    # aren't accepted by this SDK build, fall back to the plain service so a
    # call never fails to start over an STT config detail.
    _dg_key = os.getenv("DEEPGRAM_API_KEY")
    # Boost THIS call's vocabulary: dietary terms + cuisine dish pack + the
    # restaurant's name + expected dishes (build_keyterms above). Without the
    # cuisine pack, 8kHz audio mangled dal->doll, idli->Italy, Tamarind->Cameron.
    _dg_terms = build_keyterms(call_ctx)
    logger.info(f"Deepgram keyterms ({len(_dg_terms)}): {_dg_terms}")
    # nova-3 (default; lower WER incl. 8kHz) boosts via keyterm prompting. nova-2*
    # uses keywords with ":N" intensities. Passing the wrong one for the model
    # makes Deepgram reject the connection, so branch on the model.
    _dg_model = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
    if _dg_model.startswith("nova-3"):
        _dg_boost = {"keyterm": _dg_terms}
    else:
        _dg_boost = {"keywords": [f"{t}:2" for t in _dg_terms]}
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
        # 1.0 = natural pacing. Slightly LOWER stability = more natural human
        # variation on a phone line (too-high stability reads as flat/robotic).
        params=ElevenLabsTTSService.InputParams(
            speed=float(os.getenv("ELEVENLABS_SPEED", "1.0")),
            stability=0.45,
            similarity_boost=0.75,
        ),
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
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
        from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
        from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        class FirstNSpeechesUserMuteStrategy(BaseUserMuteStrategy):
            """Mute the user while the bot speaks, for its first N utterances.

            pipecat's built-in FirstSpeech mutes only the greeting; the PURPOSE
            line (utterance #2) was left exposed, and the bot's own voice echoing
            off the restaurant's speakerphone got transcribed and 'interrupted'
            it. Muting the first two utterances covers greeting + purpose, while
            still letting the callee's 'Hello?'/'Yes?' through (bot is silent then)."""

            def __init__(self, n: int = 2):
                super().__init__()
                self._n = n
                self._completed = 0
                self._bot_speaking = False

            async def process_frame(self, frame: "Frame") -> bool:
                await super().process_frame(frame)
                if isinstance(frame, BotStartedSpeakingFrame):
                    self._bot_speaking = True
                elif isinstance(frame, BotStoppedSpeakingFrame):
                    self._bot_speaking = False
                    self._completed += 1
                return self._bot_speaking and self._completed < self._n

        user_params_kwargs["user_turn_strategies"] = UserTurnStrategies(
            # Interrupt the bot only on a FINALIZED (not interim) utterance of
            # >=5 words. The old bar (3 interim words) fired on echo fragments of
            # the bot's OWN speech and on brief noise — the "cut off for no reason"
            # problem. When the bot is silent this drops to 1 word, so normal
            # turn-taking speed is unaffected.
            start=[MinWordsUserTurnStartStrategy(min_words=5, use_interim=False)],
            # Smart-turn's default 3s max-silence fallback makes the bot sit silent
            # on a clipped answer; 1.5s halves that while still letting lists finish.
            stop=[TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=1.5))
            )],
        )
        user_params_kwargs["user_mute_strategies"] = [FirstNSpeechesUserMuteStrategy(n=2)]
        logger.info("Barge-in tuning active: MinWords(5, final-only) + FirstNSpeechMute(2) + SmartTurn(1.5s)")
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
