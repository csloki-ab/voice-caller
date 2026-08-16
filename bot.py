#
# voice-caller — Pipecat outbound dietary-inquiry bot
# Based on pipecat-examples/twilio-chatbot/outbound, customized to:
#   - swappable TTS via TTS_PROVIDER (deepgram Aura-2 is current; elevenlabs |
#     rime | cartesia also wired)
#   - use Anthropic Claude (Haiku 4.5) for the LLM
#   - load our tuned dietary-inquiry system prompt from prompt.txt
#   - inject per-call context (restaurant_name, call_notes, cuisine, ...) that
#     arrives as Twilio <Stream> parameters (see server_utils.generate_twiml)
#

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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

# The prompt (including the diet rules) is committed and public — the owner is
# fine publishing the dietary restrictions, and keeping the real rules in the
# repo means a deploy can never silently fall back to a generic diet.
with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "r") as f:
    PROMPT_TEMPLATE = f.read()

import json
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


# ---------------------------------------------------------------------------
# Echo gate. pipecat 1.7 has NO acoustic echo cancellation (audio/filters/ are
# noise suppressors — none take a far-end reference), so on a restaurant's
# speakerphone our own TTS comes back transcribed as USER speech ~0.5-1s after
# BotStoppedSpeaking — i.e. right after the mute strategy lifts. Those phantom
# turns interrupt the bot and (pre-fix) caused it to answer its own words.
# Fix at the TEXT level: BotTextTap records what the bot says; EchoGate drops
# user transcriptions that are near-verbatim repeats of it.
# ---------------------------------------------------------------------------
_ECHO_WINDOW_SECS = 12.0
_ECHO_OVERLAP = 0.9


def _norm_tokens(text: str) -> list:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if t]


class BotTextTap(FrameProcessor):
    """Sits after the LLM; records outgoing bot text (+timestamp) for EchoGate."""

    def __init__(self, recent_bot_text: deque, **kwargs):
        super().__init__(**kwargs)
        self._recent = recent_bot_text

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and frame.text.strip():
            self._recent.append((time.time(), frame.text))
        await self.push_frame(frame, direction)


class EchoGate(FrameProcessor):
    """Sits between STT and the user aggregator; drops echo transcriptions.

    A transcription is an echo if >=90% of its tokens appeared in the bot's own
    output within the last 12s. A single novel content word keeps it alive, so
    "no onion no garlic, correct" passes while a verbatim bounce-back is dropped.
    """

    def __init__(self, recent_bot_text: deque, **kwargs):
        super().__init__(**kwargs)
        self._recent = recent_bot_text

    def _is_echo(self, text: str) -> bool:
        toks = _norm_tokens(text)
        if not toks:
            return False
        now = time.time()
        bot_toks = set()
        for ts, t in self._recent:
            if now - ts <= _ECHO_WINDOW_SECS:
                bot_toks.update(_norm_tokens(t))
        if not bot_toks:
            return False
        overlap = sum(1 for t in toks if t in bot_toks)
        return overlap / len(toks) >= _ECHO_OVERLAP

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and self._is_echo(
            frame.text
        ):
            logger.info(f"ECHO GATE dropped ({type(frame).__name__}): {frame.text!r}")
            return
        await self.push_frame(frame, direction)


_ZERO_TEXT_RETRY_PROMPT = (
    "Your last reply had no words in it, so the caller heard silence. Respond now "
    "with what you should have said — one short, natural sentence continuing the call."
)


class FailsafeAnthropicLLMService(AnthropicLLMService):
    """Hard failsafe: a completed inference that produced ZERO text is re-run once.

    Silence is the single worst failure on a phone call — it ends the call every
    time (a restaurant asked "what are you looking for?", our bot went mute, and
    they hung up). Interrupted responses are exempt: the interrupting turn will
    produce the answer instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spoke_this_response = False
        self._interrupted_this_response = False
        self._zero_text_retried = False

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            self._interrupted_this_response = True
        await super().process_frame(frame, direction)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMFullResponseStartFrame):
            self._spoke_this_response = False
            self._interrupted_this_response = False
        elif isinstance(frame, LLMTextFrame) and frame.text.strip():
            self._spoke_this_response = True
            self._zero_text_retried = False  # real speech re-arms the failsafe
        maybe_retry = (
            isinstance(frame, LLMFullResponseEndFrame)
            and not self._spoke_this_response
            and not self._interrupted_this_response
            and not self._zero_text_retried
        )
        await super().push_frame(frame, direction)
        # Re-check after super(): late graceful-degradation pushes count as speech.
        if maybe_retry and not self._spoke_this_response:
            self._zero_text_retried = True  # at most one retry per silence
            logger.warning("FAILSAFE: LLM response had zero text — re-running inference once")
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "developer", "content": _ZERO_TEXT_RETRY_PROMPT}]
                )
            )
            await self.push_frame(LLMRunFrame())


async def run_bot(transport: BaseTransport, call_ctx: dict, handle_sigint: bool):
    system_prompt = build_system_prompt(call_ctx)
    logger.info(f"Calling {call_ctx.get('restaurant_name')!r} — prompt {len(system_prompt)} chars")

    llm = FailsafeAnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        # Haiku 4.5 = the latency sweet spot for real-time voice (~0.4s TTFB).
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        # The dietary prompt MUST be passed as system_instruction, NOT as a system
        # message in LLMContext. The LLM turn-completion strategy appends its own
        # marker instructions by recomposing settings.system_instruction from the
        # service's BASE instruction; if the base is empty (prompt only in context)
        # the composed system becomes the marker instructions ALONE, and the
        # Anthropic adapter then DISCARDS the context's system message entirely
        # (base_llm_adapter._resolve_system_instruction: system_instruction wins).
        # Net effect of getting this wrong: the bot calls a restaurant with no diet
        # prompt at all — a generic assistant. Verified against the 1.7.0 source.
        settings=AnthropicLLMService.Settings(
            system_instruction=system_prompt,
            # Must live HERE, not in params=: the constructor ignores params
            # entirely when settings is provided (anthropic/llm.py "if not settings").
            enable_prompt_caching=True,
        ),
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

    # TTS engine is swappable via TTS_PROVIDER (elevenlabs | deepgram | rime) so
    # we can A/B whole engines from Railway, no code change. ElevenLabs is a
    # content/narration engine (reads "AI" on a phone); deepgram (Aura-2) and rime
    # are purpose-built for phone conversation. Aura-2 reuses our DEEPGRAM_API_KEY.
    def _build_elevenlabs_tts():
        # Per-cuisine voice via ELEVENLABS_VOICE_MAP {"indian":"<id>","default":"<id>"}.
        _voice_map = {}
        try:
            _voice_map = json.loads(os.getenv("ELEVENLABS_VOICE_MAP", "{}")) or {}
        except Exception as e:
            logger.warning(f"Bad ELEVENLABS_VOICE_MAP ({e}); ignoring")
        _cuisine = (call_ctx.get("cuisine") or "").lower()
        _voice_id = (
            next((v for k, v in _voice_map.items() if k != "default" and k in _cuisine), None)
            or _voice_map.get("default")
            or os.getenv("ELEVENLABS_VOICE_ID")
        )
        return ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=_voice_id,
            model=os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            params=ElevenLabsTTSService.InputParams(
                speed=float(os.getenv("ELEVENLABS_SPEED", "1.0")),
                stability=float(os.getenv("ELEVENLABS_STABILITY", "0.40")),
                similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY", "0.75")),
                style=float(os.getenv("ELEVENLABS_STYLE", "0.35")),
            ),
        )

    def _build_deepgram_tts():
        from pipecat.services.deepgram.tts import DeepgramTTSService
        return DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),  # SAME key as our STT — no new signup
            voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en"),
            sample_rate=8000,
        )

    def _build_cartesia_tts():
        # Cartesia Sonic — the other phone-native engine (fastest time-to-first-audio,
        # which matters because uniform/laggy response timing is a bigger "this is a bot"
        # tell than voice timbre). Needs CARTESIA_API_KEY + a voice id from their library.
        from pipecat.services.cartesia.tts import CartesiaTTSService
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSService.Settings(
                voice=os.getenv("CARTESIA_VOICE_ID"),
                model=os.getenv("CARTESIA_MODEL", "sonic-3.5"),
            ),
            sample_rate=8000,
        )

    def _build_rime_tts():
        from pipecat.services.rime.tts import RimeTTSService
        return RimeTTSService(
            api_key=os.getenv("RIME_API_KEY"),
            voice_id=os.getenv("RIME_VOICE_ID", "luna"),
            model=os.getenv("RIME_MODEL", "arcana"),
            sample_rate=8000,
        )

    _tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
    _tts_builders = {
        "deepgram": _build_deepgram_tts,
        "rime": _build_rime_tts,
        "cartesia": _build_cartesia_tts,
    }
    try:
        tts = _tts_builders.get(_tts_provider, _build_elevenlabs_tts)()
        logger.info(f"TTS provider: {_tts_provider}")
    except Exception as e:
        logger.warning(f"TTS provider {_tts_provider!r} failed ({e}); falling back to ElevenLabs")
        tts = _build_elevenlabs_tts()

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

            def __init__(self, n: int = 2, turn_analyzer=None,
                         patient_stop_secs: float = 4.0):
                super().__init__()
                self._n = n
                self._completed = 0
                self._bot_speaking = False
                # The live smart-turn analyzer; we raise its silence threshold
                # once the call is actually underway (see below).
                self._turn_analyzer = turn_analyzer
                self._patient_stop_secs = patient_stop_secs

            async def process_frame(self, frame: "Frame") -> bool:
                await super().process_frame(frame)
                if isinstance(frame, BotStartedSpeakingFrame):
                    self._bot_speaking = True
                elif isinstance(frame, BotStoppedSpeakingFrame):
                    self._bot_speaking = False
                    self._completed += 1
                    # PHASE SWITCH — the whole point of this hook.
                    # A single stop_secs can't serve both ends of a call:
                    #   opening: they say "Hello?" and a busy restaurant hangs up
                    #            after ~3s of dead air, so we must answer FAST;
                    #   mid-call: they read a dish list with 2-3s gaps between
                    #            items, so we must wait PATIENTLY.
                    # We therefore open snappy and, as soon as the bot has
                    # completed its first utterance (i.e. the conversation is
                    # really underway and lists become possible), raise stop_secs
                    # to the patient value. Mutating the params object works
                    # because base_smart_turn reads params.stop_secs per call.
                    # NOTE: base_smart_turn caches `_stop_ms = stop_secs * 1000`
                    # in __init__ and compares against THAT (base_smart_turn.py
                    # :70 and :132) — mutating params.stop_secs alone is a no-op.
                    # We must set the cached threshold too. Verified against the
                    # 1.7.0 source; keep both in sync if you touch this.
                    if self._completed == 1 and self._turn_analyzer is not None:
                        try:
                            # Adaptive analyzer: the value that matters is the
                            # one applied to a CONFIDENTLY-incomplete prediction.
                            # Each prediction rewrites _stop_ms, so setting only
                            # that would be overwritten on the very next pause.
                            self._turn_analyzer._incomplete_stop_secs = self._patient_stop_secs
                            self._turn_analyzer._params.stop_secs = self._patient_stop_secs
                            self._turn_analyzer._stop_ms = self._patient_stop_secs * 1000
                            logger.info(
                                f"Turn-taking: switched to patient mode "
                                f"(stop_secs={self._patient_stop_secs})"
                            )
                        except Exception as e:
                            # Never let a tuning tweak kill a live call.
                            logger.warning(f"Patient-mode switch failed ({e}); staying snappy")
                return self._bot_speaking and self._completed < self._n

        # Snappy while the call opens, patient once it's underway (flipped by the
        # mute strategy below). Both ends env-tunable so we can retune without a
        # code change if live calls show we're still too eager or too slow.
        _opening_stop_secs = float(os.getenv("SMART_TURN_OPENING_STOP_SECS", "1.0"))
        _patient_stop_secs = float(os.getenv("SMART_TURN_PATIENT_STOP_SECS", "4.0"))
        # A SINGLE flat stop_secs is what made the bot feel robotic. smart-turn
        # runs its model ONCE per pause and never re-runs it, so whenever it
        # predicted "not done" (common on 8kHz narrowband) nothing could end the
        # turn until the full fallback expired — ~4.2s of dead air, on top of
        # LLM+TTS. Calls alternated ~2s and ~6s gaps; that machine-uniform
        # bimodal rhythm reads as a bot no matter how good the voice is. The bot
        # was literally saying "I'm still here" into silence it created.
        #
        # Fix: adapt the fallback to the model's OWN confidence, which is
        # recomputed on every pause, so this is per-pause adaptive for free.
        #   p >= complete_threshold : they're done -> answer now
        #   sure_floor <= p < thresh: probably done, narrowband just unsure
        #                             -> SHORT fallback (~1.2s)
        #   p <  sure_floor         : genuinely mid-sentence ("...and we have,
        #                             um—") -> keep the FULL patient fallback
        # Threshold drops 0.5 -> 0.35 to offset v3's narrowband pessimism.
        class AdaptiveStopSmartTurnV3(LocalSmartTurnAnalyzerV3):
            def __init__(self, *, complete_threshold, sure_floor,
                         unsure_stop_secs, incomplete_stop_secs, **kwargs):
                super().__init__(**kwargs)
                self._complete_threshold = complete_threshold
                self._sure_floor = sure_floor
                self._unsure_stop_secs = unsure_stop_secs
                self._incomplete_stop_secs = incomplete_stop_secs

            def clear(self):
                # Called by the stop strategy at EVERY turn end. Necessary because
                # a turn completed via the SILENCE TIMEOUT skips _predict_endpoint
                # entirely (append_audio clears the buffer first, so the follow-up
                # analyze_end_of_turn early-returns on an empty buffer). Without
                # this, _stop_ms would carry a stale per-pause value — e.g. a 1.2s
                # "unsure" threshold — into the next turn, and could clip someone
                # mid-list at 1.2s instead of the patient 4s. Reset to the
                # confidently-incomplete baseline, which the phase switch keeps
                # current (1.0s opening -> 4.0s once underway).
                super().clear()
                self._stop_ms = self._incomplete_stop_secs * 1000

            def _predict_endpoint(self, audio_array):
                result = super()._predict_endpoint(audio_array)
                p = result.get("probability", 0.0)
                if p >= self._complete_threshold:
                    result["prediction"] = 1
                else:
                    result["prediction"] = 0
                    secs = (
                        self._unsure_stop_secs
                        if p >= self._sure_floor
                        else self._incomplete_stop_secs
                    )
                    # Each prediction rewrites the live threshold the silence
                    # counter compares against (base_smart_turn caches it).
                    self._stop_ms = secs * 1000
                return result

        _turn_params = SmartTurnParams(stop_secs=_opening_stop_secs)
        _turn_analyzer = AdaptiveStopSmartTurnV3(
            params=_turn_params,
            complete_threshold=float(os.getenv("SMART_TURN_COMPLETE_THRESHOLD", "0.35")),
            sure_floor=float(os.getenv("SMART_TURN_SURE_FLOOR", "0.15")),
            unsure_stop_secs=float(os.getenv("SMART_TURN_UNSURE_STOP_SECS", "1.2")),
            # Stay snappy through the opening; the mute strategy raises this to
            # the patient value once the conversation is actually underway.
            incomplete_stop_secs=_opening_stop_secs,
        )

        user_params_kwargs["user_turn_strategies"] = UserTurnStrategies(
            # Bar to INTERRUPT the bot (only applies while the bot is speaking;
            # when it's silent the bar is 1 word, so normal pace is unaffected).
            # Was 5 FINALIZED words — far too high: a finalized Deepgram transcript
            # only lands on endpointing, so a manager trying to reclaim the floor
            # got steamrolled for seconds. 3 words on INTERIMS reacts in ~300-600ms
            # while still filtering backchannels ("yeah", "uh-huh") and echo blips.
            start=[MinWordsUserTurnStartStrategy(min_words=3, use_interim=True)],
            # END-OF-TURN: smart-turn v3 alone, finalizing directly.
            #
            # The LLM marker-gating chain (deferred(...) + LLMTurnCompletion...) is
            # GONE. It caused THREE distinct silence failures in production:
            #   1. it rebuilt system_instruction and DISCARDED our diet prompt (P0);
            #   2. it suppressed the greeting -> up to 12s of opening dead air;
            #   3. a bare "complete" marker set an internal voiced latch that
            #      silently swallowed the NEXT inference — the bot went mute exactly
            #      when a restaurant asked "what are you looking for?", and they hung
            #      up. That call's dish list was lost with it.
            # (2) and (3) are structural races inside pipecat's own mixin/controller,
            # not prompt or model problems — a better model would not fix them.
            # Mid-list patience is now carried by patient stop_secs (below) plus the
            # prompt's let-them-finish rules, and FailsafeAnthropicLLMService
            # guarantees a completed inference can never end in silence.
            #
            # stop_secs is the ceiling on how long we trust "not done yet" before
            # force-ending the turn. It STARTS at the snappy opening value and is
            # raised to the patient value by FirstNSpeechesUserMuteStrategy after the
            # bot's first utterance: opening at 4.0 made the bot sit silent for up to
            # 4s on "Hello?" (restaurants hung up), while 4.0 mid-call is what stops
            # it barging in between dishes.
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=_turn_analyzer)],
        )
        user_params_kwargs["user_mute_strategies"] = [
            FirstNSpeechesUserMuteStrategy(
                n=2, turn_analyzer=_turn_analyzer, patient_stop_secs=_patient_stop_secs
            )
        ]
        logger.info(
            "Turn-taking active: MinWords(3, interim) + FirstNSpeechMute(2) + "
            "SmartTurn(1.0s->4.0s) + EchoGate (LLM gating removed)"
        )
    except Exception as e:
        # Loud, not quiet: silently falling back to defaults is how we shipped
        # twitchy turn-taking before without noticing.
        logger.error(f"TURN-TAKING CONFIG FAILED ({e}) — running pipecat DEFAULTS (expect barge-in)")

    # NOTE: no system message here on purpose — the tuned prompt is passed to the
    # LLM service as system_instruction above (see the comment there). Putting it
    # in the context too would just get discarded by the Anthropic adapter.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(**user_params_kwargs),
    )

    # Shared buffer: BotTextTap writes what the bot says, EchoGate reads it to
    # recognise our own voice coming back off the restaurant's speakerphone.
    recent_bot_text = deque(maxlen=50)

    pipeline = Pipeline(
        [
            transport.input(),            # audio in from Twilio
            stt,                          # speech -> text (Deepgram)
            EchoGate(recent_bot_text),    # drop our own TTS echoing back (no AEC)
            user_aggregator,
            llm,                          # Anthropic Claude (our tuned prompt)
            BotTextTap(recent_bot_text),  # record bot lines for the EchoGate
            tts,                          # text -> speech
            transport.output(),           # audio out to Twilio
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
        # Cancel FIRST, then dump. The cancel drives a CancelFrame through the user
        # aggregator, which flushes any still-buffered end-of-turn user speech into
        # the context. Dumping before the cancel silently lost whatever the caller
        # said last — that's how a restaurant's "we also have vegetable korma...
        # what are you looking for?" was audible on the recording but missing from
        # our transcript. The context object stays readable after cancel.
        try:
            await worker.cancel()
        except Exception as e:
            logger.warning(f"Worker cancel before transcript dump failed ({e})")

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
                # Strip the LLM turn-completion markers (never spoken aloud, but
                # they persist in context, e.g. "✓ Hi there!" / a bare "○").
                raw = str(content)
                content = raw.lstrip("✓○◐ ").strip()
                if not content:
                    # A bare incomplete marker = the LLM SUPPRESSED this response and
                    # the caller heard SILENCE. Previously we skipped these, which made
                    # opening dead-air invisible in the transcript (a call showing only
                    # 'USER: Hello?' looked like an instant hangup). Log it explicitly.
                    marker = raw.strip()[:1]
                    logger.info(f"TRANSCRIPT | {str(role).upper()}: [suppressed {marker} — bot stayed silent]")
                    continue
                logger.info(f"TRANSCRIPT | {str(role).upper()}: {content}")
        except Exception as e:
            logger.warning(f"Transcript dump failed ({e})")

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
