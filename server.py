#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""server.py

Webhook server to handle outbound call requests, initiate calls via Twilio API,
and handle subsequent WebSocket connections for Media Streams.
"""

import hmac
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from server_utils import (
    DialoutResponse,
    dialout_request_from_request,
    generate_twiml,
    make_twilio_call,
    parse_twiml_request,
)

load_dotenv(override=True)


# Fail loudly at boot if a required secret is blank. Without this a missing key
# doesn't crash the server — the call connects and then goes silent mid-call
# (exactly the failure we're migrating away from), which is very hard to debug.
REQUIRED_ENV = [
    "DEEPGRAM_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM_NUMBER",
    "LOCAL_SERVER_URL",
]
_missing = [k for k in REQUIRED_ENV if not (os.getenv(k) or "").strip()]
if _missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(_missing)
        + ". Set them in Railway → Variables before deploying."
    )


app = FastAPI()

# Set true only once the boot preflight has fully passed (see _preload_models).
_PREFLIGHT_OK = False


@app.get("/health")
async def health():
    """Which build is actually live, and did its preflight pass?

    Added because a push, a green 200, and the fix being live are three
    different things: Railway kept serving the previous build while the new
    one was still compiling, so a 200 ninety seconds after `git push` proved
    nothing. Two scheduled unattended calls were about to depend on a DTMF
    fix I had no way to confirm had shipped.

    Railway injects RAILWAY_GIT_COMMIT_SHA at build time, so this reports the
    exact commit serving traffic. Deliberately exposes NO secrets and no env
    values — just a short SHA and two booleans.
    """
    sha = (os.getenv("RAILWAY_GIT_COMMIT_SHA") or "").strip()
    return {
        "ok": True,
        "commit": sha[:7] or "unknown",
        "preflight_ok": _PREFLIGHT_OK,
        "tts_provider": (os.getenv("TTS_PROVIDER") or "").strip() or "unset",
    }


# Import the bot + pipecat runner eagerly at boot (they were lazy-imported inside
# the websocket handler, so the first call paid the whole import chain mid-call).
from pipecat.runner.types import WebSocketRunnerArguments  # noqa: E402
from bot import bot  # noqa: E402


@app.on_event("startup")
async def _preload_models():
    """Warm the ONNX models (Silero VAD + smart-turn) into memory/page-cache at
    boot. Otherwise the FIRST call spends 1-2s loading them AFTER the restaurant
    has already answered — the 2.3s cold-start we measured. Throwaway instances;
    we build fresh per-call ones (they hold per-stream state) but the files and
    onnxruntime are now warm, so per-call construction is ~100-200ms."""
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        SileroVADAnalyzer()
        LocalSmartTurnAnalyzerV3()
        logger.info("Preloaded Silero VAD + smart-turn models at boot")
    except Exception as e:
        logger.warning(f"Model preload skipped ({e})")

    # Preflight the turn-taking chain AT BOOT. bot.py wraps this same config in a
    # try/except and falls back to pipecat defaults on failure — which is exactly
    # the twitchy barge-in behavior we just fixed. For unattended scheduled calls
    # we want that failure to surface as a crashed Railway deploy, not as eight
    # bad live calls to real restaurants. Deliberately NOT caught.
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3 as _STA

    # Parse the SAME env vars bot.py does, UNCAUGHT: a malformed Railway value
    # would otherwise be swallowed by bot.py's per-call try/except and silently
    # drop every live call to pipecat's default (twitchy) turn-taking.
    _opening = float(os.getenv("SMART_TURN_OPENING_STOP_SECS", "1.0"))
    _patient = float(os.getenv("SMART_TURN_PATIENT_STOP_SECS", "4.0"))

    # bot.py flips the analyzer from snappy->patient mid-call by writing these
    # PRIVATE attrs (no public API exists in 1.7.0). If a pipecat upgrade renames
    # them the flip becomes a silent no-op, so assert them at deploy time.
    _sta_probe = _STA(params=SmartTurnParams(stop_secs=_opening))
    assert hasattr(_sta_probe, "_stop_ms") and hasattr(_sta_probe, "_params"), (
        "smart-turn internals renamed — bot.py's snappy->patient switch would silently no-op"
    )
    assert _sta_probe._stop_ms == _opening * 1000, "smart-turn no longer caches _stop_ms as expected"

    UserTurnStrategies(
        start=[MinWordsUserTurnStartStrategy(min_words=3, use_interim=True)],
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=_STA(params=SmartTurnParams(stop_secs=_patient))
            )
        ],
    )

    # The echo gate and the zero-text failsafe are now load-bearing (no LLM gating
    # to fall back on), so fail the deploy if either can't be constructed.
    from bot import EchoGate, BotTextTap, FailsafeAnthropicLLMService  # noqa: F401
    from collections import deque as _deque
    import time as _time

    _probe = _deque(maxlen=4)
    # NOTE: must be a CURRENT timestamp — EchoGate only considers bot text from
    # the last 12s, so a 0.0 timestamp reads as ancient and matches nothing.
    _probe.append((_time.time(), "is this the restaurant"))
    assert EchoGate(_probe)._is_echo("is this the restaurant"), "EchoGate no longer detects echo"
    assert not EchoGate(_probe)._is_echo("we have chana masala and dal"), "EchoGate too aggressive"

    # DTMF preflight. Spotted Dog was lost three times to a keypad-only
    # switchboard, and the last attempt fired the tone correctly yet the IVR
    # answered "We have not received a valid response" on a loop. The tone is
    # in-band AUDIO over 8kHz mulaw, so if the bundled wav asset ever goes
    # missing or resamples to silence, the press becomes a no-op that looks
    # perfectly healthy in the logs. Assert the whole chain at deploy time.
    from bot import IVRKeypadPresser  # noqa: F401
    from pipecat.frames.frames import OutputDTMFFrame
    from pipecat.audio.dtmf.utils import load_dtmf_audio

    # Whole-word matching: a human saying "no pressure" must never trigger a beep.
    assert IVRKeypadPresser._PRESS_RE.search("for the spotted dog, press 2"), (
        "IVR press cue no longer matches a real menu line"
    )
    assert not IVRKeypadPresser._PRESS_RE.search("sure, no pressure at all"), (
        "IVR press cue would fire at a human saying 'pressure'"
    )
    # The exact string Flanigan's switchboard loops on. If this stops matching,
    # the retry never fires and we're back to pressing once and going mute.
    _rejection = "we have not received a valid response. please try again."
    assert any(c in _rejection for c in IVRKeypadPresser._RETRY_CUES), (
        "IVR retry cues no longer match the observed rejection prompt"
    )

    _dtmf_frame = OutputDTMFFrame.from_string("2")
    assert _dtmf_frame.buttons, "OutputDTMFFrame.from_string produced no buttons"
    for _button in _dtmf_frame.buttons:
        _tone = await load_dtmf_audio(_button, sample_rate=8000)
        # 16-bit @ 8kHz: 40ms (the DTMF spec minimum) = 640 bytes. Anything
        # shorter than that and no switchboard on earth will register the press.
        assert len(_tone) >= 640, f"DTMF tone for {_button} is too short ({len(_tone)} bytes)"
        assert any(_tone), f"DTMF tone for {_button} is all silence"

    logger.info("DTMF preflight OK (press/retry cues match; 8kHz tone renders audible)")

    # Transcript write-back preflight. The bot POSTs each finished transcript to
    # the Cloudflare Worker so it lands in D1 — because Railway's log stream was
    # the only copy, and when this dashboard went down a completed call became
    # unreadable. Two ways that could silently no-op, both cheap to assert:
    #   1. row_id not forwarded to the bot. EXACTLY the bug ivr_digit had — the
    #      field existed on the request and never reached the <Stream> params, so
    #      the feature did nothing and looked fine.
    #   2. aiohttp missing, so every POST raises into a warning nobody reads.
    from bot import _persist_transcript  # noqa: F401
    from server_utils import CONTEXT_FIELDS, DialoutRequest
    import aiohttp  # noqa: F401

    assert "row_id" in CONTEXT_FIELDS, (
        "row_id is not forwarded to the bot — transcript write-back would silently no-op"
    )
    assert hasattr(DialoutRequest, "model_fields") and "row_id" in DialoutRequest.model_fields, (
        "DialoutRequest has no row_id field — the Worker's row_id would be dropped"
    )

    if (os.getenv("TRANSCRIPT_SINK_URL") or "").strip():
        logger.info("Transcript write-back preflight OK (sink configured)")
    else:
        logger.warning(
            "TRANSCRIPT_SINK_URL is unset — transcripts will live ONLY in these logs. "
            "Set it to the Worker's /transcript URL (plus TRANSCRIPT_SINK_SECRET) to save them to D1."
        )

    global _PREFLIGHT_OK
    _PREFLIGHT_OK = True

    logger.info("Turn-taking preflight OK (smart-turn only; LLM gating removed; echo gate armed)")


def _check_dialout_auth(request: Request) -> None:
    """Reject unauthenticated /dialout calls.

    This endpoint SPENDS MONEY and dials arbitrary phone numbers, so leaving it
    open means anyone who learns the service URL can place calls billed to our
    Twilio account (and make our number the origin of unsolicited calls).

    Enforced only when DIALOUT_SECRET is set, so the secret can be rolled out to
    the caller (the Cloudflare Worker) without a window where live calls break.
    Once set on both sides it is mandatory. Compared in constant time.
    """
    expected = (os.getenv("DIALOUT_SECRET") or "").strip()
    if not expected:
        logger.warning(
            "DIALOUT_SECRET is not set — /dialout is UNAUTHENTICATED and anyone "
            "who knows this URL can place calls on our Twilio account. Set it."
        )
        return

    provided = (
        request.headers.get("x-dialout-secret")
        or request.query_params.get("s")
        or ""
    ).strip()
    if not hmac.compare_digest(provided, expected):
        logger.warning("Rejected /dialout request with a missing or bad secret")
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/dialout", response_model=DialoutResponse)
async def handle_dialout_request(request: Request) -> DialoutResponse:
    """Handle outbound call request and initiate call via Twilio.

    Args:
        request (Request): FastAPI request containing JSON with 'to_number' and 'from_number'.

    Returns:
        DialoutResponse: Response containing call_sid, status, and to_number.

    Raises:
        HTTPException: If the shared secret is missing/wrong, or request data is invalid.
    """
    _check_dialout_auth(request)

    logger.info("Received outbound call request")

    dialout_request = await dialout_request_from_request(request)

    call_result = await make_twilio_call(dialout_request)

    return DialoutResponse(
        call_sid=call_result.call_sid,
        status="call_initiated",
        to_number=call_result.to_number,
    )


@app.post("/twiml")
async def get_twiml(request: Request) -> HTMLResponse:
    """Return TwiML instructions for connecting call to WebSocket.

    This endpoint is called by Twilio when a call is initiated. It returns TwiML
    that instructs Twilio to connect the call to our WebSocket endpoint with
    stream parameters containing call metadata.

    Args:
        request (Request): FastAPI request containing Twilio form data with 'To' and 'From'.

    Returns:
        HTMLResponse: TwiML XML response with Stream connection instructions.
    """
    logger.info("Serving TwiML for outbound call")

    twiml_request = await parse_twiml_request(request)

    twiml_content = generate_twiml(twiml_request)

    return HTMLResponse(content=twiml_content, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio Media Streams.

    This endpoint receives the WebSocket connection from Twilio's Media Streams
    and runs the bot to handle the voice conversation. Stream parameters passed
    from TwiML are available to the bot for customization.

    Args:
        websocket (WebSocket): FastAPI WebSocket connection from Twilio.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted for outbound call")

    try:
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args)
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
        await websocket.close()


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting Twilio outbound chatbot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
