#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""server.py

Webhook server to handle outbound call requests, initiate calls via Twilio API,
and handle subsequent WebSocket connections for Media Streams.
"""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
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
    from pipecat.turns.user_stop.deferred_user_turn_stop_strategy import deferred
    from pipecat.turns.user_stop.llm_turn_completion_user_turn_stop_strategy import (
        LLMTurnCompletionUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
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
            deferred(TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=_STA(params=SmartTurnParams(stop_secs=_patient))
            )),
            LLMTurnCompletionUserTurnStopStrategy(
                config=UserTurnCompletionConfig(
                    incomplete_short_timeout=6.0, incomplete_long_timeout=12.0
                )
            ),
        ],
    )
    logger.info("Turn-taking preflight OK (deferred smart-turn + LLM completion gating)")


@app.post("/dialout", response_model=DialoutResponse)
async def handle_dialout_request(request: Request) -> DialoutResponse:
    """Handle outbound call request and initiate call via Twilio.

    Args:
        request (Request): FastAPI request containing JSON with 'to_number' and 'from_number'.

    Returns:
        DialoutResponse: Response containing call_sid, status, and to_number.

    Raises:
        HTTPException: If request data is invalid or missing required fields.
    """
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
