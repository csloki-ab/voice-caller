#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from fastapi import HTTPException, Request
from loguru import logger
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse


class DialoutRequest(BaseModel):
    """Request data for initiating a dial-out call.

    Add any custom data here needed for the call. For example,
    you may add customer information, campaign data, or call context.

    Attributes:
        to_number (str): The phone number to dial (E.164 format recommended).
        from_number (str): The Twilio phone number to call from (E.164 format).
    """

    to_number: str
    # Optional: defaults to the server's TWILIO_FROM_NUMBER env in make_twilio_call
    # so schedulers (the Cloudflare cron) don't have to duplicate the Twilio number.
    from_number: str = ""
    # Per-call dietary context injected into the tuned prompt (all optional):
    restaurant_name: str = ""
    call_notes: str = ""
    cuisine: str = ""
    candidate_dishes: str = ""
    visit_date: str = ""
    party_size: str = ""


# The context fields we forward from /dialout -> TwiML query params -> <Stream> params -> the bot.
CONTEXT_FIELDS = ["restaurant_name", "call_notes", "cuisine", "candidate_dishes", "visit_date", "party_size"]


class TwilioCallResult(BaseModel):
    """Result of a Twilio call.

    Attributes:
        call_sid (str): The unique call SID of the initiated call.
        to_number (str): The phone number that was dialed.
    """

    call_sid: str
    to_number: str


class DialoutResponse(BaseModel):
    """Response from the dialout endpoint.

    Attributes:
        call_sid (str): The unique call SID of the initiated call.
        status (str): The status of the call initiation (e.g., "call_initiated").
        to_number (str): The phone number that was dialed.
    """

    call_sid: str
    status: str
    to_number: str


class TwimlRequest(BaseModel):
    """Request data for generating TwiML.

    Attributes:
        to_number (str): The phone number being called.
        from_number (str): The phone number calling from.
    """

    to_number: str
    from_number: str
    context: dict = {}


async def dialout_request_from_request(request: Request) -> DialoutRequest:
    """Parse and validate dial-out request data.

    Args:
        request (Request): FastAPI request object containing JSON with dial-out data.

    Returns:
        DialoutRequest: Parsed and validated dial-out request.

    Raises:
        HTTPException: If required fields are missing or request data is invalid.
    """
    data = await request.json()
    try:
        return DialoutRequest.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")


async def make_twilio_call(dialout_request: DialoutRequest) -> TwilioCallResult:
    """Initiate an outbound call via Twilio API.

    Creates a Twilio call that will request TwiML from the /twiml endpoint,
    which then connects the call to the WebSocket endpoint for bot handling.

    Args:
        dialout_request (DialoutRequest): Object containing call details including
            to_number and from_number.

    Returns:
        TwilioCallResult: Result containing the call SID and destination number.

    Raises:
        ValueError: If required environment variables are missing.
    """
    to_number = dialout_request.to_number
    # Fall back to the server's own Twilio number when the caller didn't supply one.
    from_number = (dialout_request.from_number or os.getenv("TWILIO_FROM_NUMBER") or "").strip()
    if not from_number:
        raise ValueError("Missing from_number (and TWILIO_FROM_NUMBER is unset)")

    local_server_url = (os.getenv("LOCAL_SERVER_URL") or "").strip().rstrip("/")
    if not local_server_url:
        raise ValueError("Missing LOCAL_SERVER_URL")

    # Forward the dietary context to /twiml as query params (Twilio preserves them
    # on its POST), so generate_twiml can attach them as <Stream> parameters.
    from urllib.parse import urlencode

    ctx = {f: getattr(dialout_request, f, "") for f in CONTEXT_FIELDS if getattr(dialout_request, f, "")}
    qs = ("?" + urlencode(ctx)) if ctx else ""
    twiml_url = f"{local_server_url}/twiml{qs}"
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        raise ValueError("Missing Twilio credentials")

    # Create Twilio client and make the call.
    # record=True captures BOTH sides of the call (dual channel = agent on one
    # track, restaurant on the other) so we keep an audio record of every call —
    # essential for holding a restaurant's answer, including a "no". Recordings
    # live in Twilio (Console → Monitor → Recordings) and are retrievable by API.
    # Toggle off with CALL_RECORDING=off if ever needed.
    record = os.getenv("CALL_RECORDING", "on").lower() != "off"
    client = TwilioClient(account_sid, auth_token)
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=twiml_url,
        method="POST",
        record=record,
        recording_channels="dual",
        # Hard backstop so a silent/voicemail line can't sit open running up cost.
        # Was 360s (6 min) but a patient real call — restaurant reading a full
        # curated list slowly — hit that cap mid-conversation and got cut off.
        # 600s (10 min) leaves generous room; genuine calls still finish under it.
        # Raised 600 -> 1200: the cap became the BINDING constraint on good calls,
        # twice. King David's ran a productive build-your-own conversation and was
        # cut off at exactly 600s mid-discussion; Crossroads hit the old 360s the
        # same way. A genuinely stuck/voicemail line still can't run past 20 min.
        time_limit=int(os.getenv("CALL_TIME_LIMIT", "1200")),
    )

    return TwilioCallResult(call_sid=call.sid, to_number=to_number)


async def parse_twiml_request(request: Request) -> TwimlRequest:
    """Parse and validate TwiML request data from Twilio.

    Twilio sends webhook data as form-encoded data, not JSON. This function
    extracts the 'To' and 'From' phone numbers from the form data.

    Args:
        request (Request): FastAPI request object containing Twilio form data.

    Returns:
        TwimlRequest: Parsed TwiML request with phone number metadata.
    """
    # Twilio sends form data, not JSON
    form_data = await request.form()
    to_number = form_data.get("To")
    from_number = form_data.get("From")

    # Our dietary context was passed as query params on the /twiml URL.
    context = {f: request.query_params.get(f, "") for f in CONTEXT_FIELDS if request.query_params.get(f)}

    return TwimlRequest(to_number=to_number, from_number=from_number, context=context)


def get_websocket_url() -> str:
    """Build the WebSocket URL Twilio Media Streams connects back to.

    We always self-host (Railway), so this is derived directly from
    LOCAL_SERVER_URL — we never route to Pipecat Cloud. Robust to a
    trailing slash and to an http:// (vs https://) scheme.

    Returns:
        str: WebSocket URL (wss://) for Twilio Media Streams to connect to.

    Raises:
        ValueError: If LOCAL_SERVER_URL is missing/blank.
    """
    local_server_url = (os.getenv("LOCAL_SERVER_URL") or "").strip().rstrip("/")
    if not local_server_url:
        raise ValueError("Missing LOCAL_SERVER_URL")
    # Convert https:// (or http://) to wss:// so Twilio streams over TLS.
    ws_url = local_server_url.replace("https://", "wss://").replace("http://", "wss://")
    return f"{ws_url}/ws"


def generate_twiml(twiml_request: TwimlRequest) -> str:
    """Generate TwiML response with WebSocket Stream connection.

    Creates TwiML that instructs Twilio to connect the call to our WebSocket
    endpoint. Call metadata (to_number, from_number) is passed as stream
    parameters, making them available to the bot for customization.

    Args:
        twiml_request (TwimlRequest): Request containing call metadata (phone numbers).

    Returns:
        str: TwiML XML string with Stream connection and parameters.
    """
    websocket_url = get_websocket_url()
    logger.debug(f"Generating TwiML with WebSocket URL: {websocket_url}")

    # Create TwiML response
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=websocket_url)

    # Add call metadata as stream parameters so the bot can access them
    # These will be available in the WebSocket 'start' message
    stream.parameter(name="to_number", value=twiml_request.to_number)
    stream.parameter(name="from_number", value=twiml_request.from_number)

    # Attach the dietary context so the bot can inject it into the tuned prompt.
    for name, value in (twiml_request.context or {}).items():
        stream.parameter(name=name, value=value)

    # Add Pipecat Cloud service host for production
    if os.getenv("ENV") == "production":
        agent_name = os.getenv("AGENT_NAME")
        org_name = os.getenv("ORGANIZATION_NAME")
        service_host = f"{agent_name}.{org_name}"
        stream.parameter(name="_pipecatCloudServiceHost", value=service_host)

    connect.append(stream)
    response.append(connect)
    response.pause(length=20)

    return str(response)
