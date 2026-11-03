#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import hashlib
import hmac
import os
import re
import time
from typing import Optional

from fastapi import HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
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
    # No from_number field on purpose: we always call from TWILIO_FROM_NUMBER.
    # Letting a request pick the caller ID was an abuse vector and nothing used it.
    # Which prompt the bot runs. "dietary" (default) is prompt.txt, exactly as
    # before; "general" is prompt_general.txt, for calling any business with a
    # goal written by the requester. A request that omits mode is dietary, so
    # every existing caller keeps today's behaviour.
    mode: str = Field("dietary", pattern="^(dietary|general)$")
    # Per-call dietary context injected into the tuned prompt (all optional):
    # Length caps: these go into the prompt and onto the /twiml URL.
    # restaurant_name is ALSO the business name in general mode. One field rather
    # than two, so the row label stripping (bot.spoken_name), the keyterm boost
    # and the transcript write-back all keep working unchanged in both modes.
    restaurant_name: str = Field("", max_length=200)
    call_notes: str = Field("", max_length=3000)
    cuisine: str = Field("", max_length=200)
    candidate_dishes: str = Field("", max_length=1500)
    visit_date: str = Field("", max_length=40)
    party_size: str = Field("", max_length=20)
    # General mode only: what the bot should ask or do, verbatim from the
    # requester (the owner approved this exact text before anything dialled).
    call_goal: str = Field("", max_length=2000)
    # General mode only: a number the bot may give if asked "how can they reach
    # you?". Blank = the bot says the person will follow up, never invents one.
    callback_number: str = Field("", max_length=20)
    # Who the bot says it is calling for, from the Worker's CALLER_DISPLAY_NAME
    # var. The Worker sets it on every call and no requester can, so an agent
    # cannot make the bot claim to call for somebody else. First name only.
    caller_name: str = Field("", max_length=40, pattern=r"^([A-Za-z][A-Za-z .'-]{0,39})?$")
    # Keypad digits to press at a switchboard/IVR menu (e.g. "2" for
    # "For the Spotted Dog, press 2"). Empty for normal direct lines.
    ivr_digit: str = Field("", max_length=4, pattern=r"^[0-9*#w]*$")
    ivr_digit_2: str = Field("", max_length=4, pattern=r"^[0-9*#w]*$")
    # D1 call_log row this call belongs to, so the bot can write its finished
    # transcript back (see _persist_transcript in bot.py). Blank = no write-back.
    row_id: str = Field("", max_length=12, pattern=r"^\d*$")
    # Validate everything (auth, target, TwiML URL) but don't actually dial.
    # Lets a post-deploy smoke test prove the whole path with zero risk.
    dry_run: bool = False

    @field_validator("to_number")
    @classmethod
    def _check_target(cls, v: str) -> str:
        return check_dial_target(v)

    @field_validator("callback_number")
    @classmethod
    def _check_callback(cls, v: str) -> str:
        # The bot reads this aloud to a stranger, so it must be a real E.164
        # number and nothing else (no "call me at ... or email ...").
        n = re.sub(r"[\s().-]", "", v or "")
        if n and not re.fullmatch(r"\+[1-9]\d{7,14}", n):
            raise ValueError("callback_number must be E.164, e.g. +14155550123")
        return n

    @model_validator(mode="after")
    def _general_needs_goal(self):
        # A general call with no goal would ring a business and have nothing to say.
        if self.mode == "general" and not self.call_goal.strip():
            raise ValueError("call_goal is required when mode is general")
        return self


# Where we are allowed to call. Default is US/Canada (+1) only. Toll fraud runs
# on international and premium-rate numbers, and every restaurant we have ever
# called is +1. Add e.g. "+52" here (Railway variable ALLOWED_DIAL_PREFIXES,
# comma-separated) only while a Mexico batch is live.
_NANP_RE = re.compile(r"^\+1[2-9]\d{2}[2-9]\d{6}$")
# NANP area codes that are premium-rate, personal or otherwise never a restaurant.
_BLOCKED_NANP = {"900", "976", "500", "521", "522", "533", "544", "566", "577", "588"}
# +1 is not just the US: these are Caribbean and Atlantic countries that share
# it, and several (Jamaica 876, Dominican Republic 809/829/849) are classic
# toll-fraud destinations. Canada stays allowed.
_BLOCKED_NANP |= {"242", "246", "264", "268", "284", "345", "441", "473", "649", "658", "664",
                  "721", "758", "767", "784", "809", "829", "849", "868", "869", "876"}


def check_dial_target(number: str) -> str:
    n = re.sub(r"[\s().-]", "", number or "")
    allowed = [p.strip() for p in (os.getenv("ALLOWED_DIAL_PREFIXES") or "+1").split(",") if p.strip()]
    if not any(n.startswith(p) for p in allowed):
        raise ValueError("number is outside the allowed countries")
    if n.startswith("+1"):
        if not _NANP_RE.match(n):
            raise ValueError("not a valid US/Canada number")
        if n[2:5] in _BLOCKED_NANP:
            raise ValueError("premium-rate or non-geographic number")
    elif not re.fullmatch(r"\+\d{8,15}", n):
        raise ValueError("not a valid E.164 number")
    return n


# --- Per-call signature ------------------------------------------------------
# /twiml and /ws are fetched by Twilio, not by us, so they cannot carry the
# DIALOUT_SECRET header. Instead /dialout signs (number, row, time) with the
# secret and puts the signature on the /twiml URL and the <Stream> parameters.
# The call's context (goal, notes, ...) is not signed itself: it never leaves
# this process, and is looked up BY the signature (see the context store
# below), so a valid signature can only ever reach the context /dialout
# validated. In effect the signature covers the goal too.
# A valid signature proves the call was created by our own /dialout, so a
# stranger who finds the URLs gets a 403 / a closed socket, not a free
# STT+LLM+TTS session. This needs no Twilio auth token on our side.
SIG_MAX_AGE_SECS = int(os.getenv("CALL_SIG_MAX_AGE_SECS", "1800"))


def _dialout_secret() -> str:
    return (os.getenv("DIALOUT_SECRET") or "").strip()


def sign_call(to_number: str, row_id: str, ts: str) -> str:
    msg = f"{to_number}|{row_id}|{ts}".encode()
    return hmac.new(_dialout_secret().encode(), msg, hashlib.sha256).hexdigest()


def verify_call(to_number: str, row_id: str, ts: str, sig: str) -> bool:
    if not (_dialout_secret() and ts and sig):
        return False
    try:
        age = time.time() - int(ts)
    except ValueError:
        return False
    # The websocket opens when the callee answers, which can be a while after
    # dialing (ringing + a phone menu), hence the generous window.
    if age < -60 or age > SIG_MAX_AGE_SECS:
        return False
    return hmac.compare_digest(sign_call(to_number, row_id, ts), sig)


# The context fields /dialout keeps for the bot (see the context store below).
CONTEXT_FIELDS = ["restaurant_name", "call_notes", "cuisine", "candidate_dishes", "visit_date", "party_size",
                  "ivr_digit", "ivr_digit_2", "row_id", "mode", "call_goal", "callback_number", "caller_name"]
# Only the dietary prompt reads these. Dropped from general calls so they can't
# leak diet context into the bot.
DIETARY_ONLY_FIELDS = {"cuisine", "candidate_dishes", "visit_date", "party_size"}

# What the callee hears if the call's context has gone missing. Said by Twilio
# at /twiml, or by the bot if it is lost between /twiml and /ws.
MISSING_CONTEXT_LINE = "Sorry, something went wrong on my end. Apologies for the trouble. Goodbye."


# --- Per-call context store --------------------------------------------------
# The context used to ride on the /twiml query string and then on the <Stream>
# parameters. A 2000-character goal plus 3000 characters of notes, percent-
# encoded, is far more than that path will carry, and a goal cut off halfway is
# a bot phoning a business with half its instructions. So /dialout keeps the
# validated context HERE, keyed by the call signature, and the URL and the
# <Stream> carry only row_id, ts and sig.
#
# ASSUMES A SINGLE RAILWAY REPLICA. This is plain process memory: with two
# replicas, Twilio's /twiml or /ws could land on the one that never saw the
# /dialout. It works today because server.py imports bot, so /dialout, /twiml
# and /ws all run in one process and share this module. If the container
# restarts between dialling and the answer, the context is gone and the callee
# hears MISSING_CONTEXT_LINE instead of a call with no instructions.
_CALL_CONTEXTS: dict = {}


def _prune_call_contexts(now: float) -> None:
    for sig in [s for s, (expires, _) in _CALL_CONTEXTS.items() if expires < now]:
        del _CALL_CONTEXTS[sig]


def store_call_context(sig: str, ts: str, ctx: dict, now: Optional[float] = None) -> None:
    """Keep ctx until the signature itself expires (ts + SIG_MAX_AGE_SECS)."""
    now = time.time() if now is None else now
    _prune_call_contexts(now)
    _CALL_CONTEXTS[sig] = (int(ts) + SIG_MAX_AGE_SECS, dict(ctx))


def get_call_context(sig: str, now: Optional[float] = None) -> Optional[dict]:
    """A copy of the context stored for sig, or None if unknown or expired."""
    now = time.time() if now is None else now
    _prune_call_contexts(now)
    entry = _CALL_CONTEXTS.get(sig or "")
    return dict(entry[1]) if entry else None


def drop_call_context(sig: str) -> None:
    _CALL_CONTEXTS.pop(sig or "", None)


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
    row_id: str = ""
    ts: str = ""
    sig: str = ""
    # None = the signature was valid but its context is gone (see the store).
    context: Optional[dict] = None


async def dialout_request_from_request(request: Request) -> DialoutRequest:
    """Parse and validate dial-out request data.

    Args:
        request (Request): FastAPI request object containing JSON with dial-out data.

    Returns:
        DialoutRequest: Parsed and validated dial-out request.

    Raises:
        HTTPException: If required fields are missing or request data is invalid.
    """
    try:
        data = await request.json()
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
    from_number = (os.getenv("TWILIO_FROM_NUMBER") or "").strip()
    if not from_number:
        raise ValueError("TWILIO_FROM_NUMBER is unset")

    local_server_url = (os.getenv("LOCAL_SERVER_URL") or "").strip().rstrip("/")
    if not local_server_url:
        raise ValueError("Missing LOCAL_SERVER_URL")

    # The context stays in this process (see the context store); the /twiml URL
    # carries only what is needed to find it again and prove it is ours.
    from urllib.parse import urlencode

    skip = DIETARY_ONLY_FIELDS if dialout_request.mode == "general" else set()
    ctx = {
        f: getattr(dialout_request, f, "")
        for f in CONTEXT_FIELDS
        if f not in skip and getattr(dialout_request, f, "")
    }
    ctx["to_number"] = to_number
    ts = str(int(time.time()))
    sig = sign_call(to_number, dialout_request.row_id, ts)
    twiml_url = f"{local_server_url}/twiml?{urlencode({'row_id': dialout_request.row_id, 'ts': ts, 'sig': sig})}"

    if dialout_request.dry_run:
        logger.info("dry_run: validated a call request without dialing")
        return TwilioCallResult(call_sid="dry-run", to_number=to_number)
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
    if dialout_request.mode == "general":
        # General calls go to any business in any state, and two-party-consent
        # states (California, Washington, Florida, Illinois, ...) need EVERY
        # party's consent to record. The bot does not announce a recording, so
        # these calls are not recorded unless CALL_RECORDING_GENERAL=on.
        record = os.getenv("CALL_RECORDING_GENERAL", "off").lower() == "on"
    else:
        record = os.getenv("CALL_RECORDING", "on").lower() != "off"
    # Stored BEFORE dialling: Twilio fetches /twiml as soon as the call is answered.
    store_call_context(sig, ts, ctx)
    client = TwilioClient(account_sid, auth_token)
    try:
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
    except Exception:
        drop_call_context(sig)
        raise

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

    # Only serve TwiML for a call our own /dialout created (see sign_call).
    qp = request.query_params
    row_id, ts, sig = qp.get("row_id", ""), qp.get("ts", ""), qp.get("sig", "")
    if not verify_call(to_number or "", row_id, ts, sig):
        logger.warning("Rejected /twiml request with a missing, bad or expired signature")
        raise HTTPException(status_code=403, detail="Forbidden")

    context = get_call_context(sig)
    if context is None:
        logger.error(
            f"CALL CONTEXT MISSING at /twiml for row {row_id!r}: the container restarted between "
            "dialling and the answer, or more than one replica is running. Apologising and hanging up."
        )
    return TwimlRequest(
        to_number=to_number or "", from_number=from_number or "", row_id=row_id, ts=ts, sig=sig, context=context
    )


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
    if twiml_request.context is None:
        # No instructions for this call: one polite line and hang up, rather
        # than a bot improvising at a stranger.
        response.say(MISSING_CONTEXT_LINE)
        response.hangup()
        return str(response)
    connect = Connect()
    stream = Stream(url=websocket_url)

    # Only what the bot needs to verify the call and find its context in the
    # store. These arrive in the WebSocket 'start' message.
    stream.parameter(name="to_number", value=twiml_request.to_number)
    stream.parameter(name="from_number", value=twiml_request.from_number)
    stream.parameter(name="row_id", value=twiml_request.row_id)
    stream.parameter(name="ts", value=twiml_request.ts)
    stream.parameter(name="sig", value=twiml_request.sig)

    connect.append(stream)
    response.append(connect)
    response.pause(length=20)

    return str(response)
