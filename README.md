# voice-caller

An outbound voice agent that phones restaurants and works out what the kitchen can actually make
against a strict diet. It dials during local business hours, holds a real conversation with whoever
picks up, and returns what is safe to order and why.

Pipeline: **Twilio ⇄ Deepgram (STT) → Claude → ElevenLabs (TTS) ⇄ Twilio**, built on
[Pipecat](https://github.com/pipecat-ai/pipecat) and running on Railway.

It started on Retell and was rebuilt self-hosted to own the whole media path, which removed a
one-way-audio problem introduced by the hosted SBC.

## What was actually hard

Not the integration. Turn-taking.

- Pipecat's default lets any 200ms of speech interrupt the bot, so a breath or a bit of kitchen
  noise would cut the agent off mid-greeting and end the call. The interrupt bar is now three words
  on interim transcripts, which still reacts in ~300-600ms but filters backchannels and echo.
- A first-speech mute guarantees the opening greeting plays to completion.
- The system prompt was trimmed from ~4,200 tokens to ~950 after reading real call transcripts:
  most of the instructions were solving for failure modes that never occurred. It has since grown
  back as genuine ones appeared.
- TTS is swappable from an env var (`TTS_PROVIDER`), so ElevenLabs, Deepgram Aura-2, Rime, and
  Cartesia can be A/B tested on live calls without a code change. Voice is mapped per cuisine, since
  the default voice anglicised Indian dish names that `eleven_multilingual_v2` renders natively.

## Files

- `bot.py` - the Pipecat pipeline; loads `prompt.txt` and injects per-call context
- `prompt.txt` - the dietary-inquiry prompt
- `server.py` - FastAPI: `POST /dialout` starts a call, `POST /twiml` is the Twilio callback, `WS /ws` carries media
- `server_utils.py` - Twilio call and TwiML helpers
- `pyproject.toml`, `Dockerfile`, `env.example`

## Configuration

See `env.example`. You need a Deepgram key, an Anthropic key, an ElevenLabs key and voice id, and
Twilio credentials plus a purchased number. Secrets go in the host's environment panel, never in the
repo.

## Deploy

1. Push to a GitHub repo, or run `railway up` from the Railway CLI
2. In Railway, create a project from the repo and add every variable from `env.example`
3. Railway builds from the `Dockerfile` and returns a public HTTPS/`wss` URL
4. Set `LOCAL_SERVER_URL` to that URL and redeploy

## Place a call

```bash
curl -X POST "$LOCAL_SERVER_URL/dialout" \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550000000",
    "from_number": "'"$TWILIO_FROM_NUMBER"'",
    "restaurant_name": "Example Kitchen",
    "cuisine": "Middle Eastern build-your-own",
    "call_notes": "Caller is vegan and cannot have onion or garlic in any form. Ask about falafel and hummus (made fresh, without garlic?), fried eggplant, and an allium-free build-your-own plate."
  }'
```

## Note on Pipecat versions

Pipecat's import paths move between releases. If a build or run fails on an import such as
`PipelineWorker`, `LLMContext`, or an `ElevenLabsTTSService` parameter, pin the installed version
and adjust those few lines.
