# voice-caller — your own outbound dietary-inquiry voice agent

Replaces Retell. Owns the whole media path (no Retell SBC), so the one-way-audio
bug goes away. Pipeline: **Twilio ⇄ Deepgram (STT) → OpenAI (our tuned prompt) → ElevenLabs (Monika) ⇄ Twilio**, running on **Railway**.

Based on Pipecat's official `twilio-chatbot/outbound` example, customized for our use case.

## Files
- `bot.py` — the Pipecat pipeline; loads `prompt.txt`, injects per-call context (restaurant, notes).
- `prompt.txt` — our fully-tuned dietary-inquiry prompt (ported 1:1 from Retell).
- `server.py` — FastAPI: `POST /dialout` (start a call), `POST /twiml` (Twilio callback), `WS /ws` (media).
- `server_utils.py` — Twilio call + TwiML helpers; passes restaurant/call_notes through to the bot.
- `pyproject.toml`, `Dockerfile`, `env.example`.

## Keys you need (all go into Railway → Variables; you never paste them into chat)
See `env.example`. In short: **Deepgram** key, **OpenAI** key, **ElevenLabs** key + the **Monika voice id**, and **Twilio** SID + auth token + a purchased number.

## Deploy to Railway
1. Push this folder to a GitHub repo (or use `railway up` from the Railway CLI).
2. In Railway: **New Project → Deploy from repo** (or `railway init` here).
3. Add all the variables from `env.example` in Railway's **Variables** panel.
4. Railway builds via the `Dockerfile` and gives you a public URL like
   `https://voice-caller-production.up.railway.app` (HTTPS/`wss` included — no cert setup!).
5. Set `LOCAL_SERVER_URL` to that URL and redeploy.

## Place a call
```bash
curl -X POST "$LOCAL_SERVER_URL/dialout" \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+13153730034",
    "from_number": "'"$TWILIO_FROM_NUMBER"'",
    "restaurant_name": "King David'\''s",
    "cuisine": "Middle Eastern build-your-own, Syracuse NY",
    "call_notes": "He is vegan and cannot have any onion or garlic. Ask about falafel and hummus (made fresh without garlic?), fried eggplant, and a build-your-own allium-free plate."
  }'
```

## Note on Pipecat versions
Pipecat's import paths move between releases. On the first `railway up`, if the build or a
run errors on an import (e.g. `PipelineWorker`, `LLMContext`, an `ElevenLabsTTSService` param),
we pin the exact installed version and adjust those few lines — that's normal for the first run.
