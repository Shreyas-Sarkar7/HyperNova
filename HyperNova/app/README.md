# Trackside HQ

Flask backend for worker login + the HQ fault dashboard, with a real AI
severity classifier wired in.

## Setup

1. `pip install -r requirements.txt`
2. `cp .env.example .env` and paste your real NVIDIA API key into `.env`
3. `python app.py`
4. Open http://localhost:5000

## Demo login

- Username: `worker1` (or `admin`)
- Password: `hackathon2026`

Both accounts share this demo password for convenience. The password is
stored as a hash in `data/users.json`, never in plain text -- see
"Adding real workers" below if you want to add more accounts.

## Pages

- `/` -- landing page, links to login
- `/login` -- worker login
- `/dashboard` -- HQ dashboard: lines list + announcements
- `/line/<line_id>` -- stations and sub-lines for one line
- `/line/<line_id>/segment/<subline_id>` -- 10-segment strip + flagged photos
- `/tools/classify` -- paste an image URL, see the AI classify it live
- `/api/classify` -- POST { "image_url": "..." } -> JSON severity result

## Data

`data/lines.json` is mock data -- three lines, each with stations and the
tracked stretches between them, each stretch broken into 10 segments with
a status (`green` / `yellow` / `red`) and an optional `photo_url`.

This is meant to be replaced by whatever the real segment-processing and
camera pipeline produces. The website only depends on that shape, so
swapping in real data doesn't require touching the templates.

Photos are only shown for `yellow` / `red` segments, and any segment
without a `photo_url` shows an honest "Awaiting camera capture"
placeholder rather than a fake image.

## Adding real workers

```python
from werkzeug.security import generate_password_hash
print(generate_password_hash("their-password"))
```

Add `{"username": "...", "password_hash": "<output above>"}` to
`data/users.json`.

## AI classifier

`ai_classifier.py` wraps the NVIDIA-hosted vision model from your
original script. The one change: it uses `stream=False` instead of
`stream=True`, since the backend needs the finished text to parse a
severity label rather than printing tokens as they arrive. Everything
else (endpoint, model, seed, reasoning_effort) matches your script.

Requires `NVIDIA_API_KEY` to be set in `.env`.
