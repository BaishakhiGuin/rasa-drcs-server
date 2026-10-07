# AI-DRCS Rasa Pro Server

Conversational AI backend for the AI-DRCS Merchant Support Platform.

## What's in this repo

```
rasa-drcs/
├── domain.yml                  # Intents, entities, slots, forms, responses, actions
├── config.yml                  # NLU pipeline + dialogue policies
├── endpoints.yml               # Action server + tracker store config
├── credentials.yml             # REST channel config
├── Dockerfile                  # Single-container: Rasa + action server
├── start.sh                    # Starts both servers in one container
├── data/
│   ├── nlu.yml                 # ~800 training examples across 50+ intents
│   ├── stories.yml             # Full conversation arc flows
│   ├── rules.yml               # Deterministic rules (greet, escalate, forms)
│   └── lookup_tables.yml       # Regex patterns + lookup tables for entities
├── actions/
│   ├── actions.py              # Custom Python actions (auth, RAG, refund, etc.)
│   └── requirements-actions.txt
└── tests/
    └── test_stories.yml        # End-to-end conversation tests
```

## Coverage

| Category | Intents | Slot Forms |
|---|---|---|
| Greetings & Meta | greet, goodbye, affirm, deny, thanks, help, restart | — |
| Device Issues | 14 intents (blinking, no sound, charging, GSM, replace, reset…) | device_issue_form |
| Transaction Issues | 8 intents (failed, pending, double debit, refund, status…) | transaction_form, refund_form |
| UPI / Payments | 6 intents (UPI down, QR, no beep, bank issue, limit…) | upi_form |
| Settlement | 4 intents (delay, wrong amount, not received, cycle) | — |
| Account / KYC | 5 intents (suspended, KYC, bank change, details update) | — |
| Onboarding | 4 intents (device setup, SIM, app, general) | — |
| Escalation | 3 intents (escalate, register complaint, status) | complaint_form |
| Slot Providers | 7 intents (provide MSISDN, TXN ID, serial, amount, date…) | merchant_auth_form |

## Deploy to Render.com (free, permanent URL)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial AI-DRCS Rasa backend"
git remote add origin https://github.com/YOUR_USERNAME/rasa-drcs-server.git
git push -u origin main
```

### Step 2 — Create Render Web Service

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Environment:** Docker
   - **Plan:** Free
   - **Port:** 5005
4. Add **Environment Variables**:

| Key | Value |
|---|---|
| `RASA_PRO_LICENSE` | Your full RASA_KEY JWT |
| `QDRANT_URL` | Your Qdrant cloud URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `COLLECTION_NAME` | `research_notebook` |
| `GROQ_API_KEY` | Your Groq API key |
| `HF_TOKEN` | Your HuggingFace token |
| `APPS_SCRIPT_EXEC_URL` | Your Apps Script exec URL |

5. Click **Deploy**

Your permanent URL: `https://rasa-drcs-server.onrender.com`

### Step 3 — Add to Apps Script Properties

In your Google Apps Script → Project Settings → Script Properties:
```
RASA_BASE_URL = https://rasa-drcs-server.onrender.com
```

### Step 4 — Keep Render warm (optional but recommended)

Add a free [UptimeRobot](https://uptimerobot.com) monitor:
- URL: `https://rasa-drcs-server.onrender.com`
- Interval: 14 minutes
- This prevents the 30-second cold start on free tier.

## Local development

```bash
# Install Rasa Pro
pip install rasa-pro \
  --extra-index-url https://europe-west3-python.pkg.dev/rasa-releases/rasa-pro-python/simple/

export RASA_PRO_LICENSE="your_rasa_key_here"

# Train
rasa train

# Start action server (terminal 1)
rasa run actions --port 5055

# Start Rasa server (terminal 2)
rasa run --enable-api --cors "*" --port 5005 --endpoints endpoints.yml

# Test NLU
rasa shell nlu

# Run conversation tests
rasa test

# Interactive learning (add new stories)
rasa interactive
```

## Adding your own documents

Upload PDFs/DOCx/TXT via the AI-DRCS Knowledge portal. The Apps Script backend:
1. Extracts text
2. Chunks into parent-child pairs
3. Embeds with HuggingFace all-MiniLM-L6-v2 (free, 384-dim)
4. Upserts to Qdrant

Rasa's `action_rag_search` then queries this collection for any question not covered by the trained intents.

## Retraining after adding new intents

1. Add examples to `data/nlu.yml`
2. Add stories/rules for the new flow
3. Push to GitHub → Render auto-deploys and retrains via Docker build
