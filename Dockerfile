# ============================================================
#  AI-DRCS  —  Rasa Pro Server (Render.com / Docker)
#  Single container runs both the Rasa server AND action server.
# ============================================================

FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Rasa Pro ──────────────────────────────────────────────────────────
# Uses the official Rasa Pro PyPI index (requires RASA_PRO_LICENSE env var at runtime)
RUN pip install --upgrade pip setuptools wheel

# Install Rasa Pro from the official private index
RUN pip install rasa-pro \
    --extra-index-url https://europe-west3-python.pkg.dev/rasa-releases/rasa-pro-python/simple/ \
    --no-cache-dir

# Install action server dependencies
RUN pip install requests rasa-sdk --no-cache-dir

# ── Copy project files ────────────────────────────────────────────────────────
COPY . /app

# ── Train the model at build time ─────────────────────────────────────────────
# The Rasa Pro licence is needed at train time too.
# Pass it as a Docker build arg: docker build --build-arg RASA_PRO_LICENSE=... .
ARG RASA_PRO_LICENSE=""
ENV RASA_PRO_LICENSE=${RASA_PRO_LICENSE}

# Train with a fixed model name so the run command always finds it
RUN rasa train --fixed-model-name drcs_model --quiet || true

# ── Startup script ────────────────────────────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5005

CMD ["/app/start.sh"]
