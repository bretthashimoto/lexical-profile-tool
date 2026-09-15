"""App configuration, read from environment variables with local-dev defaults."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Languages this deployment offers in the build form -- see
# webapp/app.py's WEBAPP_LANGUAGES for the original rationale (these are
# the languages webapp/requirements.txt pre-installs a trained spaCy
# pipeline for, so lemmatization works out of the box).
WEBAPP_LANGUAGES = ["en", "es", "fr", "de"]


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SESSION_DIR = Path(os.environ.get("SESSION_DIR", BASE_DIR / "instance" / "sessions"))
    SESSION_TTL = int(os.environ.get("SESSION_TTL", 6 * 60 * 60))  # 6 hours
    JOB_TTL = int(os.environ.get("JOB_TTL", 30 * 60))  # 30 minutes
    PERMANENT_SESSION_LIFETIME = SESSION_TTL
