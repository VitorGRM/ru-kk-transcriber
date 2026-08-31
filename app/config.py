"""Central configuration. Values come from the environment (.env is loaded if present)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_kw):
        return False

FROZEN = getattr(sys, "frozen", False)

# When packaged as a Windows .exe, read-only resources live in the temporary
# folder PyInstaller unpacks to, while anything written — uploads, downloaded
# models, .env — has to sit next to the executable so it survives a restart.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", "")) if FROZEN else Path(__file__).resolve().parent.parent
BASE_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("TA_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = Path(os.getenv("TA_MODEL_DIR", DATA_DIR / "models"))
LOG_DIR = DATA_DIR / "logs"

# Where the Save dialog opens by default. The user's own Downloads folder if it
# exists, so exports land somewhere they will think to look for them.
_downloads = Path.home() / "Downloads"
DOWNLOAD_DIR = Path(os.getenv("TA_DOWNLOAD_DIR", _downloads if _downloads.is_dir()
                              else DATA_DIR / "exports"))

for _d in (DATA_DIR, UPLOAD_DIR, MODEL_DIR, LOG_DIR, DOWNLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Audio ---------------------------------------------------------------
SAMPLE_RATE = 16_000
MAX_UPLOAD_MB = int(os.getenv("TA_MAX_UPLOAD_MB", "512"))

# --- The two languages this app is built around --------------------------
# Every chunk is routed to one of these and decoded under its language token.
LANGUAGE_NAMES = {"ru": "Russian", "kk": "Kazakh"}

# --- Local Whisper defaults ----------------------------------------------
DEFAULT_LOCAL_MODEL = os.getenv("TA_LOCAL_MODEL", "large-v3")
DEFAULT_COMPUTE_TYPE = os.getenv("TA_COMPUTE_TYPE", "auto")
DEFAULT_DEVICE = os.getenv("TA_DEVICE", "auto")

# This app is fully offline: no accounts, no API keys, no per-minute billing.
# Everything below runs on the local machine.

HOST = os.getenv("TA_HOST", "127.0.0.1")
PORT = int(os.getenv("TA_PORT", "8000"))

# --- How the interface is shown ------------------------------------------
#   auto     native desktop window, falling back to the browser if it fails
#   window   native window only; fail loudly rather than opening a browser
#   browser  hand the URL to the default browser (the pre-1.1 behaviour)
#   none     serve only, open nothing — used by the CI smoke test
UI_MODE = os.getenv("TA_UI", "auto").strip().lower()
