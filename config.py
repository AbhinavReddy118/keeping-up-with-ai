import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# Data directory and SQLite database path
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "news_digest.db"

# LLM Configuration (defaults to mock provider for safe offline operation)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "10"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Telegram delivery configuration (disabled unless TELEGRAM_ENABLED is set)
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_MIN_SCORE = float(os.getenv("TELEGRAM_MIN_SCORE", "0.0"))
TELEGRAM_LIMIT = max(1, int(os.getenv("TELEGRAM_LIMIT", "5")))

# Laya Shadow Triage Configuration (defaults to false for MVP)
LAYA_ENABLED = os.getenv("LAYA_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LAYA_MODEL_ID = os.getenv("LAYA_MODEL_ID", "convaiinnovations/laya")

# Continuous Monitor Polling Interval Configuration
MONITOR_POLL_INTERVAL_TIER1_MINUTES = int(os.getenv("MONITOR_POLL_INTERVAL_TIER1_MINUTES", "20"))
MONITOR_POLL_INTERVAL_TIER2_MINUTES = int(os.getenv("MONITOR_POLL_INTERVAL_TIER2_MINUTES", "40"))
MONITOR_POLL_INTERVAL_TIER3_HOURS = int(os.getenv("MONITOR_POLL_INTERVAL_TIER3_HOURS", "24"))
MONITOR_HEARTBEAT_SECONDS = int(os.getenv("MONITOR_HEARTBEAT_SECONDS", "30"))
MONITOR_BATCH_SIZE = int(os.getenv("MONITOR_BATCH_SIZE", "10"))

