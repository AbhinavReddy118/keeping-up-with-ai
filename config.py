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
LLM_MODEL = os.getenv("LLM_MODEL", "mock-model")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
