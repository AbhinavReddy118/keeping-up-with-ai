# Keeping Up With AI

A personal AI news digest that collects important AI updates, analyzes them locally, and ranks them by importance.

## What it does

The pipeline works like this:

AI Sources → Ingestion → Deduplication → Web Enrichment → AI Analysis → Ranking → Dashboard / Telegram

It currently collects from:

- OpenAI News
- Google DeepMind
- Microsoft Research
- Hugging Face
- MIT Technology Review AI
- Hacker News AI
- arXiv CS.AI

New articles are checked automatically at different intervals.

## Tech Stack

- Python
- SQLite
- FastAPI
- Jinja2
- Ollama
- Qwen 3 4B
- Telegram Bot API
- macOS launchd

The AI model runs locally using Ollama.

## AI Analysis

Each article is analyzed for:

- Relevance
- Summary
- Key points
- Topics
- Technologies
- Companies
- Models
- Event type
- Novelty
- Confidence

Articles are then given an importance score based on relevance, novelty, recency, and confidence.

## Setup

Clone the repository:

```bash
git clone https://github.com/AbhinavReddy118/Ai_news_digest.git
cd Ai_news_digest
```

Create the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama

Install Ollama and download the model:

```bash
ollama pull qwen3:4b
```

Check that the model is available:

```bash
ollama list
```

The model is not included in the repository and must be downloaded separately.

## Configuration

Create a `.env` file and configure the local model:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
LLM_BASE_URL=http://localhost:11434

TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_MIN_SCORE=0.6
TELEGRAM_LIMIT=5
```

Do not commit `.env` or API keys to GitHub.

## Run the Monitor

Start the continuous monitor:

```bash
.venv/bin/python monitor.py
```

The project can also run as a background macOS service using launchd.

## Dashboard

Start the dashboard:

```bash
.venv/bin/python -m uvicorn web.app:app --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## Telegram

Telegram notifications are optional.

Set:

```env
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

The bot sends articles that pass the configured importance threshold.

## Database

The project uses SQLite.

Database:

```text
data/news_digest.db
```

No separate database server is required.

## Tests

Run the test suite with:

```bash
.venv/bin/python -m pytest
```

## Current Limitations

- Designed to run locally on macOS.
- Ollama must be installed separately.
- Qwen 3 4B must be downloaded separately.
- The Mac needs to be running for continuous monitoring.
- News coverage depends on the configured sources.
- Some sources provide limited article content, so enrichment may not always be possible.
- Local AI analysis can sometimes be incomplete or incorrect.
- Telegram requires a configured bot and chat.
- There is currently no cloud LLM fallback.

## Project Status

The core pipeline is working end-to-end:

- Automatic source polling
- Article ingestion
- Deduplication
- Web enrichment
- Local Qwen analysis
- Importance scoring
- SQLite pipeline state
- Continuous background monitoring
- Web dashboard
- Telegram notification support
