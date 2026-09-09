# Keeping Up With AI

> A personal AI intelligence pipeline that continuously tracks new models, research, tools, companies, and major developments across the AI ecosystem.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

## The Problem

AI changes faster than it is practical to follow manually.

New model releases, research papers, developer tools, open-source projects, company announcements, and industry developments appear across hundreds of sources.

I built this project to answer one question:

### **"What changed in AI today, and what actually matters?"**

---

## What It Does

The system continuously:

**Collects** → **Cleans** → **Deduplicates** → **Analyzes** → **Ranks** → **Delivers**

| Stage | What happens |
|---|---|
| 📡 Ingestion | Collects updates from configured AI sources |
| 🧹 Normalization | Converts different source formats into a common structure |
| 🔎 Deduplication | Detects the same development appearing across multiple sources |
| 🤖 AI Analysis | Classifies and summarizes important developments |
| 📊 Ranking | Determines which developments deserve attention |
| 🌐 Web | Provides a searchable view of collected intelligence |
| 📱 Telegram | Sends important updates directly to Telegram |

---

## Architecture

```text
                  AI ECOSYSTEM
                       │
          ┌────────────┴────────────┐
          │                         │
       RSS / APIs              Other Sources
          │                         │
          └────────────┬────────────┘
                       ↓
                 DATA INGESTION
                       ↓
                 NORMALIZATION
                       ↓
                 DEDUPLICATION
                       ↓
                SQLite Database
                       ↓
                 AI ANALYSIS
                       ↓
              IMPORTANCE RANKING
                       │
              ┌────────┴────────┐
              ↓                 ↓
        WEB INTERFACE        TELEGRAM
```

---

## LLM Analysis Foundation & Live Integration (Tasks 2.9.1 & 2.9.2)

The system includes a provider-independent structured extraction engine in `llm/`:

* **Structured Schema (`AnalysisResult`)**: Validates extracted attributes including `relevance_score`, `summary`, `key_points`, `topics`, `technologies`, `companies`, `models`, `people`, `event_type`, `novelty`, and `confidence`.
* **Provider Abstraction (`LLMProvider`)**:
  - `MockLLMProvider`: Deterministic offline mock provider for automated tests and local development without network access.
  - `OpenAICompatibleProvider`: HTTP-based provider using standard library `urllib.request` (zero vendor SDK bloat) supporting OpenAI (`gpt-4o-mini`, etc.) and OpenRouter endpoints with JSON mode enforcement (`response_format={"type": "json_object"}`).
* **Prompt Versioning & Schema Validation**: Uses versioned prompt templates (`ANALYSIS_PROMPT_VERSION = "1.0"`) and typed exception mapping (`LLMConfigurationError`, `LLMRateLimitError`, `LLMProviderUnavailableError`, `LLMTimeoutError`, `LLMMalformedResponseError`, `LLMSchemaValidationError`).
* **Non-Destructive**: Existing database articles are **not bulk-analyzed automatically**; database contents remain completely read-only.

---

## LLM Configuration

Configuration is managed strictly via environment variables (loaded automatically from `.env` via `python-dotenv`):

| Variable | Description | Default | Example |
|---|---|---|---|
| `LLM_PROVIDER` | Provider identifier (`mock`, `openai`, `openrouter`) | `mock` | `openai` |
| `LLM_MODEL` | Target model identifier | Provider default | `gpt-4o-mini` |
| `LLM_API_KEY` | Secret API key for real provider | `""` | `sk-...` |
| `LLM_BASE_URL` | Base REST URL for endpoint | Provider default | `https://api.openai.com/v1` |

> [!WARNING]
> **SECURITY WARNING: NEVER COMMIT SECRETS**
> Never commit your `.env` file or hardcode API keys into source code or test fixtures. `.env` and `.env.*` are explicitly ignored in `.gitignore`. The system masks all credentials and never logs API keys in console output or exception messages.

### Mode Comparison: Offline/Mock vs. Live

* **Offline/Mock Mode (`LLM_PROVIDER=mock`)**:
  - The default configuration across all environments.
  - Generates deterministic, structured results locally without internet access or API credentials.
  - Used by all automated unit tests (`pytest`).
* **Live Mode (`LLM_PROVIDER=openai` or `openrouter`)**:
  - Sends requests over HTTP to the configured API with JSON structured response enforcement.
  - Requires `LLM_API_KEY` to be set in `.env` or the environment.

---

## Running LLM Demonstrations

### 1. Offline Demonstration (Deterministic Mock)
Analyzes one article from SQLite using the offline mock provider (no network access):
```bash
python3 demo_llm_analysis.py
```

### 2. Live LLM Demonstration (Single Article)
Analyzes exactly **ONE** article from SQLite using the configured real provider (or displays setup instructions if credentials are not configured):
```bash
# To test with live OpenAI credentials:
# 1. Add LLM_PROVIDER=openai and LLM_API_KEY=your_key to .env
# 2. Run the demo:
python3 demo_live_llm.py
```

Both demonstration scripts verify that the SQLite database article count before and after execution is identical (100% read-only).

