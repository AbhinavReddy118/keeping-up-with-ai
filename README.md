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

## LLM Analysis Foundation, Live Integration & Batch Persistence (Tasks 2.9.1, 2.9.2, 2.9.3)

The system includes a provider-independent structured extraction engine in `llm/`:

* **Structured Schema (`AnalysisResult`)**: Validates extracted attributes including `relevance_score`, `summary`, `key_points`, `topics`, `technologies`, `companies`, `models`, `people`, `event_type`, `novelty`, and `confidence`.
* **Provider Abstraction (`LLMProvider`)**:
  - `MockLLMProvider`: Deterministic offline mock provider for automated tests and local development without network access.
  - `OpenAICompatibleProvider`: HTTP-based provider using standard library `urllib.request` (zero vendor SDK bloat) supporting OpenAI (`gpt-4o-mini`, etc.) and OpenRouter endpoints with JSON mode enforcement (`response_format={"type": "json_object"}`).
* **Prompt Versioning & Schema Validation**: Uses versioned prompt templates (`ANALYSIS_PROMPT_VERSION = "1.0"`) and typed exception mapping (`LLMConfigurationError`, `LLMRateLimitError`, `LLMProviderUnavailableError`, `LLMTimeoutError`, `LLMMalformedResponseError`, `LLMSchemaValidationError`).
* **Batch Analysis & Persistence (`article_analysis`)**:
  - **Database Persistence**: Stores analysis results in a dedicated `article_analysis` table with a foreign key to `articles(id)` and a `UNIQUE(article_id)` constraint.
  - **JSON Serialization**: Multi-value attributes (`key_points`, `topics`, `technologies`, `companies`, `people`, `models`, `entities`) are cleanly serialized as JSON strings.
  - **Deterministic Selection**: Selects unanalyzed or previously failed articles (`an.id IS NULL OR an.status != 'completed'`) ordered by `articles.id ASC`.
  - **Bounded Retries**: Differentiates transient errors (`LLMTimeoutError`, `LLMProviderUnavailableError`, `LLMRateLimitError`) which retry up to `LLM_MAX_RETRIES`, from non-transient errors (`LLMMalformedResponseError`, `LLMSchemaValidationError`, `LLMConfigurationError`, `ValueError`) which fail immediately without wasting retries.
  - **Resumability & Idempotency**: Atomic upsert (`ON CONFLICT(article_id) DO UPDATE`) allows interrupted batches to resume without duplicate rows or repeated work.
  - **Non-Destructive**: The `articles` table remains 100% read-only; article records are never altered or deleted.

---

## LLM Configuration

Configuration is managed strictly via environment variables (loaded automatically from `.env` via `python-dotenv`):

| Variable | Description | Default | Example |
|---|---|---|---|
| `LLM_PROVIDER` | Provider identifier (`mock`, `openai`, `openrouter`) | `mock` | `openai` |
| `LLM_MODEL` | Target model identifier | Provider default | `gpt-4o-mini` |
| `LLM_API_KEY` | Secret API key for real provider | `""` | `sk-...` |
| `LLM_BASE_URL` | Base REST URL for endpoint | Provider default | `https://api.openai.com/v1` |
| `LLM_BATCH_SIZE` | Default number of articles per batch pass | `10` | `20` |
| `LLM_MAX_RETRIES` | Max retries for transient failures | `2` | `3` |

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

### 1. Offline Single-Article Demonstration (Deterministic Mock)
Analyzes one article from SQLite using the offline mock provider (no network access):
```bash
python3 demo_llm_analysis.py
```

### 2. Live LLM Single-Article Demonstration
Analyzes exactly **ONE** article from SQLite using the configured real provider (or displays setup instructions if credentials are not configured):
```bash
# To test with live OpenAI credentials:
# 1. Add LLM_PROVIDER=openai and LLM_API_KEY=your_key to .env
# 2. Run the demo:
python3 demo_live_llm.py
```

### 3. Batch Analysis & Persistence Demonstration
Processes a batch of unanalyzed articles, persists them to `article_analysis`, and tests resumability:
```bash
# Process 5 articles using Mock provider (offline safe):
python3 demo_batch_analysis.py --mock --limit 5

# Or run using configured provider in .env:
python3 demo_batch_analysis.py --limit 10
```

All demonstration scripts verify that the SQLite database `articles` table count before and after execution is identical (100% read-only).


