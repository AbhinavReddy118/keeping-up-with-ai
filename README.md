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

## Quick Start

Run the complete end-to-end pipeline (ingestion → deduplication → analysis → ranking → delivery):

```bash
# Single run (uses configured LLM provider from .env)
python3 main.py

# Run continuously every 60 seconds
python3 main.py --interval 60

# Skip specific stages
python3 main.py --no-dedup --no-delivery
python3 main.py --batch-size 20 --rank-limit 5

# JSON output for scripting
python3 main.py --output json
```

### Telegram Configuration

Set these environment variables in `.env` to enable Telegram delivery:

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_ENABLED` | Enable Telegram delivery (`true`/`false`) | `false` |
| `TELEGRAM_BOT_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) | `""` |
| `TELEGRAM_CHAT_ID` | Target chat or channel ID | `""` |
| `TELEGRAM_MIN_SCORE` | Minimum final score to deliver (0.0–1.0) | `0.0` |
| `TELEGRAM_LIMIT` | Maximum articles per delivery | `5` |

---



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

---

## Deterministic Article Ranking (Task 3.0)

The system includes a deterministic article ranking engine in `ranking/` that dynamically evaluates articles joined with completed `article_analysis` records without persisting dynamic scores to the database.

### Ranking Architecture

```text
    articles JOIN article_analysis (WHERE status = 'completed')
                              │
                              ▼
                   [Component Scoring]
         ┌─────────────┬─────────────┬─────────────┐
         ▼             ▼             ▼             ▼
     Relevance      Novelty       Recency     Confidence
       (40%)         (25%)         (20%)         (15%)
         │             │             │             │
         └─────────────┼─────────────┴─────────────┘
                       ▼
                 [Final Score]  (Bounded between 0.0 and 1.0)
                       │
                       ▼
             [Deterministic Sorting]
          1. final_score DESC
          2. published_at DESC (tie-breaker)
          3. article_id ASC (tie-breaker)
```

### Scoring Formula & Weights

The final score is computed as a weighted linear combination:

$$\text{Final Score} = (0.40 \times \text{Relevance}) + (0.25 \times \text{Novelty}) + (0.20 \times \text{Recency}) + (0.15 \times \text{Confidence})$$

* **Novelty Mapping**:
  * `breakthrough`: `1.0`
  * `significant`: `0.7`
  * `incremental`: `0.4`
  * `unknown`: `0.1`
* **Recency Half-Life Decay**:
  * Exponential decay formula: $\text{Recency} = 2^{-\Delta t / t_{\text{half}}}$ where $t_{\text{half}} = 7.0\text{ days}$.
  * Handles missing/unparseable dates with a safe default (`0.20`).
  * Clamps future timestamps to $\Delta t \ge 0$ (score $\le 1.0$).
* **Event Type**: Handled conservatively with default weight 0.0%, preserving strict normalization to 1.0.

### Running the Ranking Demonstration

```bash
python3 demo_ranking.py --limit 10
```

* Dynamically retrieves completed records via SQL JOIN.
* Computes component breakdown and final scores.
* Strictly read-only: verifies the SQLite `articles` table is never modified.

### Known Limitations

* **Linear Static Weighting**: Weights are fixed across sources and domains rather than learned.
* **Cold Starts**: Articles that have not yet undergone LLM analysis (`status != 'completed'`) are excluded from ranking until batch analysis runs.

---

## Web Interface (Task 3.1)

An async, server-rendered web dashboard in `web/` built with **FastAPI + Jinja2 + Vanilla CSS** that provides an interactive, read-only presentation of ingested articles and intelligence rankings.

### Architecture

```text
       SQLite Database (`articles` & `article_analysis`)
                              │
                              ▼
        Read-Only Query Layer (`web/queries.py`)
       (Joins, aggregations, pagination, ranking)
                              │
                              ▼
            FastAPI Application (`web/app.py`)
           (HTTP routes, error handlers, filters)
                              │
                              ▼
            Jinja Templates + Vanilla CSS
      (Dashboard, Article Detail, About, Responsive)
```

### Key Features & Design Reproduction
* **Visual Fidelity**: Faithful reproduction of the reference intelligence dashboard with deep navy typography (`#0f172a`), soft slate background (`#f8fafc`), and solid blue accents (`#2563eb`).
* **Summary Metrics**: 4 headline cards showing Total Articles, Analyzed & Ranked, Not Yet Analyzed, and Active Sources.
* **Interactive Filtering**: Filter by Source and Event Type with URL query parameter persistence (`/?source=OpenAI&event_type=model_release`).
* **Top Ranked AI News**: Displays top scored articles with amber score pills (`0.673`), styled thumbnails, metadata, and summaries.
* **Latest AI News Feed**: Displays newly ingested developments sorted chronologically with distinct `Analyzed` vs. `Not analyzed` badges.
* **Article Deep Dive**: Full article view (`/article/<id>`) displaying component scores breakdown, executive summary, key takeaways, and extracted entity tags.
* **100% Read-Only Safety**: Browsing, filtering, and detail navigation never insert, mutate, or delete SQLite records.

### Starting the Web Dashboard Locally

```bash
# Run using the Python module runner (uvicorn binds to 127.0.0.1:5000):
python3 -m web.app

# Or run directly with uvicorn:
uvicorn web.app:app --host 127.0.0.1 --port 5000
```
Open **`http://127.0.0.1:5000`** in your browser.

### Available Routes
* `GET /`: Main dashboard with headline metrics, sidebar filters, top ranked news, and paginated latest news feed.
* `GET /article/<article_id>`: Individual article detail page with full scoring breakdown and extracted intelligence.
* `GET /about`: Architectural overview and pipeline status.

### Known Limitations
* **Server-Rendered Only**: Interactivity relies on native HTML forms and query strings rather than dynamic client-side SPA state.
* **Synchronous Requests**: Page rendering fetches from local SQLite synchronously.

---

## End-to-End Pipeline Orchestration

The `orchestration.py` module chains all stages into a single command:

```
RSS Sources
    ↓
Ingestion (fetch, normalize, deduplicate, persist)
    ↓
LLM Batch Analysis (unanalyzed or failed articles)
    ↓
Deterministic Ranking (relevance, novelty, recency, confidence)
    ↓
Telegram Delivery (top-ranked stories above min-score threshold)
```

### Programmatic Usage

```python
from orchestration import run_full_pipeline
from llm import MockLLMProvider

result = run_full_pipeline(
    deduplicate=True,
    analyze=True,
    analysis_limit=10,
    provider=MockLLMProvider(),
    rank=True,
    rank_limit=10,
    deliver=True,
)

print(result["ingestion"]["total_inserted"])
print(result["analysis"]["completed"])
print(result["ranking"]["count"])
print(result["delivery"]["sent"])
```

### Deduplication (Task 2.10)

The `deduplicate_articles()` function groups articles by:

1. Exact URL match
2. Normalized URL (strips tracking params, fragments, case)
3. Content fingerprint (SHA-256 of normalized title + summary)
4. Normalized title match
5. Title similarity (SequenceMatcher ratio + token Jaccard with thresholds)

One representative per group is persisted; duplicates are skipped from storage and downstream analysis.

---

## Rolling Retention Window

The `retention.py` module archives and deletes articles older than `RETENTION_DAYS` (default 30). Run automatically at the start of every pipeline pass, or invoke directly:

```python
from retention import prune_articles, archive_stats

result = prune_articles(days=30)
# {"enabled": True, "archived": 12, "deleted": 12, "cutoff": "2025-08-21T..."}

stats = archive_stats()
# {"exists": True, "total": 847, "oldest": "...", "newest": "..."}
```

Set `RETENTION_DAYS=0` in `.env` to disable archival.

Archive lives at `data/news_digest_archive.db` next to the main database.

---

## Personalization (Topic-Weight Re-ranking)

Optional nudge after deterministic ranking. Set `TOPIC_WEIGHTS` in `.env`:

```
TOPIC_WEIGHTS=agent:1.5,reasoning:1.3,tool:1.2,safety:1.4
```

Articles whose title+summary contains a keyword get a bonus up to `+0.20` on the final score. Deterministic ranking itself is unchanged; only the delivery score is shifted. Original ranking in the web dashboard is unaffected.

---

## Scheduled Runs (macOS launchd)

Install a daily 09:00 IST cron that runs the full pipeline and writes logs:

```bash
./scripts/install_cron.sh
```

This copies `scripts/com.ai-news-digest.daily.plist` to `~/Library/LaunchAgents/`, loads it via `launchctl`, and points `run_daily.sh` at the project root.

Manual controls:

```bash
# Trigger a run now
launchctl start com.ai-news-digest.daily

# View recent runs
ls -lt data/daily_*.log | head

# Uninstall
launchctl unload ~/Library/LaunchAgents/com.ai-news-digest.daily.plist
rm ~/Library/LaunchAgents/com.ai-news-digest.daily.plist
```

The plist runs the pipeline with `TZ=Asia/Kolkata` so 09:00 local time fires correctly across DST changes.

---

## Running Tests

```bash
# Run all tests using virtual environment Python
.venv/bin/python -m pytest

# Or activate venv first:
# source .venv/bin/activate && pytest

# Run specific test module
.venv/bin/python -m pytest tests/test_pipeline.py -v
.venv/bin/python -m pytest tests/test_delivery.py -v
.venv/bin/python -m pytest tests/test_orchestration.py -v
.venv/bin/python -m pytest tests/test_retention.py -v
```

All tests run offline by default using `MockLLMProvider`. No network or API keys required for testing.

