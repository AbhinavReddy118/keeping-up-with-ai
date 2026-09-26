# Developer Guide: System Modification Map

**AI News Intelligence Monitor & Digest**  
*Document Version:* 1.0  
*Target Release:* Task 1.4 Baseline Audit  

This guide provides concrete, direct answers to common developer tasks. For each task, it identifies the exact file paths, class/function names, and verification steps.

---

## Quick Navigation: "If I want to change X, where do I go?"

| Goal | Target File | Key Function / Class | Verification Test |
| :--- | :--- | :--- | :--- |
| **Add or disable an RSS feed** | `sources.py` | `DEFAULT_SOURCES`, `validate_source_config()` | `tests/test_sources.py` |
| **Change HTML stripping or date parsing** | `ingestion/normalizer.py` | `clean_html()`, `normalize_datetime()` | `tests/test_normalization.py` |
| **Change web scraping / text extraction** | `ingestion/enricher.py` | `extract_article_text_from_html()`, `is_usable_content()` | `tests/test_enrichment.py` |
| **Change RSS network fetch headers/timeout** | `ingestion/rss_reader.py` | `fetch_rss_feed()`, `USER_AGENT` | `tests/test_rss_reader.py` |
| **Modify deduplication thresholds** | `deduplication.py` | `deduplicate_articles()`, `calculate_title_similarity()` | `tests/test_deduplication.py` |
| **Modify LLM prompt instructions or schema**| `llm/prompts.py` | `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `build_analysis_prompt()` | `tests/test_llm.py` |
| **Modify LLM result validation / data model**| `llm/models.py` | `AnalysisResult`, `AnalysisResult.from_dict()` | `tests/test_llm.py` |
| **Change local Qwen / Ollama configuration** | `llm/provider.py` | `OllamaProvider.__init__()`, `OllamaProvider.analyze()` | `tests/test_llm.py` |
| **Add a new LLM provider driver** | `llm/provider.py` | Subclass `LLMProvider`, update `get_llm_provider()` | `tests/test_llm.py` |
| **Adjust LLM batch size or retry policy** | `llm/batch.py` | `BatchProcessor`, `is_transient_error()` | `tests/test_batch_analysis.py` |
| **Change deterministic ranking weights** | `ranking/models.py` | `RankingWeights(relevance, novelty, recency, confidence)` | `tests/test_ranking.py` |
| **Change recency half-life or event scores** | `ranking/scorer.py` | `DEFAULT_HALF_LIFE_DAYS`, `NOVELTY_SCORES`, `EVENT_TYPE_SCORES` | `tests/test_ranking.py` |
| **Change topic-weight personalization** | `orchestration.py` | `apply_personalization()`, `_load_topic_weights()` | `tests/test_orchestration.py` |
| **Change Telegram message format or filters**| `delivery/telegram.py` | `format_telegram_update()`, `deliver_ranked_articles()` | `tests/test_delivery.py` |
| **Change web dashboard layout / templates** | `web/templates/*.html` | `index.html`, `article.html`, `base.html` | `tests/test_web.py` |
| **Change web styling / theme colors** | `web/static/style.css` | CSS custom properties, layout grid classes | Browser visual check |
| **Change web data queries or star ratings** | `web/queries.py` | `score_to_stars()`, `clean_human_summary()`, `get_ranked_articles_for_dashboard()` | `tests/test_web.py` |
| **Change database schema or add columns** | `database.py` | `init_db()` | `tests/test_database.py` |
| **Modify retention policy or archival window**| `retention.py` | `prune_articles()`, `_retention_days()` | `tests/test_retention.py` |
| **Add or run automated tests** | `tests/test_*.py` | `pytest tests/` | `.venv/bin/python -m pytest` |

---

## Detailed Task Workflows

### 1. Add an RSS Source
- **File**: `sources.py`
- **Location**: `DEFAULT_SOURCES` list.
- **Action**: Add a new entry dictionary:
  ```python
  {
      "name": "MIT CSAIL News",
      "url": "https://www.csail.mit.edu/news/rss.xml",
      "type": "rss",  # or "atom"
      "enabled": True,
  }
  ```
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_sources.py
  ```

### 2. Change Webpage Article Extraction Logic
- **File**: `ingestion/enricher.py`
- **Location**:
  - `extract_article_text_from_html()`: Controls BeautifulSoup container traversal and noise decomposition.
  - `NON_CONTENT_TAGS`: Tags to prune (e.g. `<script>`, `<style>`, `<nav>`).
  - `is_usable_content()`: Length and word threshold rules (`min_chars=100`, `min_words=15`).
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_enrichment.py
  ```

### 3. Change LLM Prompts & Extraction Schema
- **Prompt Text**: `llm/prompts.py` (`SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`).
  - If changing the prompt structure, increment `ANALYSIS_PROMPT_VERSION = "1.1"`.
- **Validation & Model Contract**: `llm/models.py`.
  - Update `AnalysisResult` dataclass and `from_dict()` if new fields are added.
- **Database Schema**: `database.py`.
  - If new fields must be stored, update `init_db()` migration logic and `llm/batch.py::save_analysis_success()`.
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_llm.py tests/test_batch_analysis.py
  ```

### 4. Change Local Ollama / Qwen3 4B Configuration
- **File**: `llm/provider.py`
- **Class**: `OllamaProvider`
- **Key Parameters**:
  - `self.model`: Defaults to `"qwen3:4b"`.
  - `self.base_url`: Defaults to `"http://localhost:11434"`.
  - `payload["think"] = False`: Bypasses reasoning trace to prevent timeout.
  - `payload["options"]["temperature"]`: Controls sampling temperature (default `0.0`).
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_llm.py -k test_ollama
  ```

### 5. Add a New LLM Provider Driver
- **File**: `llm/provider.py`
- **Steps**:
  1. Create a class subclassing `LLMProvider`:
     ```python
     class AnthropicProvider(LLMProvider):
         @property
         def name(self) -> str:
             return "anthropic"

         def analyze(self, request: AnalysisRequest) -> AnalysisResult:
             # Implement HTTP call, response parsing, and return AnalysisResult
             ...
     ```
  2. Update factory function `get_llm_provider()` to instantiate the new provider based on `LLM_PROVIDER` in `.env`.
  3. Export new provider in `llm/__init__.py`.
- **Validation**: Add unit tests in `tests/test_llm.py` mocking the HTTP endpoint.

### 6. Change Deterministic Ranking Weights
- **File**: `ranking/models.py`
- **Class**: `RankingWeights`
- **Rule**: All weights must sum to `1.0` ($\pm 0.01$).
  ```python
  @dataclass(frozen=True)
  class RankingWeights:
      relevance: float = 0.40
      novelty: float = 0.25
      recency: float = 0.20
      confidence: float = 0.15
      event_type: float = 0.00
  ```
- **File**: `ranking/scorer.py`
  - Modify `DEFAULT_HALF_LIFE_DAYS = 7.0` to change how quickly old news decays.
  - Modify `NOVELTY_SCORES` or `EVENT_TYPE_SCORES` dictionaries to adjust category points.
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_ranking.py
  ```

### 7. Change Telegram Delivery Behavior
- **File**: `delivery/telegram.py`
- **Key Functions**:
  - `format_telegram_update()`: Formats message title, score, source, and link.
  - `deliver_ranked_articles()`: Filters by `config.min_score` and `config.limit`.
- **Environment Variables** (`.env`):
  - `TELEGRAM_ENABLED=true`
  - `TELEGRAM_BOT_TOKEN=<your_token>`
  - `TELEGRAM_CHAT_ID=<your_chat_id>`
  - `TELEGRAM_MIN_SCORE=0.6`
  - `TELEGRAM_LIMIT=5`
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_delivery.py
  ```

### 8. Change Web Presentation & Dashboard
- **Server Routes**: `web/app.py` (`index()`, `article_detail()`, `about()`).
- **Read-Only Data Layer**: `web/queries.py` (`get_ranked_articles_for_dashboard()`, `clean_human_summary()`, `score_to_stars()`).
- **HTML Templates**:
  - `web/templates/index.html`: Dashboard cards, stats header, filters, pagination.
  - `web/templates/article.html`: Detailed view of summary, takeaways, and entities.
  - `web/templates/base.html`: Common HTML shell, navigation bar, footer.
- **CSS Styles**: `web/static/style.css`.
- **Running the Web App Locally**:
  ```bash
  .venv/bin/python -m uvicorn web.app:app --host 127.0.0.1 --port 5000 --reload
  ```
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_web.py
  ```

### 9. Change Database Schema
- **File**: `database.py`
- **Function**: `init_db()`
- **Guidelines**:
  1. Add table or column definitions to `CREATE TABLE IF NOT EXISTS`.
  2. Implement an idempotent migration check using `PRAGMA table_info(...)` for existing production databases.
  3. Ensure `PRAGMA foreign_keys = ON`.
- **Validation**:
  ```bash
  .venv/bin/python -m pytest tests/test_database.py tests/test_storage.py
  ```

### 10. Running Tests & Linters
- **Run Complete Test Suite**:
  ```bash
  .venv/bin/python -m pytest
  ```
- **Run Subsystem Tests Only**:
  ```bash
  .venv/bin/python -m pytest tests/test_llm.py
  .venv/bin/python -m pytest tests/test_ranking.py
  .venv/bin/python -m pytest tests/test_web.py
  ```
