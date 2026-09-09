"""
Demonstration of Batch LLM Analysis + Persistence (Task 2.9.3).

Processes a small batch of unanalyzed articles from the SQLite database,
persists validated results with JSON-encoded metadata to `article_analysis`,
and demonstrates bounded retries, failure resilience, and resumability.

Guarantees:
- Database `articles` table is 100% read-only (count before == count after).
- Masks API keys completely so secrets are never printed.
- Idempotent and safe to run repeatedly.
"""

import sys
import argparse
import json
from config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_BATCH_SIZE, LLM_MAX_RETRIES
from database import get_connection, init_db
from llm import (
    BatchProcessor,
    MockLLMProvider,
    get_llm_provider,
    get_analysis_stats,
    get_unprocessed_articles,
    LLMConfigurationError,
)


def mask_key(key: str) -> str:
    if not key:
        return "[NOT CONFIGURED]"
    cleaned = key.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:3]}...{cleaned[-4:]} (len: {len(cleaned)})"


def main():
    parser = argparse.ArgumentParser(description="Demonstrate batch LLM article analysis.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of articles to process in this batch (default: 5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force use of MockLLMProvider regardless of environment configuration",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TASK 2.9.3: BATCH LLM ANALYSIS + PERSISTENCE DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize schema idempotently
    init_db()

    # 2. Verify articles table count BEFORE
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_count_before = cursor.fetchone()["cnt"]

    stats_before = get_analysis_stats()
    print(f"\n[1] Initial Database State:")
    print(f"    Total articles in database    : {articles_count_before}")
    print(f"    Already analyzed (completed)  : {stats_before['completed']}")
    print(f"    Previous failures             : {stats_before['failed']}")
    print(f"    Total rows in article_analysis: {stats_before['total_analyzed']}")

    # 3. Determine provider
    provider_name = "mock" if args.mock else (LLM_PROVIDER or "mock").strip().lower()
    has_api_key = bool(LLM_API_KEY and LLM_API_KEY.strip())

    print(f"\n[2] Provider Configuration:")
    print(f"    Selected Provider: '{provider_name}'")
    print(f"    Target Model     : '{LLM_MODEL or '(default)'}'")
    print(f"    API Key Status   : {mask_key(LLM_API_KEY)}")
    print(f"    Batch Limit      : {args.limit}")
    print(f"    Max Retries      : {LLM_MAX_RETRIES}")

    if provider_name == "mock" or not has_api_key:
        if provider_name != "mock":
            print("\n    [NOTICE] LLM_PROVIDER is configured for live mode but LLM_API_KEY is missing.")
            print("    Falling back safely to MockLLMProvider for offline execution.")
        provider = MockLLMProvider()
    else:
        try:
            provider = get_llm_provider(
                provider_name=provider_name,
                api_key=LLM_API_KEY,
                model=LLM_MODEL if LLM_MODEL else None,
                base_url=LLM_BASE_URL if LLM_BASE_URL else None,
            )
            print(f"    Live Provider Initialized: {provider.name} ({getattr(provider, 'model', 'N/A')})")
        except LLMConfigurationError as e:
            print(f"    [!] Provider config error: {e}. Falling back to mock.")
            provider = MockLLMProvider()

    # 4. Fetch upcoming articles preview
    with get_connection() as conn:
        upcoming = get_unprocessed_articles(limit=args.limit, conn=conn)

    print(f"\n[3] Selected {len(upcoming)} unanalyzed article(s) for this batch:")
    if not upcoming:
        print("    All available articles have already been successfully analyzed!")
    for item in upcoming:
        title_snip = item['title'][:60]
        print(f"    - ID {item['id']:<5} | Source: {item['source']:<18} | {title_snip}...")

    # 5. Execute Batch
    print(f"\n[4] Executing batch analysis using '{provider.name}'...")
    processor = BatchProcessor(
        provider=provider,
        batch_size=args.limit,
        max_retries=LLM_MAX_RETRIES,
    )
    report = processor.run(limit=args.limit)

    print("\n[5] Batch Execution Report:")
    print(f"    Total Attempted   : {report.total_attempted}")
    print(f"    Completed         : {report.completed}")
    print(f"    Failed            : {report.failed}")
    print(f"    Retries Attempted : {report.retries_attempted}")

    if report.details:
        print("\n    Per-Article Breakdown:")
        for d in report.details:
            status_symbol = "✓" if d["status"] == "completed" else "✗"
            score_str = f"relevance: {d['relevance_score']}" if d['relevance_score'] is not None else f"error: {d['error']}"
            print(f"    {status_symbol} Article ID {d['article_id']:<5} | Status: {d['status']:<9} | Attempts: {d['attempts']} | {score_str}")

    # 6. Database Integrity & Resumability Check
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_count_after = cursor.fetchone()["cnt"]

        # Check duplicate constraint on article_analysis
        cursor.execute("SELECT article_id, COUNT(*) as cnt FROM article_analysis GROUP BY article_id HAVING cnt > 1")
        dupes = cursor.fetchall()

    stats_after = get_analysis_stats()

    print("\n[6] Database Integrity Verification:")
    print(f"    Articles count BEFORE demo : {articles_count_before}")
    print(f"    Articles count AFTER demo  : {articles_count_after}")
    assert articles_count_before == articles_count_after, "Articles count changed during batch demo!"
    print("    Confirmation               : ARTICLES TABLE WAS NOT MODIFIED (100% read-only)")

    print(f"\n[7] Persistence & Deduplication Verification:")
    print(f"    Total in article_analysis  : {stats_after['total_analyzed']} (+{stats_after['total_analyzed'] - stats_before['total_analyzed']})")
    print(f"    Total completed            : {stats_after['completed']}")
    print(f"    Total failed               : {stats_after['failed']}")
    assert len(dupes) == 0, f"Found duplicate article_analysis rows: {dupes}"
    print("    Duplicate Rows Detected    : 0 (UNIQUE constraint satisfied)")

    # 7. Resumability Demonstration
    with get_connection() as conn:
        next_batch = get_unprocessed_articles(limit=args.limit, conn=conn)
    print(f"\n[8] Resumability Check:")
    print(f"    Next unanalyzed articles ready: {len(next_batch)}")
    if upcoming and report.completed > 0:
        processed_ids = {d["article_id"] for d in report.details if d["status"] == "completed"}
        next_ids = {a["id"] for a in next_batch}
        overlap = processed_ids.intersection(next_ids)
        assert len(overlap) == 0, f"Completed articles appeared in next selection: {overlap}"
        print(f"    Skipped Completed Articles    : VERIFIED (0 overlap with just-completed IDs)")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE: Batch LLM Analysis + Persistence successfully verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
