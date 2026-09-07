"""
Ingestion Pipeline Orchestration Module.
Coordinates source loading, RSS fetching, normalization, SQLite persistence, and health auditing.
"""

import logging
import sqlite3
from typing import Optional
from sources import get_sources
from ingestion.rss_reader import fetch_rss_feed
from ingestion.normalizer import normalize_article
from storage import save_article
from database import init_db, get_connection
from ingestion.health import evaluate_source_health, format_health_report

logger = logging.getLogger(__name__)


def format_summary_report(summary: dict) -> str:
    """Format the ingestion summary dictionary using the standard health report."""
    return format_health_report(summary)


def run_pipeline(
    sources: Optional[list[dict]] = None,
    conn: Optional[sqlite3.Connection] = None,
    timeout: int = 10,
) -> dict:
    """
    Execute the end-to-end ingestion pipeline with source reliability and health tracking:
    Sources -> RSS Fetching -> Normalization -> Storage -> Health Evaluation.

    Args:
        sources: Optional list of source dictionaries (defaults to enabled sources from sources.py).
        conn: Optional active SQLite connection (defaults to project database).
        timeout: Timeout in seconds for RSS requests.

    Returns:
        dict: Complete ingestion metrics, health statuses, and per-source diagnostics.
    """
    logger.info("Starting AI Intelligence Monitor ingestion pipeline...")

    # Load sources if not explicitly supplied
    if sources is None:
        sources = get_sources(enabled_only=True)
    else:
        # Filter to enabled sources
        sources = [s for s in sources if s.get("enabled", True)]

    # Prepare database connection
    should_close_conn = False
    if conn is None:
        init_db()
        conn = get_connection()
        should_close_conn = True

    source_summaries = []
    total_fetched = 0
    total_normalized = 0
    total_inserted = 0
    total_duplicates = 0
    total_invalid = 0
    healthy_sources = 0
    warning_sources = 0
    failed_sources = 0

    try:
        for source in sources:
            source_name = source.get("name", "Unknown Source")
            source_url = source.get("url", "")
            logger.info(f"Processing source: '{source_name}' ({source_url})")

            fetch_success = False
            raw_entries = []
            fetch_error = None

            # 1. Fetch entries from feed
            try:
                raw_entries = fetch_rss_feed(source_url, timeout=timeout)
                fetch_success = True
                logger.info(f"Fetched {len(raw_entries)} raw entries from '{source_name}'")
            except Exception as e:
                logger.error(f"Failed to fetch feed for source '{source_name}': {e}")
                fetch_error = str(e)

            # Metrics for this source
            inserted_count = 0
            duplicate_count = 0
            invalid_count = 0
            normalized_count = 0
            missing_dates_count = 0
            valid_dates = []

            # 2. Normalize and Store each entry if fetch succeeded
            if fetch_success:
                for entry in raw_entries:
                    try:
                        article = normalize_article(entry, source)
                        normalized_count += 1
                        pub_date = article.get("published_at")
                        if pub_date:
                            valid_dates.append(pub_date)
                        else:
                            missing_dates_count += 1
                    except Exception as norm_err:
                        logger.warning(f"Invalid entry in source '{source_name}': {norm_err}")
                        invalid_count += 1
                        continue

                    # 3. Store normalized article
                    try:
                        was_inserted = save_article(article, conn=conn)
                        if was_inserted:
                            inserted_count += 1
                        else:
                            duplicate_count += 1
                    except Exception as store_err:
                        logger.error(f"Storage error for article from '{source_name}': {store_err}")
                        invalid_count += 1

                # Commit changes for this source
                try:
                    conn.commit()
                except Exception as commit_err:
                    logger.error(f"Failed to commit transaction for '{source_name}': {commit_err}")

            oldest_pub = min(valid_dates) if valid_dates else None
            newest_pub = max(valid_dates) if valid_dates else None

            # 4. Evaluate health status deterministically
            src_health = evaluate_source_health(
                name=source_name,
                url=source_url,
                fetch_success=fetch_success,
                fetched=len(raw_entries),
                normalized=normalized_count,
                inserted=inserted_count,
                duplicates=duplicate_count,
                invalid=invalid_count,
                missing_dates=missing_dates_count,
                newest_published_at=newest_pub,
                oldest_published_at=oldest_pub,
                error=fetch_error,
            )

            if src_health["status"] == "HEALTHY":
                healthy_sources += 1
            elif src_health["status"] == "WARNING":
                warning_sources += 1
            else:
                failed_sources += 1

            source_summaries.append(src_health)

            total_fetched += src_health["fetched"]
            total_normalized += src_health["normalized"]
            total_inserted += src_health["inserted"]
            total_duplicates += src_health["duplicates"]
            total_invalid += src_health["invalid"]

            logger.info(
                f"Completed '{source_name}' [{src_health['status']}]: "
                f"{src_health['inserted']} inserted, {src_health['duplicates']} duplicates, "
                f"{src_health['invalid']} invalid"
            )

    finally:
        if should_close_conn:
            try:
                conn.commit()
            except Exception:
                pass
            conn.close()

    summary = {
        "total_sources": len(sources),
        "successful_sources": healthy_sources + warning_sources,
        "healthy_sources": healthy_sources,
        "warning_sources": warning_sources,
        "failed_sources": failed_sources,
        "total_fetched": total_fetched,
        "total_normalized": total_normalized,
        "total_inserted": total_inserted,
        "total_duplicates": total_duplicates,
        "total_invalid": total_invalid,
        "sources": source_summaries,
    }

    # Print formatted health report
    print(format_health_report(summary))

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_pipeline()
