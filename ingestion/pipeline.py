"""
Ingestion Pipeline Orchestration Module.
Coordinates source loading, RSS fetching, normalization, and SQLite persistence.
"""

import logging
import sqlite3
from typing import Optional
from sources import get_sources
from ingestion.rss_reader import fetch_rss_feed
from ingestion.normalizer import normalize_article
from storage import save_article
from database import init_db, get_connection

logger = logging.getLogger(__name__)


def format_summary_report(summary: dict) -> str:
    """Format the ingestion summary dictionary into a clean human-readable table."""
    lines = [
        "\n" + "=" * 80,
        "INGESTION PIPELINE EXECUTION SUMMARY",
        "=" * 80,
        f"{'Source Name':<28} | {'Fetched':<7} | {'Norm':<5} | {'Insert':<6} | {'Dupl':<5} | {'Inval':<5} | {'Status':<7}",
        "-" * 80,
    ]

    for s in summary["sources"]:
        status_display = s["status"].upper()
        lines.append(
            f"{s['name'][:28]:<28} | {s['fetched']:<7} | {s['normalized']:<5} | "
            f"{s['inserted']:<6} | {s['duplicates']:<5} | {s['invalid']:<5} | {status_display:<7}"
        )
        if s["error"]:
            lines.append(f"  -> Error: {s['error']}")

    lines.extend([
        "-" * 80,
        f"TOTALS: Sources: {summary['total_sources']} (Succeeded: {summary['successful_sources']}, Failed: {summary['failed_sources']})",
        f"        Fetched: {summary['total_fetched']} | Normalized: {summary['total_normalized']} | "
        f"Inserted: {summary['total_inserted']} | Duplicates: {summary['total_duplicates']} | Invalid: {summary['total_invalid']}",
        "=" * 80 + "\n",
    ])
    return "\n".join(lines)


def run_pipeline(
    sources: Optional[list[dict]] = None,
    conn: Optional[sqlite3.Connection] = None,
    timeout: int = 10,
) -> dict:
    """
    Execute the end-to-end ingestion pipeline:
    Sources -> RSS Fetching -> Normalization -> Storage.

    Args:
        sources: Optional list of source dictionaries (defaults to enabled sources from sources.py).
        conn: Optional active SQLite connection (defaults to project database).
        timeout: Timeout in seconds for RSS requests.

    Returns:
        dict: Complete ingestion metrics and per-source breakdown.
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
    successful_sources = 0
    failed_sources = 0

    try:
        for source in sources:
            source_name = source.get("name", "Unknown Source")
            source_url = source.get("url", "")
            logger.info(f"Processing source: '{source_name}' ({source_url})")

            src_summary = {
                "name": source_name,
                "url": source_url,
                "status": "success",
                "fetched": 0,
                "normalized": 0,
                "inserted": 0,
                "duplicates": 0,
                "invalid": 0,
                "error": None,
            }

            # 1. Fetch entries from feed
            try:
                raw_entries = fetch_rss_feed(source_url, timeout=timeout)
                src_summary["fetched"] = len(raw_entries)
                logger.info(f"Fetched {len(raw_entries)} raw entries from '{source_name}'")
            except Exception as e:
                logger.error(f"Failed to fetch feed for source '{source_name}': {e}")
                src_summary["status"] = "failed"
                src_summary["error"] = str(e)
                failed_sources += 1
                source_summaries.append(src_summary)
                continue  # Continue to the next source!

            # 2. Normalize and Store each entry
            for entry in raw_entries:
                try:
                    article = normalize_article(entry, source)
                    src_summary["normalized"] += 1
                except Exception as norm_err:
                    logger.warning(f"Invalid entry in source '{source_name}': {norm_err}")
                    src_summary["invalid"] += 1
                    continue

                # 3. Store normalized article
                try:
                    was_inserted = save_article(article, conn=conn)
                    if was_inserted:
                        src_summary["inserted"] += 1
                    else:
                        src_summary["duplicates"] += 1
                except Exception as store_err:
                    logger.error(f"Storage error for article from '{source_name}': {store_err}")
                    src_summary["invalid"] += 1

            # Commit changes for this source
            try:
                conn.commit()
            except Exception as commit_err:
                logger.error(f"Failed to commit transaction for '{source_name}': {commit_err}")

            successful_sources += 1
            source_summaries.append(src_summary)

            total_fetched += src_summary["fetched"]
            total_normalized += src_summary["normalized"]
            total_inserted += src_summary["inserted"]
            total_duplicates += src_summary["duplicates"]
            total_invalid += src_summary["invalid"]

            logger.info(
                f"Completed '{source_name}': {src_summary['inserted']} inserted, "
                f"{src_summary['duplicates']} duplicates, {src_summary['invalid']} invalid"
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
        "successful_sources": successful_sources,
        "failed_sources": failed_sources,
        "total_fetched": total_fetched,
        "total_normalized": total_normalized,
        "total_inserted": total_inserted,
        "total_duplicates": total_duplicates,
        "total_invalid": total_invalid,
        "sources": source_summaries,
    }

    # Print human-readable summary
    print(format_summary_report(summary))

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_pipeline()
