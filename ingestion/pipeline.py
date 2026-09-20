"""
Ingestion pipeline orchestration.

The ingestion stage collects RSS/Atom entries, normalizes them, applies
progressive deduplication, persists the surviving articles, and reports
source health.
"""

import logging
from collections import defaultdict
from typing import Optional

from database import get_connection, init_db
from deduplication import deduplicate_articles
from ingestion.health import evaluate_source_health, format_health_report
from ingestion.normalizer import normalize_article
from ingestion.rss_reader import fetch_rss_feed
from sources import get_sources
from storage import save_article

logger = logging.getLogger(__name__)


def format_summary_report(summary: dict) -> str:
    """Format the ingestion summary dictionary using the standard health report."""
    return format_health_report(summary)


def _collect_entries(sources: list[dict], timeout: int) -> dict:
    collected_articles = []
    source_articles = []
    source_summaries = []
    total_fetched = 0
    total_normalized = 0
    total_invalid = 0

    for source in sources:
        source_name = source.get("name", "Unknown Source")
        source_url = source.get("url", "")
        logger.info("Processing source: '%s' (%s)", source_name, source_url)

        fetch_success = False
        raw_entries = []
        fetch_error = None
        inserted_count = 0
        duplicate_count = 0
        invalid_count = 0
        normalized_count = 0
        missing_dates_count = 0
        valid_dates = []
        current_source_articles = []

        try:
            raw_entries = fetch_rss_feed(source_url, timeout=timeout)
            fetch_success = True
            logger.info(
                "Fetched %d raw entries from '%s'", len(raw_entries), source_name
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch feed for source '%s': %s", source_name, exc
            )
            fetch_error = str(exc)

        if fetch_success:
            for entry in raw_entries:
                try:
                    article = normalize_article(entry, source)
                    current_source_articles.append(article)
                    collected_articles.append(article)
                    normalized_count += 1
                    published_at = article.get("published_at")
                    if published_at:
                        valid_dates.append(published_at)
                    else:
                        missing_dates_count += 1
                except Exception as normalization_error:
                    logger.warning(
                        "Invalid entry in source '%s': %s",
                        source_name,
                        normalization_error,
                    )
                    invalid_count += 1

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
            newest_published_at=(
                max(valid_dates) if valid_dates else None
            ),
            oldest_published_at=(
                min(valid_dates) if valid_dates else None
            ),
            error=fetch_error,
        )
        source_summaries.append(src_health)
        if current_source_articles:
            source_articles.append(current_source_articles)

        total_fetched += src_health["fetched"]
        total_normalized += src_health["normalized"]
        total_invalid += src_health["invalid"]
        logger.info(
            "Completed '%s' [fetch: %s]: %d normalized, %d invalid",
            source_name,
            "ok" if fetch_success else "failed",
            normalized_count,
            invalid_count,
        )

    return {
        "articles": collected_articles,
        "source_articles": source_articles,
        "sources": source_summaries,
        "total_fetched": total_fetched,
        "total_normalized": total_normalized,
        "total_invalid": total_invalid,
    }


def run_pipeline(
    sources: Optional[list[dict]] = None,
    conn=None,
    timeout: int = 10,
    deduplicate: bool = False,
    db_path=None,
) -> dict:
    """
    Execute collection, optional progressive deduplication, persistence, and health reporting.

    Args:
        sources: Optional list of source dictionaries (defaults to enabled sources).
        conn: Optional active SQLite connection.
        timeout: Timeout in seconds for RSS requests.
        deduplicate: Remove URL/title/content candidates before persistence.
        db_path: Optional database path used when conn is not supplied.

    Returns:
        Complete ingestion metrics, health statuses, and optional deduplication metrics.
    """
    logger.info("Starting AI Intelligence Monitor ingestion pipeline...")

    if sources is not None:
        sources = [source for source in sources if source.get("enabled", True)]
    else:
        sources = get_sources(enabled_only=True)

    should_close = False
    if conn is None:
        init_db(db_path)
        conn = get_connection(db_path)
        should_close = True

    try:
        collected = _collect_entries(sources, timeout)
        articles = collected["articles"]
        source_articles = collected["source_articles"]
        article_source_names = [
            article.get("source", "Unknown Source") for article in articles
        ]
        representative_for = list(range(len(articles)))
        deduplication_result = None

        if deduplicate and articles:
            deduplication_result = deduplicate_articles(articles)
            articles = deduplication_result["articles"]
            representative_for = deduplication_result["representative_for"]

        saved_indices = set()
        failed_indices = set()
        for unique_index, article in enumerate(articles):
            try:
                if save_article(article, conn=conn):
                    saved_indices.add(unique_index)
            except Exception as storage_error:
                logger.error(
                    "Storage error for article from source '%s': %s",
                    article.get("source", "Unknown Source"),
                    storage_error,
                )
                failed_indices.add(unique_index)

        try:
            conn.commit()
        except Exception:
            logger.exception("Failed to commit ingestion transaction")

        inserted_by_source = defaultdict(int)
        duplicate_by_source = defaultdict(int)
        invalid_by_source = defaultdict(int)

        original_source_names = [
            name
            for source_entries in source_articles
            for _ in source_entries
            for name in [source_entries[0].get("source", "Unknown Source") if source_entries else "Unknown Source"]
        ]
        # Build original->unique index mapping and unique->original source name
        unique_source_names = []
        for unique_index in range(len(articles)):
            for original_index, rep_unique_index in enumerate(representative_for):
                if rep_unique_index == unique_index:
                    unique_source_names.append(article_source_names[original_index])
                    break

        global_index = 0
        for source_entries in source_articles:
            for article in source_entries:
                if global_index >= len(representative_for):
                    break
                unique_index = representative_for[global_index]
                source_name = article_source_names[global_index]
                representative_source = (
                    unique_source_names[unique_index]
                    if unique_index < len(unique_source_names)
                    else source_name
                )
                if (
                    unique_index in saved_indices
                    and source_name == representative_source
                ):
                    inserted_by_source[source_name] += 1
                elif (
                    unique_index in failed_indices
                    and source_name == representative_source
                ):
                    invalid_by_source[source_name] += 1
                else:
                    duplicate_by_source[source_name] += 1
                global_index += 1
            if global_index >= len(representative_for):
                break

        source_summaries = []
        for source, initial_summary in zip(sources, collected["sources"]):
            source_name = source.get("name", "Unknown Source")
            health = dict(initial_summary)
            health["inserted"] = inserted_by_source[source_name]
            health["duplicates"] = duplicate_by_source[source_name]
            health["invalid"] += invalid_by_source[source_name]
            health = evaluate_source_health(
                name=source_name,
                url=source.get("url", ""),
                fetch_success=health["fetch_success"],
                fetched=health["fetched"],
                normalized=health["normalized"],
                inserted=health["inserted"],
                duplicates=health["duplicates"],
                invalid=health["invalid"],
                missing_dates=health["missing_dates"],
                newest_published_at=health["newest_published_at"],
                oldest_published_at=health["oldest_published_at"],
                error=health["error"],
            )
            source_summaries.append(health)

        total_inserted = sum(inserted_by_source.values())
        total_duplicates = sum(duplicate_by_source.values())
        total_invalid = collected["total_invalid"] + sum(invalid_by_source.values())
        healthy_sources = sum(
            source["status"] == "HEALTHY" for source in source_summaries
        )
        warning_sources = sum(
            source["status"] == "WARNING" for source in source_summaries
        )
        failed_sources = sum(
            source["status"] == "FAILED" for source in source_summaries
        )

        summary = {
            "total_sources": len(sources),
            "successful_sources": healthy_sources + warning_sources,
            "healthy_sources": healthy_sources,
            "warning_sources": warning_sources,
            "failed_sources": failed_sources,
            "total_fetched": collected["total_fetched"],
            "total_normalized": collected["total_normalized"],
            "total_inserted": total_inserted,
            "total_duplicates": total_duplicates,
            "total_invalid": total_invalid,
            "sources": source_summaries,
        }
        if deduplication_result is not None:
            summary["deduplicated"] = deduplication_result["duplicates"]
            summary["deduplication"] = deduplication_result["report"]

        print(format_summary_report(summary))
        return summary
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_pipeline()
