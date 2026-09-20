"""
End-to-end pipeline orchestration.

Chains ingestion, deduplication, LLM analysis, deterministic ranking,
and Telegram delivery into a single command.
"""

import logging
from typing import Optional

from delivery import TelegramConfig, deliver_ranked_articles
from ingestion.pipeline import run_pipeline
from llm.batch import BatchProcessor, get_unprocessed_articles
from ranking import (
    get_completed_articles_for_ranking,
    rank_articles,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    sources: Optional[list[dict]] = None,
    timeout: int = 10,
    deduplicate: bool = True,
    analyze: bool = True,
    analysis_limit: Optional[int] = None,
    max_retries: Optional[int] = None,
    provider=None,
    rank: bool = True,
    rank_limit: int = 10,
    deliver: bool = True,
    telegram_config: Optional[TelegramConfig] = None,
    db_path=None,
) -> dict:
    """
    Execute the complete intelligence pipeline.

    Stages:
        1. Ingestion (fetch, normalize, deduplicate, persist)
        2. LLM batch analysis of unanalyzed articles
        3. Deterministic ranking of completed analyses
        4. Telegram delivery of top ranked stories

    Returns:
        dict with keys ``ingestion``, ``analysis``, ``ranking``, ``delivery``.
    """
    result = {}

    ingestion_summary = run_pipeline(
        sources=sources,
        timeout=timeout,
        deduplicate=deduplicate,
        db_path=db_path,
    )
    result["ingestion"] = ingestion_summary
    logger.info(
        "Ingestion complete: %d fetched, %d inserted, %d duplicates",
        ingestion_summary["total_fetched"],
        ingestion_summary["total_inserted"],
        ingestion_summary["total_duplicates"],
    )

    if analyze:
        processor = BatchProcessor(
            provider=provider,
            batch_size=analysis_limit,
            max_retries=max_retries,
            db_path=db_path,
        )
        report = processor.run(limit=analysis_limit)
        result["analysis"] = {
            "total_attempted": report.total_attempted,
            "completed": report.completed,
            "failed": report.failed,
            "retries_attempted": report.retries_attempted,
        }
        logger.info(
            "Analysis complete: %d attempted, %d completed, %d failed",
            report.total_attempted,
            report.completed,
            report.failed,
        )

    ranked_articles = []
    if rank:
        with _open_connection(db_path) as conn:
            completed = get_completed_articles_for_ranking(conn=conn)
        ranked = rank_articles(completed, limit=rank_limit)
        ranked_articles = [
            {
                "article_id": item.article_id,
                "title": item.title,
                "source": item.source,
                "link": item.link,
                "published_at": item.published_at,
                "final_score": item.final_score,
                "event_type": item.event_type,
                "summary": item.summary,
            }
            for item in ranked
        ]
        result["ranking"] = {
            "count": len(ranked_articles),
            "articles": ranked_articles,
        }
        logger.info("Ranking complete: %d articles ranked", len(ranked_articles))

    if deliver and ranked_articles:
        config = telegram_config or TelegramConfig.from_env()
        delivery_report = deliver_ranked_articles(ranked_articles, config)
        result["delivery"] = delivery_report
        if delivery_report["enabled"]:
            logger.info(
                "Delivery complete: %d sent, %d failed",
                delivery_report["sent"],
                delivery_report["failed"],
            )
        else:
            logger.info("Delivery skipped: Telegram not configured")
    else:
        result["delivery"] = {"enabled": False, "skipped": True}

    return result


def _open_connection(db_path):
    """Helper to open a SQLite connection, importing lazily to avoid cycles."""
    from database import get_connection

    return get_connection(db_path)
