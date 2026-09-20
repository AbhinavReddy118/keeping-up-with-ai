"""
End-to-end pipeline orchestration.

Chains ingestion, deduplication, LLM analysis, deterministic ranking,
and Telegram delivery into a single command.
"""

import logging
import os
import re
from typing import Optional

from delivery import TelegramConfig, deliver_ranked_articles
from ingestion.pipeline import run_pipeline
from llm.batch import BatchProcessor, get_unprocessed_articles
from ranking import (
    get_completed_articles_for_ranking,
    rank_articles,
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _load_topic_weights() -> dict[str, float]:
    """Read topic weights from TOPIC_WEIGHTS env var.

    Format: "agent:1.5,reasoning:1.3,tool:1.2".
    Empty / missing => empty dict (no personalization).
    """
    raw = os.environ.get("TOPIC_WEIGHTS", "").strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, val = pair.split(":", 1)
        key = key.strip().lower()
        try:
            w = float(val.strip())
        except ValueError:
            continue
        if key and w > 0:
            out[key] = w
    return out


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def apply_personalization(
    ranked_articles: list[dict],
    topic_weights: Optional[dict[str, float]] = None,
    bonus_cap: float = 0.20,
) -> list[dict]:
    """Re-rank after a topic-weight nudge.

    Each article's final_score gets a bonus in [0, bonus_cap] based on whether
    any of its topic keywords appears in the title+summary. The deterministic
    ranking itself is unchanged; only the score used for delivery is shifted.

    Bonus is additive, capped, and never demotes an article below the original.
    """
    if not ranked_articles or not topic_weights:
        return ranked_articles

    # Pre-tokenize keywords once
    keyword_tokens: dict[str, set[str]] = {
        k: {t for t in _WORD_RE.findall(k.lower()) if t}
        for k in topic_weights
    }

    adjusted: list[dict] = []
    for art in ranked_articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        toks = _tokenize(text)
        bonus = 0.0
        for keyword, weight in topic_weights.items():
            kw_tokens = keyword_tokens.get(keyword) or set()
            if kw_tokens & toks:
                # weight=1.0 => no nudge; weight=1.5 => +50% of cap; etc.
                bonus += bonus_cap * max(0.0, (weight - 1.0))
        bonus = min(bonus, bonus_cap)

        original = float(art.get("final_score", 0.0) or 0.0)
        new_score = min(1.0, original + bonus)

        new_art = dict(art)
        new_art["final_score"] = round(new_score, 6)
        new_art["_personalization_bonus"] = round(bonus, 6)
        adjusted.append(new_art)

    def _sort_key(a: dict) -> tuple:
        # Mirror rank_articles' tie-breaking: score DESC, published_at DESC, id ASC
        pub = a.get("published_at")
        epoch = 0.0
        if pub:
            try:
                from ranking.scorer import parse_datetime_safe

                dt = parse_datetime_safe(pub)
                if dt is not None:
                    epoch = dt.timestamp()
            except Exception:
                epoch = 0.0
        return (-round(float(a["final_score"]), 6), -epoch, a.get("article_id", 0))

    adjusted.sort(key=_sort_key)
    return adjusted


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
    topic_weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    Execute the complete intelligence pipeline.

    Stages:
        1. Ingestion (fetch, normalize, deduplicate, persist)
        2. LLM batch analysis of unanalyzed articles
        3. Deterministic ranking of completed analyses
        4. Optional topic-weight personalization re-rank
        5. Telegram delivery of top ranked stories

    Returns:
        dict with keys ``ingestion``, ``analysis``, ``ranking``, ``delivery``.
    """
    if topic_weights is None:
        topic_weights = _load_topic_weights()

    result = {
        "personalization": {
            "enabled": bool(topic_weights),
            "weights": dict(topic_weights),
        }
    }

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
        if topic_weights:
            ranked_articles = apply_personalization(ranked_articles, topic_weights)
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
