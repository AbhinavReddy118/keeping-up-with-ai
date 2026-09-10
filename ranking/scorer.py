"""
Deterministic scoring and ranking engine for analyzed articles (Task 3.0).
"""

import math
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, Sequence
from database import get_connection
from ranking.models import ComponentScores, RankedArticle, RankingWeights

logger = logging.getLogger(__name__)

# Explicit mapping for novelty categories (Task 3.0 Requirement 4)
NOVELTY_SCORES = {
    "breakthrough": 1.0,
    "significant": 0.7,
    "incremental": 0.4,
    "unknown": 0.1,
}

# Conservative event_type importance mapping (Task 3.0 Requirement 6)
EVENT_TYPE_SCORES = {
    "model_release": 1.0,
    "research_paper": 0.9,
    "tool_release": 0.8,
    "business_partnership": 0.7,
    "industry_news": 0.6,
    "regulation_policy": 0.6,
    "opinion_editorial": 0.4,
    "unknown": 0.5,
}

# Default recency half-life in days
DEFAULT_HALF_LIFE_DAYS = 7.0

# Safe fallback recency score for missing or unparseable dates
DEFAULT_MISSING_DATE_RECENCY = 0.20


def map_novelty_score(novelty: Optional[str]) -> float:
    """Map novelty category string to a normalized score in [0.0, 1.0]."""
    if not novelty or not isinstance(novelty, str):
        return NOVELTY_SCORES["unknown"]
    clean = novelty.strip().lower()
    return NOVELTY_SCORES.get(clean, NOVELTY_SCORES["unknown"])


def map_event_type_score(event_type: Optional[str]) -> float:
    """Map event type string to a normalized score in [0.0, 1.0]."""
    if not event_type or not isinstance(event_type, str):
        return EVENT_TYPE_SCORES["unknown"]
    clean = event_type.strip().lower()
    return EVENT_TYPE_SCORES.get(clean, EVENT_TYPE_SCORES["unknown"])


def parse_datetime_safe(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO 8601 publication date string into a timezone-aware datetime."""
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        return None
    raw = date_str.strip()
    try:
        # Handle trailing 'Z' for UTC
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def calculate_recency_score(
    published_at: Optional[str],
    reference_time: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """
    Calculate exponential half-life recency decay score in [0.0, 1.0].

    Formula:
        recency = 2 ** (-delta_days / half_life_days)

    Guarantees:
        - Never produces negative scores.
        - Missing or invalid dates return a safe non-negative fallback (0.20).
        - Newer articles receive higher scores.
        - Future dates (slight clock skew) are clamped to 1.0.
    """
    dt = parse_datetime_safe(published_at)
    if dt is None:
        return DEFAULT_MISSING_DATE_RECENCY

    ref = reference_time if reference_time is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    delta_seconds = (ref - dt).total_seconds()
    # Clamp future dates to zero age (score = 1.0)
    delta_days = max(0.0, delta_seconds / 86400.0)

    if half_life_days <= 0:
        half_life_days = DEFAULT_HALF_LIFE_DAYS

    score = math.pow(2.0, -delta_days / half_life_days)
    return max(0.0, min(1.0, score))


def score_article(
    article_data: dict,
    weights: Optional[RankingWeights] = None,
    reference_time: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> Optional[RankedArticle]:
    """
    Compute individual component scores and final weighted score for one article record.
    Returns None if article does not have completed analysis or missing required metrics.
    """
    if not isinstance(article_data, dict):
        return None

    # Verify status is completed (Task 3.0 Requirement 9)
    status = article_data.get("status")
    if status != "completed":
        return None


    relevance_raw = article_data.get("relevance_score")
    if relevance_raw is None:
        return None

    try:
        relevance_score = max(0.0, min(1.0, float(relevance_raw)))
    except (ValueError, TypeError):
        return None

    confidence_raw = article_data.get("confidence", 1.0)
    try:
        confidence_score = max(0.0, min(1.0, float(confidence_raw)))
    except (ValueError, TypeError):
        confidence_score = 0.5

    novelty_label = article_data.get("novelty")
    novelty_score = map_novelty_score(novelty_label)

    published_at = article_data.get("published_at")
    recency_score = calculate_recency_score(
        published_at, reference_time=reference_time, half_life_days=half_life_days
    )

    event_type_label = article_data.get("event_type")
    event_type_score = map_event_type_score(event_type_label)

    w = weights if weights is not None else RankingWeights()

    final_score = (
        (w.relevance * relevance_score)
        + (w.novelty * novelty_score)
        + (w.recency * recency_score)
        + (w.confidence * confidence_score)
        + (w.event_type * event_type_score)
    )

    # Ensure strictly bounded in [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))

    component_scores = ComponentScores(
        relevance=relevance_score,
        novelty=novelty_score,
        recency=recency_score,
        confidence=confidence_score,
        event_type=event_type_score if w.event_type > 0 else None,
    )

    article_id = article_data.get("article_id") or article_data.get("id", 0)

    return RankedArticle(
        article_id=int(article_id),
        title=article_data.get("title", "Untitled"),
        source=article_data.get("source", "Unknown Source"),
        published_at=published_at,
        link=article_data.get("link", ""),
        final_score=final_score,
        component_scores=component_scores,
        summary=article_data.get("summary"),
        event_type=event_type_label,
        novelty_label=novelty_label,
    )


def get_completed_articles_for_ranking(
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """
    Retrieve all articles with completed analysis records using an efficient SQL JOIN.
    Excludes unanalyzed and failed articles.
    """
    query = """
        SELECT 
            a.id as article_id,
            a.title,
            a.link,
            a.source,
            a.published_at,
            an.relevance_score,
            an.novelty,
            an.confidence,
            an.event_type,
            an.summary,
            an.status
        FROM articles a
        JOIN article_analysis an ON a.id = an.article_id
        WHERE an.status = 'completed'
        ORDER BY a.id ASC
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        if should_close:
            conn.close()


def rank_articles(
    articles: Sequence[dict],
    weights: Optional[RankingWeights] = None,
    reference_time: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    limit: Optional[int] = None,
) -> list[RankedArticle]:
    """
    Score and rank a collection of article records deterministically.

    Tie-breaking policy:
        1. final_score DESC (rounded to 6 decimal places to prevent float precision ties)
        2. published_at DESC (newer timestamp first; missing date treated as epoch 0)
        3. article_id ASC (lower article_id first)

    Args:
        articles: Sequence of article dicts (must include analysis columns).
        weights: Optional RankingWeights instance.
        reference_time: Optional reference datetime for recency decay.
        half_life_days: Half-life decay duration in days.
        limit: Optional maximum number of ranked articles to return.

    Returns:
        Sorted list of RankedArticle instances.
    """
    scored: list[RankedArticle] = []

    for art in articles:
        ranked = score_article(
            art,
            weights=weights,
            reference_time=reference_time,
            half_life_days=half_life_days,
        )
        if ranked is not None:
            scored.append(ranked)

    def sort_key(item: RankedArticle):
        dt = parse_datetime_safe(item.published_at)
        epoch = dt.timestamp() if dt is not None else 0.0
        # -final_score for DESC, -epoch for DESC, article_id for ASC
        return (-round(item.final_score, 6), -epoch, item.article_id)

    scored.sort(key=sort_key)

    if limit is not None and limit > 0:
        return scored[:limit]

    return scored
