"""
Article Ranking Layer (Task 3.0).
Provides deterministic scoring, novelty mapping, recency half-life decay, and tie-breaking.
"""

from ranking.models import ComponentScores, RankedArticle, RankingWeights
from ranking.scorer import (
    NOVELTY_SCORES,
    EVENT_TYPE_SCORES,
    DEFAULT_HALF_LIFE_DAYS,
    map_novelty_score,
    map_event_type_score,
    calculate_recency_score,
    score_article,
    rank_articles,
    get_completed_articles_for_ranking,
)

__all__ = [
    "ComponentScores",
    "RankedArticle",
    "RankingWeights",
    "NOVELTY_SCORES",
    "EVENT_TYPE_SCORES",
    "DEFAULT_HALF_LIFE_DAYS",
    "map_novelty_score",
    "map_event_type_score",
    "calculate_recency_score",
    "score_article",
    "rank_articles",
    "get_completed_articles_for_ranking",
]
