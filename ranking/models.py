"""
Data models and weight configuration for deterministic article ranking.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RankingWeights:
    """Configurable weights for deterministic scoring components (must sum to ~1.0)."""
    relevance: float = 0.40
    novelty: float = 0.25
    recency: float = 0.20
    confidence: float = 0.15
    event_type: float = 0.00  # Conservative default: 0.0

    def __post_init__(self):
        total = self.relevance + self.novelty + self.recency + self.confidence + self.event_type
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ranking weights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class ComponentScores:
    """Individual normalized score components (each bounded between 0.0 and 1.0)."""
    relevance: float
    novelty: float
    recency: float
    confidence: float
    event_type: Optional[float] = None

    def to_dict(self) -> dict:
        data = {
            "relevance": round(self.relevance, 4),
            "novelty": round(self.novelty, 4),
            "recency": round(self.recency, 4),
            "confidence": round(self.confidence, 4),
        }
        if self.event_type is not None:
            data["event_type"] = round(self.event_type, 4)
        return data


@dataclass
class RankedArticle:
    """A scored and ranked article."""
    article_id: int
    title: str
    source: str
    published_at: Optional[str]
    link: str
    final_score: float
    component_scores: ComponentScores
    summary: Optional[str] = None
    event_type: Optional[str] = None
    novelty_label: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at,
            "link": self.link,
            "final_score": round(self.final_score, 4),
            "component_scores": self.component_scores.to_dict(),
            "summary": self.summary,
            "event_type": self.event_type,
            "novelty_label": self.novelty_label,
        }
