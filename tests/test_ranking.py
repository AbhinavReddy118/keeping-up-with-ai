"""
Offline unit tests for Task 3.0: Deterministic Article Ranking.
Guarantees zero network calls and full database immutability.
"""

from datetime import datetime, timezone, timedelta
import pytest
from database import init_db, get_connection
from storage import save_article
from llm.batch import save_analysis_success, save_analysis_failure
from llm.models import AnalysisResult
from ranking import (
    RankedArticle,
    ComponentScores,
    RankingWeights,
    NOVELTY_SCORES,
    EVENT_TYPE_SCORES,
    map_novelty_score,
    calculate_recency_score,
    score_article,
    rank_articles,
    get_completed_articles_for_ranking,
)


@pytest.fixture
def temp_ranking_db(tmp_path):
    """Create an isolated test database initialized with article and article_analysis tables."""
    db_file = tmp_path / "test_ranking.db"
    init_db(str(db_file))
    return str(db_file)


@pytest.fixture
def sample_raw_articles():
    """A collection of heterogeneous article records with analysis data for testing."""
    return [
        {
            "article_id": 1,
            "title": "Breakthrough in Reasoning LLMs",
            "source": "OpenAI News",
            "published_at": "2026-09-09T12:00:00Z",
            "link": "https://example.com/1",
            "relevance_score": 0.95,
            "novelty": "breakthrough",
            "confidence": 0.98,
            "event_type": "model_release",
            "status": "completed",
        },
        {
            "article_id": 2,
            "title": "Minor Python Library Update",
            "source": "Tech Blog",
            "published_at": "2026-09-09T10:00:00Z",
            "link": "https://example.com/2",
            "relevance_score": 0.40,
            "novelty": "incremental",
            "confidence": 0.90,
            "event_type": "tool_release",
            "status": "completed",
        },
        {
            "article_id": 3,
            "title": "Significant AI Safety Research Paper",
            "source": "ArXiv AI",
            "published_at": "2026-09-08T12:00:00Z",
            "link": "https://example.com/3",
            "relevance_score": 0.85,
            "novelty": "significant",
            "confidence": 0.95,
            "event_type": "research_paper",
            "status": "completed",
        },
    ]


def test_highest_relevance_article_ranking(sample_raw_articles):
    """1. High relevance combined with high novelty ranks at the top."""
    ranked = rank_articles(
        sample_raw_articles,
        reference_time=datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert len(ranked) == 3
    # Article 1 has 0.95 relevance and breakthrough novelty
    assert ranked[0].article_id == 1
    assert ranked[0].final_score > ranked[1].final_score > ranked[2].final_score
    assert ranked[2].article_id == 2


def test_novelty_mapping():
    """2. Explicit novelty mapping conforms to defined score values."""
    assert map_novelty_score("breakthrough") == 1.0
    assert map_novelty_score("significant") == 0.7
    assert map_novelty_score("incremental") == 0.4
    assert map_novelty_score("unknown") == 0.1
    # Case insensitivity and fallback
    assert map_novelty_score("BreakThrough") == 1.0
    assert map_novelty_score(None) == 0.1
    assert map_novelty_score("non_existent") == 0.1


def test_recency_behavior():
    """3. Recency decay gives newer articles higher scores and clamps future dates."""
    ref_time = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Exactly at reference time -> age = 0 -> recency = 1.0
    s_now = calculate_recency_score("2026-09-10T12:00:00Z", reference_time=ref_time)
    assert s_now == 1.0

    # 2. Exactly 7 days old (1 half-life) -> recency = 0.5
    s_7days = calculate_recency_score("2026-09-03T12:00:00Z", reference_time=ref_time)
    assert pytest.approx(s_7days, 0.001) == 0.50

    # 3. Exactly 14 days old (2 half-lives) -> recency = 0.25
    s_14days = calculate_recency_score("2026-08-27T12:00:00Z", reference_time=ref_time)
    assert pytest.approx(s_14days, 0.001) == 0.25

    # 4. Newer article has higher recency score than older article
    assert s_now > s_7days > s_14days

    # 5. Future date (clock skew) clamped to 1.0 (never exceeds 1.0)
    s_future = calculate_recency_score("2026-09-15T12:00:00Z", reference_time=ref_time)
    assert s_future == 1.0

    # 6. Never produces negative scores even for very old articles
    s_ancient = calculate_recency_score("2020-01-01T00:00:00Z", reference_time=ref_time)
    assert s_ancient > 0.0
    assert s_ancient < 0.01


def test_confidence_behavior():
    """4. Confidence score properly influences the final score."""
    base_article = {
        "article_id": 10,
        "title": "Test Confidence Impact",
        "source": "Source",
        "published_at": "2026-09-09T12:00:00Z",
        "relevance_score": 0.8,
        "novelty": "significant",
        "status": "completed",
    }
    art_high_conf = dict(base_article, confidence=1.0)
    art_low_conf = dict(base_article, confidence=0.2)

    score_high = score_article(art_high_conf).final_score
    score_low = score_article(art_low_conf).final_score

    assert score_high > score_low


def test_missing_publication_date_handling():
    """5. Missing or unparseable publication dates produce valid bounded fallback scores."""
    art_no_date = {
        "article_id": 20,
        "title": "Article With No Date",
        "source": "Source",
        "published_at": None,
        "relevance_score": 0.8,
        "novelty": "incremental",
        "confidence": 0.9,
        "status": "completed",
    }
    ranked = score_article(art_no_date)
    assert ranked is not None
    assert ranked.component_scores.recency == 0.20
    assert 0.0 <= ranked.final_score <= 1.0

    # Malformed date string
    art_bad_date = dict(art_no_date, published_at="not-a-date")
    ranked_bad = score_article(art_bad_date)
    assert ranked_bad is not None
    assert ranked_bad.component_scores.recency == 0.20


def test_missing_analysis_does_not_crash():
    """6. Articles with missing analysis fields are safely excluded without crashing."""
    raw_articles = [
        {"article_id": 1, "title": "Incomplete Article", "status": "completed"},  # Missing relevance_score
        {"article_id": 2, "title": "Missing Status Article", "relevance_score": 0.8},  # Missing status
        None,  # Malformed input
        "not-a-dict",
    ]
    ranked = rank_articles(raw_articles)
    assert len(ranked) == 0


def test_failed_analysis_exclusion():
    """7. Articles with status 'failed' are excluded from ranking."""
    articles = [
        {
            "article_id": 31,
            "title": "Failed Article",
            "source": "Source",
            "status": "failed",
            "relevance_score": 0.9,
            "novelty": "breakthrough",
        },
        {
            "article_id": 32,
            "title": "Completed Article",
            "source": "Source",
            "status": "completed",
            "relevance_score": 0.7,
            "novelty": "incremental",
        },
    ]
    ranked = rank_articles(articles)
    assert len(ranked) == 1
    assert ranked[0].article_id == 32


def test_score_bounds():
    """8. Final score and all component scores are strictly bounded in [0.0, 1.0]."""
    extreme_article = {
        "article_id": 40,
        "title": "Extreme Bounds Test",
        "source": "Source",
        "published_at": "2026-09-09T12:00:00Z",
        "relevance_score": 1.5,  # Out of bounds high
        "novelty": "breakthrough",
        "confidence": 2.0,  # Out of bounds high
        "status": "completed",
    }
    ranked = score_article(extreme_article)
    assert ranked is not None
    assert ranked.final_score <= 1.0
    assert ranked.component_scores.relevance <= 1.0
    assert ranked.component_scores.confidence <= 1.0
    assert ranked.component_scores.novelty <= 1.0
    assert ranked.component_scores.recency <= 1.0


def test_deterministic_tie_breaking():
    """9. Articles with identical final scores break ties deterministically by published_at DESC, then article_id ASC."""
    articles = [
        {
            "article_id": 52,
            "title": "Older Duplicate",
            "source": "Source",
            "published_at": "2026-09-05T12:00:00Z",
            "relevance_score": 0.8,
            "novelty": "incremental",
            "confidence": 0.9,
            "status": "completed",
        },
        {
            "article_id": 51,
            "title": "Newer Duplicate A",
            "source": "Source",
            "published_at": "2026-09-09T12:00:00Z",
            "relevance_score": 0.8,
            "novelty": "incremental",
            "confidence": 0.9,
            "status": "completed",
        },
        {
            "article_id": 50,
            "title": "Newer Duplicate B",
            "source": "Source",
            "published_at": "2026-09-09T12:00:00Z",
            "relevance_score": 0.8,
            "novelty": "incremental",
            "confidence": 0.9,
            "status": "completed",
        },
    ]
    # Fixed reference time so recency is identical between ID 51 and 50
    ref_time = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    ranked = rank_articles(articles, reference_time=ref_time)

    # 1. Newer date (2026-09-09) beats older date (2026-09-05)
    # 2. For identical dates and scores, lower ID (50) beats higher ID (51)
    assert ranked[0].article_id == 50
    assert ranked[1].article_id == 51
    assert ranked[2].article_id == 52


def test_empty_database_handling():
    """10. Empty collection produces an empty ranked list without error."""
    assert rank_articles([]) == []


def test_repeated_execution_producing_identical_ordering(sample_raw_articles):
    """11 & 12. Ranking is 100% deterministic across repeated executions."""
    ref_time = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    run1 = [a.article_id for a in rank_articles(sample_raw_articles, reference_time=ref_time)]
    run2 = [a.article_id for a in rank_articles(sample_raw_articles, reference_time=ref_time)]
    run3 = [a.article_id for a in rank_articles(sample_raw_articles, reference_time=ref_time)]
    assert run1 == run2 == run3


def test_sql_join_repository_and_database_immutability(temp_ranking_db):
    """13. get_completed_articles_for_ranking correctly queries SQL JOIN and never alters articles table."""
    # 1. Insert 2 articles into articles table
    with get_connection(temp_ranking_db) as conn:
        save_article(
            {"title": "Article 1", "link": "https://test.com/1", "source": "Source A", "published_at": "2026-09-09T10:00:00Z"},
            conn=conn,
        )
        save_article(
            {"title": "Article 2", "link": "https://test.com/2", "source": "Source B", "published_at": "2026-09-09T11:00:00Z"},
            conn=conn,
        )
        conn.commit()

    # 2. Save 1 successful analysis and 1 failure
    result = AnalysisResult(
        relevance_score=0.9,
        summary="A summary",
        key_points=[],
        topics=[],
        technologies=[],
        companies=[],
        people=[],
        models=[],
        event_type="model_release",
        novelty="breakthrough",
        confidence=0.95,
    )
    with get_connection(temp_ranking_db) as conn:
        save_analysis_success(article_id=1, result=result, provider="mock", model="mock-model", conn=conn)
        save_analysis_failure(article_id=2, error=Exception("Simulated error"), provider="mock", model="mock-model", conn=conn)

    # 3. Check article count before query
    with get_connection(temp_ranking_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_before = cursor.fetchone()["cnt"]

    # 4. Fetch completed articles using SQL JOIN
    with get_connection(temp_ranking_db) as conn:
        completed_rows = get_completed_articles_for_ranking(conn=conn)

    # 5. Check article count after query
    with get_connection(temp_ranking_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_after = cursor.fetchone()["cnt"]

    # Assertions
    assert count_before == count_after == 2
    assert len(completed_rows) == 1
    assert completed_rows[0]["article_id"] == 1
    assert completed_rows[0]["relevance_score"] == 0.9
    assert completed_rows[0]["status"] == "completed"

    # Ranking the row
    ranked = rank_articles(completed_rows)
    assert len(ranked) == 1
    assert ranked[0].article_id == 1
    assert ranked[0].final_score > 0.0
