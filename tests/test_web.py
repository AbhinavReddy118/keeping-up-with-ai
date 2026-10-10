"""
Offline integration tests for Task 3.1: Minimal Web Interface.
FastAPI + starlette TestClient. Zero network calls. Full database immutability.
"""

import pytest
from fastapi.testclient import TestClient

from database import init_db, get_connection
from storage import save_article
from llm.batch import save_analysis_success
from llm.models import AnalysisResult
from web.app import create_app


@pytest.fixture
def test_app_and_db(tmp_path, monkeypatch):
    """Create an isolated test database and configured FastAPI test client."""
    db_file = str(tmp_path / "test_web.db")
    init_db(db_file)

    # Point default database to test db
    monkeypatch.setattr("config.DATABASE_PATH", tmp_path / "test_web.db")
    monkeypatch.setattr("database.DATABASE_PATH", tmp_path / "test_web.db")


    # Seed test articles
    with get_connection(db_file) as conn:
        for i in range(1, 26):
            save_article(
                {
                    "title": f"AI News Article #{i}",
                    "link": f"https://example.com/news/{i}",
                    "source": "OpenAI News" if i % 2 == 0 else "Hacker News AI",
                    "published_at": f"2026-09-{i:02d}T10:00:00Z",
                    "summary": f"Summary for test article {i}",
                },
                conn=conn,
            )
        conn.commit()

        # Seed 3 completed analyses
        for aid in [1, 2, 3]:
            res = AnalysisResult(
                relevance_score=0.85 + (aid * 0.03),
                summary=f"Analyzed summary for {aid}",
                key_points=[f"Key takeaway {aid}"],
                topics=["Artificial Intelligence"],
                technologies=["Transformers"],
                companies=["OpenAI" if aid % 2 == 0 else "Anthropic"],
                people=[],
                models=[f"Model-V{aid}"],
                event_type="model_release" if aid == 1 else "industry_news",
                novelty="breakthrough" if aid == 1 else "incremental",
                confidence=0.95,
            )
            save_analysis_success(
                article_id=aid,
                result=res,
                provider="mock",
                model="mock-model",
                conn=conn,
            )

        # Seed pipeline states: articles 1-3 ANALYZED, articles 4-25 NEW
        for aid in range(1, 26):
            st = "ANALYZED" if aid in [1, 2, 3] else "NEW"
            conn.execute(
                "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, ?)",
                (aid, st),
            )
        conn.commit()

    app = create_app()
    return app, db_file


@pytest.fixture
def client(test_app_and_db):
    app, _ = test_app_and_db
    return TestClient(app)


def test_app_creation(test_app_and_db):
    """1. FastAPI application creates successfully."""
    app, _ = test_app_and_db
    assert app is not None
    assert getattr(app, "title", "") == "AI News Dashboard"


def test_index_route_status_200(client):
    """2. Index route returns HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "AI News Dashboard" in body
    assert "Stay informed. Smarter. Faster." in body


def test_summary_statistics_displayed(client):
    """3. Summary statistics match database counts."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    # Total seeded: 25, analyzed: 3, pending: 22, sources: 2
    assert "25" in body
    assert "Total Articles" in body
    assert "Analyzed &amp; Ranked" in body or "Analyzed & Ranked" in body
    assert "Pending Analysis" in body


def test_top_ranked_articles_appear(client):
    """4. Top ranked articles appear with scores."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Hot Topics" in body
    assert "AI News Article #3" in body or "AI News Article #1" in body


def test_latest_articles_appear_including_unanalyzed(client):
    """5 & 6. Latest articles appear and display unanalyzed status badges."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Latest AI News" in body
    assert "Not analyzed" in body
    assert "Analyzed" in body


def test_source_filtering(client):
    """7. Source filter limits returned articles."""
    response = client.get("/?source=OpenAI+News")
    assert response.status_code == 200
    body = response.text
    # Only OpenAI News articles
    assert "OpenAI News" in body


def test_event_type_filtering(client):
    """8. Event type filter limits ranked results."""
    response = client.get("/?event_type=model_release")
    assert response.status_code == 200
    assert "model_release" in response.text


def test_pagination(client):
    """9. Pagination navigates between pages."""
    # Page 1
    resp_p1 = client.get("/?page=1")
    assert resp_p1.status_code == 200
    assert "Page <strong>1</strong> of" in resp_p1.text

    # Page 2
    resp_p2 = client.get("/?page=2")
    assert resp_p2.status_code == 200
    assert "Page <strong>2</strong> of" in resp_p2.text


def test_article_detail_page(client):
    """10. Article detail page loads for analyzed article with scores."""
    response = client.get("/article/1")
    assert response.status_code == 200
    body = response.text
    assert "AI News Article #1" in body
    assert "About This Article" in body
    assert "Key Takeaways" in body
    assert "Importance:" in body


def test_article_detail_page_unanalyzed(client):
    """11. Article detail page loads gracefully for unanalyzed article without crashing."""
    response = client.get("/article/15")
    assert response.status_code == 200
    body = response.text
    assert "AI News Article #15" in body
    assert "Analysis Pending" in body
    assert "Not analyzed" in body


def test_invalid_article_id_returns_404(client):
    """12. Requesting a non-existent article returns HTTP 404."""
    response = client.get("/article/999999")
    assert response.status_code == 404
    assert "404" in response.text


def test_about_page(client):
    """13. About page loads with pipeline architecture."""
    response = client.get("/about")
    assert response.status_code == 200
    assert "Pipeline Architecture" in response.text


def test_empty_database(tmp_path, monkeypatch):
    """14. Empty database renders empty states cleanly without error."""
    empty_db = str(tmp_path / "empty.db")
    init_db(empty_db)
    monkeypatch.setattr("config.DATABASE_PATH", tmp_path / "empty.db")
    monkeypatch.setattr("database.DATABASE_PATH", tmp_path / "empty.db")


    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "AI News Dashboard" in response.text
    assert "0" in response.text


def test_read_only_database_immutability(test_app_and_db, client):
    """15. GET requests never modify database records."""
    _, db_file = test_app_and_db

    # Check counts before
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_before = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) as cnt FROM article_analysis")
        analysis_before = cursor.fetchone()["cnt"]

    # Execute multiple web requests
    client.get("/")
    client.get("/?source=OpenAI+News")
    client.get("/?page=2")
    client.get("/article/1")
    client.get("/article/10")
    client.get("/about")

    # Check counts after
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_after = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) as cnt FROM article_analysis")
        analysis_after = cursor.fetchone()["cnt"]

    assert articles_before == articles_after == 25
    assert analysis_before == analysis_after == 3


def test_dashboard_pending_count_reports_only_eligible_pipeline_stages(tmp_path):
    """Dashboard pending count reports only records in NEW, ENRICHED, ANALYSIS_PENDING."""
    from web.queries import get_summary_stats

    db_file = str(tmp_path / "test_pending_stages.db")
    init_db(db_file)

    with get_connection(db_file) as conn:
        # Create 8 articles
        for aid in range(1, 9):
            save_article(
                {
                    "title": f"Article {aid}",
                    "link": f"https://example.com/art/{aid}",
                    "source": "Tech AI",
                    "published_at": f"2026-10-0{aid}T10:00:00Z",
                    "summary": f"Summary {aid}",
                },
                conn=conn,
            )

        # 1: NEW (eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (1, 'NEW')")
        # 2: ENRICHED (eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (2, 'ENRICHED')")
        # 3: ANALYSIS_PENDING (eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (3, 'ANALYSIS_PENDING')")
        # 4: STORED (not eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (4, 'STORED')")
        # 5: ANALYZED (not eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (5, 'ANALYZED')")
        # 6: NOTIFIED (not eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (6, 'NOTIFIED')")
        # 7: FAILED (not eligible)
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (7, 'FAILED')")
        # 8: Unstaged / no record in article_pipeline_state (not eligible)

        # Mark article 5 as completed analysis
        res = AnalysisResult(
            relevance_score=0.9,
            summary="Analyzed summary",
            key_points=["Pt"],
            topics=["AI"],
            technologies=[],
            companies=[],
            people=[],
            models=[],
            event_type="industry_news",
            novelty="incremental",
            confidence=0.9,
        )
        save_analysis_success(5, res, "mock", "mock-model", conn=conn)
        conn.commit()

        stats = get_summary_stats(conn=conn)

    assert stats["total_articles"] == 8
    assert stats["analyzed_articles"] == 1
    # Operational queue consists strictly of NEW (1), ENRICHED (2), ANALYSIS_PENDING (3) -> 3
    assert stats["pending_analysis"] == 3
    assert stats["not_yet_analyzed"] == 3
