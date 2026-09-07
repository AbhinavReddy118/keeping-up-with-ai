import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from ingestion.pipeline import run_pipeline, format_summary_report
from ingestion.health import evaluate_source_health, format_health_report


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database connection with initialized schema for pipeline testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            link TEXT UNIQUE NOT NULL,
            summary TEXT,
            source TEXT NOT NULL,
            published_at TEXT,
            raw_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    yield conn
    conn.close()


def test_multiple_enabled_sources_processed(db_conn):
    sources = [
        {"name": "Source Alpha", "url": "https://example.com/alpha.xml", "type": "rss", "enabled": True},
        {"name": "Source Beta", "url": "https://example.com/beta.xml", "type": "rss", "enabled": True},
    ]

    mock_entries_alpha = [
        {"title": "Alpha 1", "link": "https://example.com/a1", "summary": "Sum A1", "published_at": "2026-09-07T10:00:00Z"},
    ]
    mock_entries_beta = [
        {"title": "Beta 1", "link": "https://example.com/b1", "summary": "Sum B1", "published_at": "2026-09-07T11:00:00Z"},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed") as mock_fetch:
        mock_fetch.side_effect = [mock_entries_alpha, mock_entries_beta]

        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 2
        assert summary["healthy_sources"] == 2
        assert summary["failed_sources"] == 0
        assert summary["total_fetched"] == 2
        assert summary["total_inserted"] == 2
        assert summary["total_duplicates"] == 0


def test_disabled_sources_skipped(db_conn):
    sources = [
        {"name": "Active Source", "url": "https://example.com/active.xml", "type": "rss", "enabled": True},
        {"name": "Disabled Source", "url": "https://example.com/disabled.xml", "type": "rss", "enabled": False},
    ]

    mock_entries = [
        {"title": "Active 1", "link": "https://example.com/act1", "summary": "Sum", "published_at": "2026-09-07T10:00:00Z"},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed") as mock_fetch:
        mock_fetch.return_value = mock_entries

        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 1
        assert summary["sources"][0]["name"] == "Active Source"
        assert summary["sources"][0]["status"] == "HEALTHY"
        mock_fetch.assert_called_once_with("https://example.com/active.xml", timeout=10)


def test_empty_feed_detected_as_warning(db_conn):
    sources = [
        {"name": "Empty Source", "url": "https://example.com/empty.xml", "type": "rss", "enabled": True},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed", return_value=[]):
        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 1
        assert summary["warning_sources"] == 1
        assert summary["sources"][0]["status"] == "WARNING"
        assert any("empty feed" in issue.lower() for issue in summary["sources"][0]["issues"])


def test_source_failure_does_not_stop_remaining_sources(db_conn):
    sources = [
        {"name": "Failing Source", "url": "https://example.com/fail.xml", "type": "rss", "enabled": True},
        {"name": "Working Source", "url": "https://example.com/work.xml", "type": "rss", "enabled": True},
    ]

    mock_working_entries = [
        {"title": "Work 1", "link": "https://example.com/w1", "summary": "Working", "published_at": "2026-09-07T12:00:00Z"},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed") as mock_fetch:
        mock_fetch.side_effect = [RuntimeError("Connection refused by server"), mock_working_entries]

        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 2
        assert summary["healthy_sources"] == 1
        assert summary["failed_sources"] == 1
        assert summary["total_fetched"] == 1
        assert summary["total_inserted"] == 1

        fail_src = summary["sources"][0]
        assert fail_src["status"] == "FAILED"
        assert "Connection refused" in fail_src["error"]

        work_src = summary["sources"][1]
        assert work_src["status"] == "HEALTHY"
        assert work_src["inserted"] == 1


def test_feed_with_invalid_entries_marked_warning(db_conn):
    sources = [
        {"name": "Mixed Source", "url": "https://example.com/mixed.xml", "type": "rss", "enabled": True},
    ]

    mock_entries = [
        {"title": "", "link": "https://example.com/no-title"},  # Invalid
        {"title": "Valid Article", "link": "https://example.com/valid", "summary": "OK", "published_at": "2026-09-07T10:00:00Z"},
        {"title": "No Link", "link": ""},  # Invalid
    ]

    with patch("ingestion.pipeline.fetch_rss_feed", return_value=mock_entries):
        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_fetched"] == 3
        assert summary["total_normalized"] == 1
        assert summary["total_invalid"] == 2
        assert summary["sources"][0]["status"] == "WARNING"
        assert any("failed normalization" in issue for issue in summary["sources"][0]["issues"])


def test_feed_with_all_invalid_entries_marked_failed(db_conn):
    # If a feed returns items but NONE can be normalized, status should be FAILED
    health = evaluate_source_health(
        name="Broken Schema Feed",
        url="https://example.com/broken.xml",
        fetch_success=True,
        fetched=5,
        normalized=0,
        inserted=0,
        duplicates=0,
        invalid=5,
    )
    assert health["status"] == "FAILED"


def test_missing_publication_dates_marked_warning():
    health = evaluate_source_health(
        name="No Date Feed",
        url="https://example.com/nodate.xml",
        fetch_success=True,
        fetched=10,
        normalized=10,
        inserted=10,
        duplicates=0,
        invalid=0,
        missing_dates=10,
        newest_published_at=None,
    )
    assert health["status"] == "WARNING"
    assert any("missing valid publication dates" in i for i in health["issues"])


def test_suspiciously_old_content_marked_warning():
    ref_time = datetime(2026, 9, 7, tzinfo=timezone.utc)
    old_date = "2025-01-01T00:00:00Z"  # > 1.5 years old

    health = evaluate_source_health(
        name="Stale Feed",
        url="https://example.com/stale.xml",
        fetch_success=True,
        fetched=5,
        normalized=5,
        inserted=5,
        duplicates=0,
        invalid=0,
        newest_published_at=old_date,
        current_time=ref_time,
    )
    assert health["status"] == "WARNING"
    assert any("stale" in i for i in health["issues"])


def test_duplicate_heavy_feed_remains_healthy(db_conn):
    # Articles that are already in DB should result in duplicates without marking the feed FAILED
    sources = [
        {"name": "Repeat Source", "url": "https://example.com/repeat.xml", "type": "rss", "enabled": True},
    ]

    mock_entries = [
        {"title": "Article A", "link": "https://example.com/same-link", "published_at": "2026-09-07T10:00:00Z"},
        {"title": "Article A Duplicate", "link": "https://example.com/same-link", "published_at": "2026-09-07T10:00:00Z"},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed", return_value=mock_entries):
        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_fetched"] == 2
        assert summary["total_inserted"] == 1
        assert summary["total_duplicates"] == 1
        # Successfully fetched and normalized, so healthy!
        assert summary["sources"][0]["status"] == "HEALTHY"


def test_health_reporting_does_not_modify_stored_records(db_conn):
    # Insert an article directly
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO articles (title, link, summary, source, published_at, raw_content)
        VALUES ('Original Title', 'https://example.com/orig', 'Original Summary', 'Direct Insert', '2026-09-07T12:00:00Z', '{}')
    """)
    db_conn.commit()

    # Run pipeline with a failing source
    sources = [
        {"name": "Failing Source", "url": "https://example.com/fail.xml", "type": "rss", "enabled": True},
    ]
    with patch("ingestion.pipeline.fetch_rss_feed", side_effect=RuntimeError("Timeout")):
        run_pipeline(sources=sources, conn=db_conn)

    # Verify original record is unchanged
    cursor.execute("SELECT * FROM articles WHERE link = 'https://example.com/orig'")
    row = cursor.fetchone()
    assert row is not None
    assert row["title"] == "Original Title"
    assert row["summary"] == "Original Summary"


def test_format_health_report():
    sample_summary = {
        "total_sources": 2,
        "healthy_sources": 1,
        "warning_sources": 0,
        "failed_sources": 1,
        "total_fetched": 10,
        "total_normalized": 10,
        "total_inserted": 8,
        "total_duplicates": 2,
        "total_invalid": 0,
        "sources": [
            {
                "name": "Working Feed",
                "status": "HEALTHY",
                "fetched": 10,
                "normalized": 10,
                "inserted": 8,
                "duplicates": 2,
                "invalid": 0,
                "issues": [],
                "error": None,
            },
            {
                "name": "Failed Feed",
                "status": "FAILED",
                "fetched": 0,
                "normalized": 0,
                "inserted": 0,
                "duplicates": 0,
                "invalid": 0,
                "issues": ["Fetch failed: HTTP 404"],
                "error": "HTTP 404: Not Found",
            },
        ],
    }

    report = format_health_report(sample_summary)
    assert "INGESTION HEALTH REPORT" in report
    assert "Working Feed" in report
    assert "Failed Feed" in report
    assert "DIAGNOSTIC DETAILS FOR WARNING / FAILED SOURCES" in report
    assert "HTTP 404: Not Found" in report
