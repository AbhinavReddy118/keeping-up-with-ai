import sqlite3
import pytest
from unittest.mock import patch
from ingestion.pipeline import run_pipeline, format_summary_report


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
        assert summary["successful_sources"] == 2
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

        # Disabled source should be filtered out before processing
        assert summary["total_sources"] == 1
        assert summary["sources"][0]["name"] == "Active Source"
        mock_fetch.assert_called_once_with("https://example.com/active.xml", timeout=10)


def test_zero_entries_handled_correctly(db_conn):
    sources = [
        {"name": "Empty Source", "url": "https://example.com/empty.xml", "type": "rss", "enabled": True},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed", return_value=[]):
        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 1
        assert summary["successful_sources"] == 1
        assert summary["total_fetched"] == 0
        assert summary["total_inserted"] == 0
        assert summary["sources"][0]["status"] == "success"


def test_source_failure_does_not_stop_remaining_sources(db_conn):
    sources = [
        {"name": "Failing Source", "url": "https://example.com/fail.xml", "type": "rss", "enabled": True},
        {"name": "Working Source", "url": "https://example.com/work.xml", "type": "rss", "enabled": True},
    ]

    mock_working_entries = [
        {"title": "Work 1", "link": "https://example.com/w1", "summary": "Working", "published_at": "2026-09-07T12:00:00Z"},
    ]

    with patch("ingestion.pipeline.fetch_rss_feed") as mock_fetch:
        # First source raises RuntimeError, second succeeds
        mock_fetch.side_effect = [RuntimeError("Connection refused by server"), mock_working_entries]

        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_sources"] == 2
        assert summary["successful_sources"] == 1
        assert summary["failed_sources"] == 1
        assert summary["total_fetched"] == 1
        assert summary["total_inserted"] == 1

        # Check failing source summary
        fail_src = summary["sources"][0]
        assert fail_src["name"] == "Failing Source"
        assert fail_src["status"] == "failed"
        assert "Connection refused" in fail_src["error"]

        # Check working source summary
        work_src = summary["sources"][1]
        assert work_src["name"] == "Working Source"
        assert work_src["status"] == "success"
        assert work_src["inserted"] == 1


def test_invalid_article_does_not_crash_pipeline(db_conn):
    sources = [
        {"name": "Mixed Source", "url": "https://example.com/mixed.xml", "type": "rss", "enabled": True},
    ]

    mock_entries = [
        {"title": "", "link": "https://example.com/no-title"},  # Invalid: missing title
        {"title": "Valid Article", "link": "https://example.com/valid", "summary": "OK", "published_at": "2026-09-07T10:00:00Z"},
        {"title": "No Link", "link": ""},  # Invalid: missing link
    ]

    with patch("ingestion.pipeline.fetch_rss_feed", return_value=mock_entries):
        summary = run_pipeline(sources=sources, conn=db_conn)

        assert summary["total_fetched"] == 3
        assert summary["total_normalized"] == 1
        assert summary["total_invalid"] == 2
        assert summary["total_inserted"] == 1
        assert summary["sources"][0]["invalid"] == 2


def test_duplicate_article_counts_in_summary(db_conn):
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


def test_format_summary_report():
    sample_summary = {
        "total_sources": 2,
        "successful_sources": 1,
        "failed_sources": 1,
        "total_fetched": 10,
        "total_normalized": 9,
        "total_inserted": 8,
        "total_duplicates": 1,
        "total_invalid": 1,
        "sources": [
            {
                "name": "Working Feed",
                "status": "success",
                "fetched": 10,
                "normalized": 9,
                "inserted": 8,
                "duplicates": 1,
                "invalid": 1,
                "error": None,
            },
            {
                "name": "Failed Feed",
                "status": "failed",
                "fetched": 0,
                "normalized": 0,
                "inserted": 0,
                "duplicates": 0,
                "invalid": 0,
                "error": "HTTP 404: Not Found",
            },
        ],
    }

    report = format_summary_report(sample_summary)
    assert "INGESTION PIPELINE EXECUTION SUMMARY" in report
    assert "Working Feed" in report
    assert "Failed Feed" in report
    assert "HTTP 404: Not Found" in report
