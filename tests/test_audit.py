import sqlite3
import pytest
from audit import audit_database, normalize_title, format_audit_report


@pytest.fixture
def mock_db():
    """Create an isolated in-memory SQLite database populated with representative audit test cases."""
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

    sample_articles = [
        # Normal article 1 from Source Alpha
        ("Breakthrough in LLMs", "https://alpha.com/a1", "A full detailed summary of the breakthrough.", "Source Alpha", "2026-09-01T10:00:00Z", "{}"),
        # Normal article 2 from Source Alpha
        ("Quantum Machine Learning", "https://alpha.com/a2", "Summary of quantum ML.", "Source Alpha", "2026-09-02T12:00:00Z", "{}"),
        # Article from Source Beta sharing the same normalized title (cross-source overlap!)
        ("Breakthrough in LLMs!", "https://beta.com/b1", "Beta summary.", "Source Beta", "2026-09-03T14:00:00Z", "{}"),
        # Article from Source Beta with same title as another in Beta (same-source title duplicate)
        ("Quantum Machine Learning", "https://beta.com/b2", "Beta duplicate title.", "Source Beta", "2026-09-04T16:00:00Z", "{}"),
        # Article with missing published_at
        ("No Date Article", "https://gamma.com/g1", "Summary without date.", "Source Gamma", None, "{}"),
        # Article with empty summary
        ("Empty Summary Article", "https://gamma.com/g2", "", "Source Gamma", "2026-09-05T08:00:00Z", "{}"),
        # Article with short summary (<20 chars)
        ("Short Summary Article", "https://gamma.com/g3", "Too short", "Source Gamma", "2026-09-06T09:00:00Z", "{}"),
    ]

    cursor.executemany("""
        INSERT INTO articles (title, link, summary, source, published_at, raw_content)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_articles)
    conn.commit()

    yield conn
    conn.close()


def test_normalize_title():
    assert normalize_title("Hello, World!") == "hello world"
    assert normalize_title("  AI  Breakthrough...  ") == "ai breakthrough"
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_audit_volume_and_sources(mock_db):
    results = audit_database(conn=mock_db)

    assert results["total_articles"] == 7
    assert results["unique_urls"] == 7
    assert results["duplicate_urls"] == 0

    sources = {s["source"]: s["count"] for s in results["source_breakdown"]}
    assert sources["Source Alpha"] == 2
    assert sources["Source Beta"] == 2
    assert sources["Source Gamma"] == 3


def test_audit_date_coverage(mock_db):
    results = audit_database(conn=mock_db)
    dc = results["date_coverage"]

    assert dc["earliest"] == "2026-09-01T10:00:00Z"
    assert dc["latest"] == "2026-09-06T09:00:00Z"
    assert dc["missing_dates"] == 1


def test_audit_duplicate_and_overlap_detection(mock_db):
    results = audit_database(conn=mock_db)

    # Cross-source overlaps: "Breakthrough in LLMs" and "Quantum Machine Learning" both exist in Alpha and Beta
    overlaps = results["cross_source_overlap"]
    assert len(overlaps) == 2
    overlap_titles = [o["normalized_title"] for o in overlaps]
    assert "breakthrough in llms" in overlap_titles
    assert "quantum machine learning" in overlap_titles

    # Normalized title duplicates: "breakthrough in llms" and "quantum machine learning"
    norm_dups = results["duplicate_analysis"]["normalized_title_duplicates"]
    assert len(norm_dups) == 2


def test_audit_data_quality_issues(mock_db):
    results = audit_database(conn=mock_db)
    dq = results["data_quality"]

    assert dq["empty_titles"] == 0
    assert dq["invalid_links"] == 0
    assert dq["missing_sources"] == 0
    assert dq["missing_dates"] == 1
    assert dq["empty_summaries"] == 1
    assert dq["short_summaries"] == 2  # "Beta summary." (13 ch) and "Too short" (9 ch)


def test_format_audit_report(mock_db):
    results = audit_database(conn=mock_db)
    report = format_audit_report(results)

    assert "DATABASE VOLUME & URL INTEGRITY" in report
    assert "Total Articles Stored : 7" in report
    assert "Cross-Source Overlaps Detected" in report
    assert "Breakthrough in LLMs" in report
