import pytest
import sqlite3
from database import init_db, get_connection
from storage import save_article, save_articles, get_article_by_link, get_all_articles


@pytest.fixture
def db_conn():
    """Create an isolated in-memory SQLite database connection with initialized schema for testing."""
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


def test_insert_valid_article(db_conn):
    article = {
        "title": "LLM Efficiency Research",
        "link": "https://example.com/llm-research",
        "summary": "Clean summary text",
        "source": "AI Journal",
        "published_at": "2026-09-07T12:00:00Z",
        "raw_content": '{"raw_entry": {}}',
    }

    inserted = save_article(article, conn=db_conn)
    assert inserted is True

    retrieved = get_article_by_link("https://example.com/llm-research", conn=db_conn)
    assert retrieved is not None
    assert retrieved["title"] == "LLM Efficiency Research"
    assert retrieved["link"] == "https://example.com/llm-research"
    assert retrieved["summary"] == "Clean summary text"
    assert retrieved["source"] == "AI Journal"
    assert retrieved["published_at"] == "2026-09-07T12:00:00Z"
    assert retrieved["raw_content"] == '{"raw_entry": {}}'


def test_duplicate_url_handling(db_conn):
    article = {
        "title": "Duplicate Link Article",
        "link": "https://example.com/duplicate-link",
        "summary": "Original summary",
        "source": "Hacker News AI",
        "published_at": "2026-09-07T10:00:00Z",
        "raw_content": '{"version": 1}',
    }

    # First insertion -> True
    first_insert = save_article(article, conn=db_conn)
    assert first_insert is True

    # Duplicate insertion with same link -> False (does not throw exception)
    duplicate_article = {
        "title": "Modified Title for Same Link",
        "link": "https://example.com/duplicate-link",
        "summary": "New summary",
        "source": "Hacker News AI",
        "published_at": "2026-09-07T10:00:00Z",
        "raw_content": '{"version": 2}',
    }

    second_insert = save_article(duplicate_article, conn=db_conn)
    assert second_insert is False

    # Retrieve record to verify original title remains unchanged
    record = get_article_by_link("https://example.com/duplicate-link", conn=db_conn)
    assert record["title"] == "Duplicate Link Article"


def test_insert_optional_fields_null(db_conn):
    article_null_fields = {
        "title": "Headline with Null Optional Fields",
        "link": "https://example.com/null-fields",
        "summary": None,
        "source": "arXiv CS.AI",
        "published_at": None,
        "raw_content": "{}",
    }

    inserted = save_article(article_null_fields, conn=db_conn)
    assert inserted is True

    record = get_article_by_link("https://example.com/null-fields", conn=db_conn)
    assert record is not None
    assert record["summary"] is None
    assert record["published_at"] is None


def test_multiple_articles_batch_storage(db_conn):
    articles = [
        {
            "title": "Article One",
            "link": "https://example.com/article-1",
            "summary": "Summary 1",
            "source": "Source A",
            "published_at": "2026-09-07T08:00:00Z",
            "raw_content": "{}",
        },
        {
            "title": "Article Two",
            "link": "https://example.com/article-2",
            "summary": "Summary 2",
            "source": "Source B",
            "published_at": "2026-09-07T09:00:00Z",
            "raw_content": "{}",
        },
        {
            "title": "Duplicate Article One",
            "link": "https://example.com/article-1",  # Same link as article 1
            "summary": "Summary 1 Duplicate",
            "source": "Source A",
            "published_at": "2026-09-07T08:00:00Z",
            "raw_content": "{}",
        },
    ]

    result = save_articles(articles, conn=db_conn)
    assert result["total"] == 3
    assert result["inserted"] == 2
    assert result["duplicates"] == 1

    stored = get_all_articles(conn=db_conn)
    assert len(stored) == 2


def test_save_article_invalid_schema(db_conn):
    invalid_article = {
        "summary": "Missing title, link, and source",
    }
    with pytest.raises(ValueError, match="Article dictionary missing required fields"):
        save_article(invalid_article, conn=db_conn)
