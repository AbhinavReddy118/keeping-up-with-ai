"""
Unit tests for Article Content Enrichment Layer (Task 1.2).
Covers content sufficiency, HTML parsing, HTTP fetch handling, error resilience,
database persistence, idempotency, and batch enrichment.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import httpx
import pytest

from database import init_db
from ingestion.enricher import (
    DEFAULT_USER_AGENT,
    DEFAULT_TIMEOUT,
    enrich_article,
    enrich_articles_batch,
    extract_article_text_from_html,
    fetch_webpage_content,
    is_usable_content,
)
from llm.analyzer import prepare_article_input
from storage import get_article_by_id, save_article


@pytest.fixture
def test_db():
    """Create an isolated in-memory database with initialized schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db_schema(conn)
    yield conn
    conn.close()


def init_db_schema(conn: sqlite3.Connection):
    """Initialize schema for in-memory connection."""
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
            content TEXT,
            enrichment_status TEXT DEFAULT 'pending',
            enrichment_error TEXT,
            enriched_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


# ==============================================================================
# 1. Content Sufficiency Tests
# ==============================================================================

def test_is_usable_content_empty_and_none():
    assert is_usable_content(None) is False
    assert is_usable_content("") is False
    assert is_usable_content("   \n\t  ") is False
    assert is_usable_content(12345) is False


def test_is_usable_content_rss_metadata_json():
    # RSS raw_content JSON alone is insufficient
    raw_rss_json = json.dumps({
        "raw_entry": {
            "title": "Sample Title",
            "link": "https://example.com/test",
            "summary": "",
            "published_at": "2026-09-01T00:00:00Z",
        },
        "source_info": {"name": "Test Source", "url": "https://example.com/feed.xml"},
    })
    assert is_usable_content(raw_rss_json) is False

    # Arbitrary dict JSON string alone is insufficient
    dict_json = '{"key": "value", "description": "some metadata"}'
    assert is_usable_content(dict_json) is False


def test_is_usable_content_very_short_strings():
    assert is_usable_content("Short headline") is False
    assert is_usable_content("Condé Nast") is False
    assert is_usable_content("Introducing canvas") is False
    assert is_usable_content("Read more at the original article link here.") is False


def test_is_usable_content_valid_substantive_text():
    good_text = (
        "Researchers at Google DeepMind and OpenAI have published a landmark study "
        "demonstrating new mathematical reasoning capabilities in large language models. "
        "The model achieves unprecedented accuracy on formal proof benchmarks through "
        "test-time reinforcement learning and tree search verification."
    )
    assert is_usable_content(good_text) is True


# ==============================================================================
# 2. HTML Text Extraction Tests
# ==============================================================================

def test_extract_main_content_over_navigation_and_boilerplate():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title><style>.hidden { display: none; }</style></head>
    <body>
        <header>
            <nav>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/news">News</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
        </header>
        <div class="cookie-banner">This site uses cookies. Accept or decline.</div>
        <div class="sidebar">Related posts and advertisements go here.</div>
        <main>
            <article class="prose blog-content">
                <h1>Breakthrough in Autonomous AI Agents</h1>
                <p>
                    Today we are announcing a major leap forward in autonomous AI systems.
                    Our new framework enables agents to plan, execute, and verify code tasks
                    with significantly higher reliability across multi-file repositories.
                </p>
                <p>
                    The benchmark results show a 40% improvement in first-pass correctness
                    compared to standard zero-shot code generation methods.
                </p>
                <div class="not-prose comments-widget">
                    <h3>Comments</h3>
                    <p>User123: Great article!</p>
                </div>
            </article>
        </main>
        <footer>
            <p>&copy; 2026 AI News Digest Inc. All rights reserved.</p>
        </footer>
    </body>
    </html>
    """
    extracted = extract_article_text_from_html(html)
    assert extracted is not None
    # Main content present
    assert "Breakthrough in Autonomous AI Agents" in extracted
    assert "Today we are announcing a major leap forward" in extracted
    assert "benchmark results show a 40% improvement" in extracted

    # Boilerplate / navigation / ads / comments removed
    assert "Home" not in extracted
    assert "Contact" not in extracted
    assert "This site uses cookies" not in extracted
    assert "Related posts" not in extracted
    assert "All rights reserved" not in extracted
    assert "User123" not in extracted


def test_extract_malformed_and_empty_html():
    assert extract_article_text_from_html("") is None
    assert extract_article_text_from_html("   ") is None
    assert extract_article_text_from_html("<html><body></body></html>") is None
    assert extract_article_text_from_html("<p>Hi</p>") is None  # Too short to be usable


# ==============================================================================
# 3. HTTP Fetch & Enrichment Pipeline Tests
# ==============================================================================

def test_missing_rss_summary_triggers_successful_extraction(test_db):
    """Article with missing RSS summary triggers HTTP fetch and stores extracted content."""
    article = {
        "title": "Hugging Face Model Training",
        "link": "https://huggingface.co/blog/how-to-train-sample",
        "summary": None,
        "source": "Hugging Face Blog",
        "published_at": "2026-09-01T00:00:00Z",
        "raw_content": json.dumps({"raw_entry": {"title": "Test", "summary": ""}}),
    }
    save_article(article, conn=test_db)
    row = get_article_by_id(1, conn=test_db)
    assert row["summary"] is None

    mock_html = """
    <html>
    <body>
        <main>
            <div class="blog-content prose">
                <h1>How to Train a New Language Model from Scratch</h1>
                <p>
                    Over the past few months, we made several improvements to our transformers
                    and tokenizers libraries, with the goal of making it easier than ever to
                    train a new language model from scratch.
                </p>
                <p>
                    In this post we will demonstrate how to train a small model on Esperanto,
                    including dataset preprocessing, tokenizer training, and model evaluation.
                </p>
            </div>
        </main>
    </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_html

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        enriched = enrich_article(1, conn=test_db)
        assert mock_get.called
        # Verify custom User-Agent was passed
        args, kwargs = mock_get.call_args
        assert args[0] == "https://huggingface.co/blog/how-to-train-sample"

    assert enriched["enrichment_status"] == "enriched"
    assert enriched["content"] is not None
    assert "How to Train a New Language Model" in enriched["content"]
    assert "Esperanto" in enriched["content"]

    # Verify original raw_content and summary were NOT destroyed
    assert enriched["summary"] is None
    assert "raw_entry" in enriched["raw_content"]

    # Verify database persistence
    db_row = get_article_by_id(1, conn=test_db)
    assert db_row["enrichment_status"] == "enriched"
    assert db_row["content"] == enriched["content"]


def test_usable_rss_content_skips_fetch(test_db):
    """Article with good existing summary does not trigger unnecessary HTTP fetch."""
    good_summary = (
        "OpenAI has introduced a new multimodal reasoning architecture that achieves "
        "state-of-the-art results on competitive programming and mathematical benchmarks, "
        "utilizing dynamic reasoning chains."
    )
    article = {
        "title": "OpenAI Reasoning Architecture",
        "link": "https://example.com/openai-reasoning",
        "summary": good_summary,
        "source": "OpenAI News",
        "published_at": "2026-09-02T10:00:00Z",
        "raw_content": '{"summary": "valid"}',
    }
    save_article(article, conn=test_db)

    with patch("httpx.Client.get") as mock_get:
        enriched = enrich_article(1, conn=test_db)
        assert not mock_get.called

    assert enriched["enrichment_status"] == "not_needed"
    assert enriched["content"] is None
    assert enriched["summary"] == good_summary


def test_http_failure_handled_gracefully(test_db):
    """HTTP 404 or 500 error does not crash the system, sets enrichment_status='failed'."""
    article = {
        "title": "Missing Article Page",
        "link": "https://example.com/not-found",
        "summary": None,
        "source": "Tech Blog",
        "published_at": "2026-09-03T10:00:00Z",
        "raw_content": "{}",
    }
    save_article(article, conn=test_db)

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.reason_phrase = "Not Found"

    with patch("httpx.Client.get", return_value=mock_resp):
        enriched = enrich_article(1, conn=test_db)

    assert enriched["enrichment_status"] == "failed"
    assert enriched["content"] is None
    assert "404" in enriched["enrichment_error"]

    # Verify database state
    db_row = get_article_by_id(1, conn=test_db)
    assert db_row["enrichment_status"] == "failed"
    assert db_row["content"] is None


def test_http_timeout_handled_gracefully(test_db):
    """Network timeout does not crash the system, sets enrichment_status='failed'."""
    article = {
        "title": "Slow Responding Server",
        "link": "https://example.com/slow-page",
        "summary": None,
        "source": "Slow Source",
        "published_at": "2026-09-04T10:00:00Z",
        "raw_content": "{}",
    }
    save_article(article, conn=test_db)

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Read timed out")):
        enriched = enrich_article(1, conn=test_db)

    assert enriched["enrichment_status"] == "failed"
    assert enriched["content"] is None
    assert "Timeout" in enriched["enrichment_error"]


def test_repeated_enrichment_idempotency(test_db):
    """Running enrichment multiple times does not make duplicate requests or overwrite content."""
    article = {
        "title": "Idempotent Article Test",
        "link": "https://example.com/idempotency",
        "summary": None,
        "source": "AI Daily",
        "published_at": "2026-09-05T10:00:00Z",
        "raw_content": "{}",
    }
    save_article(article, conn=test_db)

    mock_html = """
    <html><body><article>
        <h1>Deterministic Processing in Production Pipelines</h1>
        <p>
            Building idempotent data pipelines is essential for reliable background processing.
            When an article is already enriched, repeated runs must be safe no-ops without
            redundant network traffic or database state churn.
        </p>
    </article></body></html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_html

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        # First call -> fetches and enriches
        first_run = enrich_article(1, conn=test_db)
        assert mock_get.call_count == 1
        assert first_run["enrichment_status"] == "enriched"
        first_content = first_run["content"]

        # Second call -> skips network request
        second_run = enrich_article(1, conn=test_db)
        assert mock_get.call_count == 1  # Unchanged!
        assert second_run["enrichment_status"] == "enriched"
        assert second_run["content"] == first_content


def test_prepare_article_input_prefers_enriched_content():
    """prepare_article_input uses enriched content if present, falling back to summary."""
    article_enriched = {
        "id": 100,
        "title": "Transformers and Tokenizers Guide",
        "link": "https://example.com/hf",
        "source": "Hugging Face",
        "summary": None,
        "content": (
            "Extracted full article text detailing step-by-step training of a language model "
            "using the Hugging Face Transformers and Tokenizers libraries from scratch."
        ),
    }
    req = prepare_article_input(article_enriched)
    assert req.summary is not None
    assert "Extracted full article text" in req.summary

    # When content is None, uses summary
    article_standard = {
        "id": 101,
        "title": "Standard Article",
        "link": "https://example.com/std",
        "source": "OpenAI",
        "summary": "Standard RSS feed summary text.",
        "content": None,
    }
    req2 = prepare_article_input(article_standard)
    assert req2.summary == "Standard RSS feed summary text."


def test_enrich_articles_batch(test_db):
    """enrich_articles_batch accurately tallies enriched, skipped, and failed articles."""
    articles = [
        {
            "title": "Article Needs Enrichment",
            "link": "https://example.com/needs-enrichment",
            "summary": None,
            "source": "Blog A",
        },
        {
            "title": "Article With Usable Summary",
            "link": "https://example.com/has-summary",
            "summary": (
                "This is a completely sufficient and substantive RSS feed summary with well over "
                "one hundred characters describing a significant new AI research milestone."
            ),
            "source": "Blog B",
        },
        {
            "title": "Article That 404s",
            "link": "https://example.com/will-404",
            "summary": None,
            "source": "Blog C",
        },
    ]
    for a in articles:
        save_article(a, conn=test_db)

    def mock_get_side_effect(url, **kwargs):
        if "needs-enrichment" in url:
            resp = MagicMock()
            resp.status_code = 200
            resp.text = (
                "<article>"
                "<h1>Enriched Headline on Frontier AI Models</h1>"
                "<p>"
                "Full content extracted from the webpage successfully detailing new performance "
                "benchmarks, training methodology, and deployment guidelines for production systems."
                "</p>"
                "</article>"
            )
            return resp
        elif "will-404" in url:
            resp = MagicMock()
            resp.status_code = 404
            resp.reason_phrase = "Not Found"
            return resp
        raise RuntimeError(f"Unexpected URL: {url}")

    with patch("httpx.Client.get", side_effect=mock_get_side_effect):
        summary = enrich_articles_batch([1, 2, 3], conn=test_db)

    assert summary["total"] == 3
    assert summary["enriched"] == 1
    assert summary["skipped"] == 1
    assert summary["failed"] == 1
