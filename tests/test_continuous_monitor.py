"""
Focused tests for the Continuous Monitor Pipeline (Step 18).

Tests:
1. Source polling schedule and tiers
2. New article persistence and queueing
3. Duplicate article handling
4. Pending processing (enrichment + LLM analysis)
5. Failed enrichment graceful fallback
6. Failed Qwen analysis isolation
7. Retry behavior for transient LLM errors
8. Notification deduplication across restarts
9. Restart and idempotency state
10. Source failure isolation
11. Status endpoint telemetry
12. Laya shadow mode non-blocking behavior
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import init_db, get_connection
from delivery.telegram import TelegramConfig
from llm.models import AnalysisRequest, AnalysisResult
from llm.provider import MockLLMProvider
from monitor import ContinuousMonitor
from web.app import app


@pytest.fixture(autouse=True)
def default_mock_env(monkeypatch):
    """Ensure tests run offline deterministically using mock LLM by default."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")


@pytest.fixture
def temp_db(tmp_path):
    """Create a clean isolated SQLite database for monitor testing."""
    db_file = tmp_path / "test_monitor.db"
    init_db(str(db_file))
    return str(db_file)


@pytest.fixture
def mock_provider():
    """Mock LLM provider returning realistic analysis results."""
    return MockLLMProvider(
        custom_response={
            "relevance_score": 0.95,
            "summary": "Mock headline summary of AI development.",
            "key_points": ["First breakthrough point", "Second practical point"],
            "topics": ["LLM", "Reasoning"],
            "technologies": ["Transformer"],
            "companies": ["OpenAI"],
            "people": ["Sam Altman"],
            "models": ["GPT-5"],
            "event_type": "model_release",
            "novelty": "breakthrough",
            "confidence": 0.98,
        }
    )


# --------------------------------------------------------------------------
# 1. Source Polling Schedule & Tiers
# --------------------------------------------------------------------------
def test_source_polling_schedule_and_tiers(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)
    sources = monitor.get_due_sources(force_all=True)

    assert len(sources) >= 7
    source_map = {s["source_name"]: s for s in sources}

    # Verify Tier 1
    assert source_map["OpenAI News"]["tier"] == "tier1"
    assert source_map["OpenAI News"]["poll_interval_seconds"] == 20 * 60

    # Verify Tier 2
    assert source_map["MIT Tech Review AI"]["tier"] == "tier2"
    assert source_map["MIT Tech Review AI"]["poll_interval_seconds"] == 40 * 60

    # Verify Tier 3
    assert source_map["arXiv CS.AI Papers"]["tier"] == "tier3"
    assert source_map["arXiv CS.AI Papers"]["poll_interval_seconds"] == 24 * 3600


# --------------------------------------------------------------------------
# 2. New Article Persistence & Queueing
# --------------------------------------------------------------------------
def test_new_article_persistence_and_queue(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)

    mock_entries = [
        {
            "guid": "guid-test-1",
            "title": "OpenAI Unveils New Model",
            "link": "https://openai.com/news/unveils-new-model",
            "summary": "Details about the new frontier model.",
            "published_at": "2026-10-05T12:00:00Z",
        }
    ]

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_entries, "etag-1", "mod-1", False)):
        src = monitor.get_due_sources(force_all=True)[0]
        inserted = monitor.poll_single_source(src)

    assert inserted == 1

    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, link FROM articles WHERE link = ?", (mock_entries[0]["link"],))
        art = cursor.fetchone()
        assert art is not None
        assert art["title"] == mock_entries[0]["title"]

        cursor.execute("SELECT stage, article_id FROM article_pipeline_state WHERE article_id = ?", (art["id"],))
        queue_row = cursor.fetchone()
        assert queue_row is not None
        assert queue_row["stage"] == "NEW"


# --------------------------------------------------------------------------
# 3. Duplicate Article Handling
# --------------------------------------------------------------------------
def test_duplicate_article_handling(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)

    mock_entries = [
        {
            "guid": "guid-dup-1",
            "title": "Duplicate Article Title",
            "link": "https://example.com/duplicate",
            "summary": "Summary of article.",
            "published_at": "2026-10-05T12:00:00Z",
        }
    ]

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_entries, "e1", "m1", False)):
        src = monitor.get_due_sources(force_all=True)[0]
        first_run = monitor.poll_single_source(src)
        assert first_run == 1

        # Second poll with identical entries
        second_run = monitor.poll_single_source(src)
        assert second_run == 0

    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles WHERE link = ?", (mock_entries[0]["link"],))
        assert cursor.fetchone()["cnt"] == 1


# --------------------------------------------------------------------------
# 4. Pending Processing (Enrichment + LLM Analysis)
# --------------------------------------------------------------------------
def test_pending_processing_enrichment_and_llm(temp_db, mock_provider):
    monitor = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    # Insert an article directly in NEW state
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, enrichment_status)
            VALUES ('Breakthrough In AI Reasoning', 'https://example.com/reasoning', 'OpenAI News', 'Brief summary.', 'pending')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')",
            (art_id,),
        )
        conn.commit()

    with patch("monitor.enrich_article", return_value={"enrichment_status": "enriched", "content": "Extracted text."}):
        completed = monitor.process_pending_articles()

    assert completed == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        assert c.fetchone()["stage"] == "ANALYZED"

        c.execute("SELECT status, relevance_score, event_type FROM article_analysis WHERE article_id = ?", (art_id,))
        analysis = c.fetchone()
        assert analysis is not None
        assert analysis["status"] == "completed"
        assert analysis["relevance_score"] == 0.95
        assert analysis["event_type"] == "model_release"


# --------------------------------------------------------------------------
# 5. Failed Enrichment Graceful Fallback
# --------------------------------------------------------------------------
def test_failed_enrichment_graceful_fallback(temp_db, mock_provider):
    monitor = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, enrichment_status)
            VALUES ('Article with Failed Enrichment', 'https://example.com/fail-enrich', 'MIT Tech Review AI', 'Valid RSS summary exists.', 'pending')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')",
            (art_id,),
        )
        conn.commit()

    # Simulate enrichment returning failed status
    failed_enrichment = {
        "enrichment_status": "failed",
        "enrichment_error": "HTTP 404 Not Found",
        "content": None,
    }
    with patch("monitor.enrich_article", return_value=failed_enrichment):
        completed = monitor.process_pending_articles()

    assert completed == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        assert c.fetchone()["stage"] == "ANALYZED"


# --------------------------------------------------------------------------
# 6. Failed Qwen Analysis Isolation
# --------------------------------------------------------------------------
def test_failed_qwen_analysis_isolation(temp_db):
    fail_provider = MockLLMProvider(simulate_error="unavailable")
    monitor = ContinuousMonitor(db_path=temp_db, provider=fail_provider, max_retries=2)

    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, enrichment_status)
            VALUES ('Article Failing Analysis', 'https://example.com/fail-analysis', 'OpenAI News', 'RSS summary.', 'enriched')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'ENRICHED')",
            (art_id,),
        )
        conn.commit()

    completed = monitor.process_pending_articles()
    assert completed == 0

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage, analysis_attempts, last_error FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        row = c.fetchone()
        assert row["stage"] == "ANALYSIS_PENDING"
        assert row["analysis_attempts"] >= 1
        assert "LLMProviderUnavailableError" in str(row["last_error"])


# --------------------------------------------------------------------------
# 7. Retry Behavior for Transient Errors
# --------------------------------------------------------------------------
def test_retry_behavior_transient_error(temp_db, mock_provider):
    monitor = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, enrichment_status)
            VALUES ('Retried Article', 'https://example.com/retry-art', 'OpenAI News', 'Summary.', 'enriched')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage, analysis_attempts) VALUES (?, 'ANALYSIS_PENDING', 1)",
            (art_id,),
        )
        conn.commit()

    # Now the provider succeeds
    completed = monitor.process_pending_articles()
    assert completed == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage, last_error FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        row = c.fetchone()
        assert row["stage"] == "ANALYZED"
        assert row["last_error"] is None


# --------------------------------------------------------------------------
# 8. Notification Deduplication Across Restarts
# --------------------------------------------------------------------------
def test_notification_deduplication(temp_db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock-chat-id")
    monkeypatch.setenv("TELEGRAM_MIN_SCORE", "0.5")

    monitor = ContinuousMonitor(db_path=temp_db)

    # Insert analyzed article
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, published_at, enrichment_status)
            VALUES ('High Impact Article', 'https://example.com/high-impact', 'OpenAI News', 'Great summary.', '2026-10-05T12:00:00Z', 'enriched')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO article_analysis (article_id, relevance_score, novelty, confidence, event_type, prompt_version, provider, model, status)
            VALUES (?, 0.95, 'breakthrough', 0.99, 'model_release', 'v1', 'mock', 'mock', 'completed')
            """,
            (art_id,),
        )
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'ANALYZED')",
            (art_id,),
        )
        conn.commit()

    with patch("monitor.send_telegram_message", return_value={"ok": True}) as mock_send:
        sent = monitor.send_notifications()
        assert sent == 1
        assert mock_send.call_count == 1

        # Second delivery pass: must be deduplicated
        sent_again = monitor.send_notifications()
        assert sent_again == 0
        assert mock_send.call_count == 1  # Not called again


# --------------------------------------------------------------------------
# 9. Restart and Idempotency State
# --------------------------------------------------------------------------
def test_restart_idempotency_state(temp_db, mock_provider, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock-chat-id")
    monkeypatch.setenv("TELEGRAM_MIN_SCORE", "0.1")

    # Run initial monitor instance
    monitor_1 = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    mock_entries = [
        {
            "guid": "guid-restart-1",
            "title": "Frontier AI Released",
            "link": "https://example.com/frontier-ai",
            "summary": "Frontier AI release summary.",
            "published_at": "2026-10-05T12:00:00Z",
        }
    ]

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_entries, "e1", "m1", False)):
        with patch("monitor.enrich_article", return_value={"enrichment_status": "enriched", "content": "Full content."}):
            with patch("monitor.send_telegram_message", return_value={"ok": True}) as mock_send:
                step_1 = monitor_1.step(force_poll=True)
                assert step_1["inserted"] == 1
                assert step_1["analyzed"] == 1
                assert step_1["notified"] == 1
                assert mock_send.call_count == 1

    # Simulate restart: create brand new ContinuousMonitor on same database
    monitor_2 = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_entries, "e1", "m1", False)):
        with patch("monitor.send_telegram_message", return_value={"ok": True}) as mock_send_2:
            step_2 = monitor_2.step(force_poll=True)
            # 0 new inserted, 0 analyzed, 0 notified
            assert step_2["inserted"] == 0
            assert step_2["analyzed"] == 0
            assert step_2["notified"] == 0
            assert mock_send_2.call_count == 0


# --------------------------------------------------------------------------
# 10. Source Failure Isolation
# --------------------------------------------------------------------------
def test_source_failure_isolation(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)
    sources = monitor.get_due_sources(force_all=True)

    failing_src = sources[0]
    healthy_src = sources[1]

    # Failing source raises network exception
    with patch("monitor.fetch_rss_feed_conditional", side_effect=RuntimeError("Connection timeout")):
        inserted_fail = monitor.poll_single_source(failing_src)
        assert inserted_fail == 0

    # Healthy source succeeds
    healthy_entries = [
        {
            "guid": "guid-healthy-1",
            "title": "Healthy Feed Announcement",
            "link": "https://healthy.org/news/1",
            "summary": "Healthy summary.",
            "published_at": "2026-10-05T12:00:00Z",
        }
    ]
    with patch("monitor.fetch_rss_feed_conditional", return_value=(healthy_entries, "e2", "m2", False)):
        inserted_ok = monitor.poll_single_source(healthy_src)
        assert inserted_ok == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT consecutive_failures, last_error FROM source_state WHERE source_name = ?", (failing_src["source_name"],))
        fail_row = c.fetchone()
        assert fail_row["consecutive_failures"] >= 1
        assert "Connection timeout" in fail_row["last_error"]

        c.execute("SELECT consecutive_failures, last_error FROM source_state WHERE source_name = ?", (healthy_src["source_name"],))
        ok_row = c.fetchone()
        assert ok_row["consecutive_failures"] == 0
        assert ok_row["last_error"] is None


# --------------------------------------------------------------------------
# 11. Status Endpoint Telemetry
# --------------------------------------------------------------------------
def test_status_endpoint_telemetry(temp_db):
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in {"ok", "degraded"}
    assert "monitor_running" in data
    assert "database_reachable" in data
    assert "pending_articles" in data
    assert "pending_analyses" in data
    assert "sources" in data
    assert "laya_enabled" in data


# --------------------------------------------------------------------------
# 12. Laya Shadow Mode Optional & Non-Blocking
# --------------------------------------------------------------------------
def test_laya_shadow_mode_non_blocking(temp_db, mock_provider, monkeypatch):
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monitor = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, summary, enrichment_status)
            VALUES ('Laya Shadow Article', 'https://example.com/laya-shadow', 'Hugging Face Blog', 'Deep learning post.', 'enriched')
            """
        )
        art_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')",
            (art_id,),
        )
        conn.commit()

    # Even if Laya throws or fails internally, pipeline proceeds to Qwen
    with patch("monitor.run_laya_shadow_triage", return_value={"status": "completed", "prediction": "MUST_KNOW"}):
        with patch("monitor.enrich_article", return_value={"enrichment_status": "enriched", "content": "Text"}):
            completed = monitor.process_pending_articles()

    assert completed == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage, laya_prediction FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        row = c.fetchone()
        assert row["stage"] == "ANALYZED"
        assert row["laya_prediction"] == "MUST_KNOW"

        # Qwen analysis also succeeded
        c.execute("SELECT status FROM article_analysis WHERE article_id = ?", (art_id,))
        assert c.fetchone()["status"] == "completed"
