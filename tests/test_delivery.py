"""Tests for the Telegram delivery module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from delivery import (
    TelegramConfig,
    TelegramDeliveryError,
    deliver_ranked_articles,
    format_telegram_update,
    send_notification_for_article,
    send_telegram_message,
)
from delivery.telegram import is_article_delivered, record_article_delivery


@pytest.fixture
def sample_article():
    return {
        "article_id": 42,
        "title": "OpenAI Releases GPT-5",
        "source": "OpenAI News",
        "published_at": "2026-09-07T10:00:00Z",
        "link": "https://openai.com/news/gpt-5",
        "final_score": 0.92,
        "event_type": "model_release",
    }


def test_telegram_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    config = TelegramConfig.from_env()
    assert config.enabled is False
    assert config.bot_token == ""
    assert config.chat_id == ""
    assert config.min_score == 0.0
    assert config.limit == 5


def test_telegram_config_from_env_enabled(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("TELEGRAM_MIN_SCORE", "0.5")
    monkeypatch.setenv("TELEGRAM_LIMIT", "3")
    config = TelegramConfig.from_env()
    assert config.enabled is True
    assert config.bot_token == "test-token"
    assert config.chat_id == "123456"
    assert config.min_score == 0.5
    assert config.limit == 3


def test_format_telegram_update(sample_article):
    msg = format_telegram_update(sample_article, position=1, total=3)
    assert "1/3" in msg
    assert "OpenAI Releases GPT-5" in msg
    assert "OpenAI News" in msg
    assert "0.920" in msg
    assert "https://openai.com/news/gpt-5" in msg


def test_format_telegram_update_truncates_long_text():
    article = {
        "title": "A" * 5000,
        "source": "Test",
        "published_at": "2026-09-07",
        "link": "",
        "final_score": 0.5,
    }
    msg = format_telegram_update(article, position=1, total=1)
    assert len(msg) <= 3900


def test_send_telegram_message_missing_credentials():
    with pytest.raises(TelegramDeliveryError, match="required"):
        send_telegram_message("", "123", "test")
    with pytest.raises(TelegramDeliveryError, match="required"):
        send_telegram_message("token", "", "test")


def test_send_telegram_message_success():
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"ok": True, "result": {"message_id": 1}}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("delivery.telegram.urllib.request.urlopen", return_value=mock_response):
        result = send_telegram_message("test-token", "123456", "Hello")
    assert result["ok"] is True


def test_send_telegram_message_api_error():
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"ok": False, "description": "Bad chat"}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("delivery.telegram.urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(TelegramDeliveryError, match="Bad chat"):
            send_telegram_message("test-token", "123456", "Hello")


def test_send_telegram_message_http_error():
    from email.message import Message
    import urllib.error

    http_error = urllib.error.HTTPError(
        url="https://api.telegram.org/bot/sendMessage",
        code=403,
        msg="Forbidden",
        hdrs=Message(),
        fp=None,
    )
    with patch("delivery.telegram.urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TelegramDeliveryError, match="HTTP 403"):
            send_telegram_message("test-token", "123456", "Hello")


def test_deliver_ranked_articles_disabled():
    config = TelegramConfig(bot_token="t", chat_id="c", enabled=False)
    articles = [{"final_score": 0.9, "title": "Test"}]
    result = deliver_ranked_articles(articles, config)
    assert result["enabled"] is False
    assert result["sent"] == 0


def test_deliver_ranked_articles_missing_credentials():
    config = TelegramConfig(bot_token="", chat_id="", enabled=True)
    articles = [{"final_score": 0.9, "title": "Test"}]
    result = deliver_ranked_articles(articles, config)
    assert result["enabled"] is False
    assert result["sent"] == 0


def test_deliver_ranked_articles_filters_by_min_score():
    config = TelegramConfig(
        bot_token="test-token",
        chat_id="123456",
        min_score=0.8,
        limit=5,
        enabled=True,
    )
    articles = [
        {"article_id": 1, "title": "High", "final_score": 0.9, "source": "S", "link": "", "published_at": ""},
        {"article_id": 2, "title": "Low", "final_score": 0.5, "source": "S", "link": "", "published_at": ""},
        {"article_id": 3, "title": "High2", "final_score": 0.85, "source": "S", "link": "", "published_at": ""},
    ]

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("delivery.telegram.urllib.request.urlopen", return_value=mock_response):
        result = deliver_ranked_articles(articles, config)
    assert result["enabled"] is True
    assert result["sent"] == 2
    assert result["skipped"] == 1


def test_deliver_ranked_articles_respects_limit():
    config = TelegramConfig(
        bot_token="test-token",
        chat_id="123456",
        limit=2,
        enabled=True,
    )
    articles = [
        {"article_id": i, "title": f"T{i}", "final_score": 1.0 - i*0.1, "source": "S", "link": "", "published_at": ""}
        for i in range(5)
    ]

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("delivery.telegram.urllib.request.urlopen", return_value=mock_response):
        result = deliver_ranked_articles(articles, config)
    assert result["sent"] == 2
    assert result["skipped"] == 3


def test_deliver_ranked_articles_continues_on_failure():
    config = TelegramConfig(
        bot_token="test-token",
        chat_id="123456",
        limit=3,
        enabled=True,
    )
    articles = [
        {"article_id": 1, "title": "A1", "final_score": 0.9, "source": "S", "link": "", "published_at": ""},
        {"article_id": 2, "title": "A2", "final_score": 0.8, "source": "S", "link": "", "published_at": ""},
        {"article_id": 3, "title": "A3", "final_score": 0.7, "source": "S", "link": "", "published_at": ""},
    ]

    success_response = MagicMock()
    success_response.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    success_response.__enter__.return_value = success_response

    error_response = MagicMock()
    error_response.read.return_value = json.dumps({"ok": False, "description": "Rate limited"}).encode("utf-8")
    error_response.__enter__.return_value = error_response

    responses = [success_response, error_response, success_response]

    def side_effect(*args, **kwargs):
        return responses.pop(0)

    with patch("delivery.telegram.urllib.request.urlopen", side_effect=side_effect):
        result = deliver_ranked_articles(articles, config)
    assert result["sent"] == 2
    assert result["failed"] == 1


# --------------------------------------------------------------------------
# Targeted Delivery (send_notification_for_article) Test Suite
# --------------------------------------------------------------------------

@pytest.fixture
def delivery_test_db(tmp_path):
    """Isolated SQLite database for targeted delivery tests."""
    from database import init_db
    db_file = tmp_path / "delivery_test.db"
    init_db(str(db_file))
    return str(db_file)


def _seed_article(
    db_path: str,
    title: str = "Test Breakthrough",
    link: str = "https://example.com/art1",
    source: str = "Test Source",
    stage: str = "ANALYZED",
    analysis_status: str | None = "completed",
    relevance_score: float = 0.95,
    novelty: str = "breakthrough",
    confidence: float = 0.99,
    event_type: str = "model_release",
) -> int:
    from database import get_connection
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO articles (title, link, source, published_at, content)
            VALUES (?, ?, ?, '2026-10-12T10:00:00Z', 'Test content')
            """,
            (title, link, source),
        )
        last_id = cursor.lastrowid
        assert last_id is not None
        art_id = int(last_id)
        cursor.execute(
            """
            INSERT INTO article_pipeline_state (article_id, stage)
            VALUES (?, ?)
            """,
            (art_id, stage),
        )
        if analysis_status is not None:
            cursor.execute(
                """
                INSERT INTO article_analysis (
                    article_id, relevance_score, novelty, confidence,
                    event_type, prompt_version, provider, model, status
                ) VALUES (?, ?, ?, ?, ?, '1.0', 'mock', 'mock', ?)
                """,
                (art_id, relevance_score, novelty, confidence, event_type, analysis_status),
            )
        conn.commit()
        return art_id


def test_targeted_delivery_valid_analyzed_article(delivery_test_db):
    """Requirement 1: Valid ANALYZED article reaches the delivery function and sends."""
    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed", relevance_score=0.95)
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True, min_score=0.0)

    mock_resp = {"ok": True, "result": {"message_id": 999}}
    with patch("delivery.telegram.send_telegram_message", return_value=mock_resp) as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is True
    assert result["status"] == "sent"
    assert result["delivered"] is True
    assert result["article_id"] == art_id
    assert mock_send.call_count == 1

    # Verify message content contains title and score
    call_args = mock_send.call_args[0]
    assert "Test Breakthrough" in call_args[2]

    # Verify pipeline state updated to NOTIFIED
    from database import get_connection
    with get_connection(delivery_test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        assert cursor.fetchone()["stage"] == "NOTIFIED"
        cursor.execute("SELECT status, score FROM article_deliveries WHERE article_id = ?", (art_id,))
        row = cursor.fetchone()
        assert row["status"] == "sent"
        assert row["score"] is not None


def test_targeted_delivery_notified_article_rejected(delivery_test_db):
    """Requirement 2: NOTIFIED article is rejected."""
    art_id = _seed_article(delivery_test_db, stage="NOTIFIED", analysis_status="completed")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["delivered"] is False
    assert "NOTIFIED" in result["reason"]
    assert mock_send.call_count == 0


def test_targeted_delivery_stored_article_rejected(delivery_test_db):
    """Requirement 3: STORED article is rejected."""
    art_id = _seed_article(delivery_test_db, stage="STORED", analysis_status="completed")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["delivered"] is False
    assert "STORED" in result["reason"]
    assert mock_send.call_count == 0


def test_targeted_delivery_failed_article_rejected(delivery_test_db):
    """Requirement 4: FAILED article is rejected."""
    art_id = _seed_article(delivery_test_db, stage="FAILED", analysis_status="completed")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["delivered"] is False
    assert "FAILED" in result["reason"]
    assert mock_send.call_count == 0


def test_targeted_delivery_missing_article_rejected(delivery_test_db):
    """Requirement 5: Missing article is rejected."""
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(999999, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["delivered"] is False
    assert "does not exist" in result["reason"]
    assert mock_send.call_count == 0


def test_targeted_delivery_incomplete_analysis_rejected(delivery_test_db):
    """Requirement 6: Incomplete analysis is rejected."""
    # Test 6a: Status is 'pending'
    art_id_pending = _seed_article(
        delivery_test_db,
        link="https://example.com/pending",
        stage="ANALYZED",
        analysis_status="pending",
    )
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id_pending, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert "pending" in result["reason"]
    assert mock_send.call_count == 0

    # Test 6b: No analysis row at all
    art_id_no_analysis = _seed_article(
        delivery_test_db,
        link="https://example.com/no_analysis",
        stage="ANALYZED",
        analysis_status=None,
    )
    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result2 = send_notification_for_article(art_id_no_analysis, config=config, db_path=delivery_test_db)

    assert result2["ok"] is False
    assert result2["status"] == "rejected"
    assert mock_send.call_count == 0


def test_targeted_delivery_already_sent_article_rejected(delivery_test_db):
    """Requirement 7: Already-sent article is rejected."""
    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed")
    record_article_delivery(art_id, channel="telegram", score=0.9, status="sent", db_path=delivery_test_db)
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["delivered"] is False
    assert "already been delivered" in result["reason"]
    assert mock_send.call_count == 0


def test_targeted_delivery_score_below_min_score_skipped(delivery_test_db):
    """Requirement 8: Score below TELEGRAM_MIN_SCORE is skipped according to delivery semantics."""
    art_id = _seed_article(
        delivery_test_db,
        stage="ANALYZED",
        analysis_status="completed",
        relevance_score=0.1,
        novelty="incremental",
    )
    config = TelegramConfig(
        bot_token="test-token",
        chat_id="123456",
        enabled=True,
        min_score=0.8,
    )

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "skipped"
    assert result["delivered"] is False
    assert mock_send.call_count == 0

    # Pipeline state updated to NOTIFIED
    from database import get_connection
    with get_connection(delivery_test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        assert cursor.fetchone()["stage"] == "NOTIFIED"
        cursor.execute("SELECT status FROM article_deliveries WHERE article_id = ?", (art_id,))
        assert cursor.fetchone()["status"] == "skipped"


def test_targeted_delivery_disabled_no_network_request(delivery_test_db):
    """Requirement 9: TELEGRAM_ENABLED=false results in NO network request."""
    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed", relevance_score=0.95)
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=False)

    with patch("delivery.telegram.send_telegram_message") as mock_send:
        result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "disabled"
    assert result["delivered"] is False
    assert mock_send.call_count == 0

    # Pipeline state remains ANALYZED when called with default mark_skipped_if_disabled=False
    from database import get_connection
    with get_connection(delivery_test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_id,))
        assert cursor.fetchone()["stage"] == "ANALYZED"


def test_targeted_delivery_isolation_does_not_affect_other_articles(delivery_test_db):
    """Requirement 10: Targeted delivery cannot affect any other article."""
    art_1 = _seed_article(delivery_test_db, title="Art 1", link="https://example.com/1", stage="ANALYZED")
    art_2 = _seed_article(delivery_test_db, title="Art 2", link="https://example.com/2", stage="ANALYZED")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True, min_score=0.0)

    with patch("delivery.telegram.send_telegram_message", return_value={"ok": True}):
        result = send_notification_for_article(art_1, config=config, db_path=delivery_test_db)

    assert result["ok"] is True
    assert result["article_id"] == art_1

    from database import get_connection
    with get_connection(delivery_test_db) as conn:
        cursor = conn.cursor()
        # art_1 moved to NOTIFIED
        cursor.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_1,))
        assert cursor.fetchone()["stage"] == "NOTIFIED"
        # art_2 remains untouched in ANALYZED
        cursor.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art_2,))
        assert cursor.fetchone()["stage"] == "ANALYZED"
        cursor.execute("SELECT 1 FROM article_deliveries WHERE article_id = ?", (art_2,))
        assert cursor.fetchone() is None


def test_targeted_delivery_secrets_never_logged(delivery_test_db, caplog):
    """Requirement 11: Secrets (tokens, chat IDs) are never written to logs."""
    import logging
    secret_token = "SECRET_BOT_TOKEN_XYZ987654321"
    secret_chat = "SECRET_CHAT_ID_11223344"
    config = TelegramConfig(bot_token=secret_token, chat_id=secret_chat, enabled=True, min_score=0.0)

    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed")

    from email.message import Message
    import urllib.error
    http_error = urllib.error.HTTPError(
        url=f"https://api.telegram.org/bot{secret_token}/sendMessage",
        code=403,
        msg="Forbidden",
        hdrs=Message(),
        fp=None,
    )

    with caplog.at_level(logging.DEBUG):
        with patch("delivery.telegram.urllib.request.urlopen", side_effect=http_error):
            result = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["delivered"] is False

    # Verify secrets are NOT anywhere in captured log text
    assert secret_token not in caplog.text
    assert secret_chat not in caplog.text
    assert "botSECRET_BOT_TOKEN" not in caplog.text


def test_targeted_delivery_duplicate_delivery_protection_intact(delivery_test_db):
    """Requirement 12: Existing duplicate-delivery protection remains intact."""
    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True, min_score=0.0)

    mock_resp = {"ok": True, "result": {"message_id": 101}}
    with patch("delivery.telegram.send_telegram_message", return_value=mock_resp) as mock_send:
        # First send attempt: succeeds
        res1 = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)
        assert res1["ok"] is True
        assert res1["delivered"] is True
        assert mock_send.call_count == 1

        # Second send attempt: rejected by duplicate protection
        res2 = send_notification_for_article(art_id, config=config, db_path=delivery_test_db)
        assert res2["ok"] is False
        assert res2["status"] == "rejected"
        assert res2["delivered"] is False
        assert mock_send.call_count == 1  # Network call was NOT repeated


def test_targeted_delivery_via_continuous_monitor_method(delivery_test_db):
    """Verify ContinuousMonitor.send_notification_for_article method."""
    from monitor import ContinuousMonitor
    art_id = _seed_article(delivery_test_db, stage="ANALYZED", analysis_status="completed")
    config = TelegramConfig(bot_token="test-token", chat_id="123456", enabled=True, min_score=0.0)

    monitor = ContinuousMonitor(db_path=delivery_test_db)
    with patch("delivery.telegram.send_telegram_message", return_value={"ok": True}) as mock_send:
        result = monitor.send_notification_for_article(art_id, config=config)

    assert result["ok"] is True
    assert result["status"] == "sent"
    assert result["delivered"] is True
    assert mock_send.call_count == 1

