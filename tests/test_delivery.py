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
    send_telegram_message,
)


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
    import urllib.error

    http_error = urllib.error.HTTPError(
        url="https://api.telegram.org/bot/sendMessage",
        code=403,
        msg="Forbidden",
        hdrs={},
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
