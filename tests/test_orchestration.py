"""Tests for the end-to-end pipeline orchestration."""

import json
from unittest.mock import patch

import pytest

from database import init_db
from llm import MockLLMProvider
from orchestration import run_full_pipeline


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_orchestration.db")
    init_db(db_file)
    return db_file


@pytest.fixture
def sample_sources():
    return [
        {
            "name": "Test Source A",
            "url": "https://example.com/a.xml",
            "type": "rss",
            "enabled": True,
        },
        {
            "name": "Test Source B",
            "url": "https://example.com/b.xml",
            "type": "rss",
            "enabled": True,
        },
    ]


@pytest.fixture
def mock_entries_a():
    return [
        {
            "title": "OpenAI Releases GPT-6",
            "link": "https://example.com/a/gpt6",
            "summary": "Official announcement of GPT-6.",
            "published_at": "2026-09-07T10:00:00Z",
        },
        {
            "title": "DeepMind Achieves AGI Milestone",
            "link": "https://example.com/a/agi",
            "summary": "New results in artificial general intelligence.",
            "published_at": "2026-09-06T15:00:00Z",
        },
    ]


@pytest.fixture
def mock_entries_b():
    return [
        {
            "title": "OpenAI Releases GPT-6",
            "link": "https://example.com/b/gpt6",
            "summary": "Coverage of the GPT-6 release.",
            "published_at": "2026-09-07T11:00:00Z",
        },
        {
            "title": "Microsoft Releases Phi-4",
            "link": "https://example.com/b/phi4",
            "summary": "Microsoft announces Phi-4 model.",
            "published_at": "2026-09-05T12:00:00Z",
        },
    ]


def test_run_full_pipeline_basic(temp_db, sample_sources, mock_entries_a, mock_entries_b):
    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=True,
            analyze=True,
            analysis_limit=10,
            provider=MockLLMProvider(),
            rank=True,
            rank_limit=5,
            deliver=False,
            db_path=temp_db,
        )

    assert "ingestion" in result
    assert "analysis" in result
    assert "ranking" in result
    assert result["ingestion"]["total_sources"] == 2
    assert result["ingestion"]["total_fetched"] == 4
    assert result["analysis"]["completed"] >= 1
    assert result["ranking"]["count"] >= 1


def test_run_full_pipeline_with_mock_provider(temp_db, sample_sources, mock_entries_a, mock_entries_b):
    provider = MockLLMProvider()
    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=True,
            analyze=True,
            analysis_limit=10,
            provider=provider,
            rank=True,
            rank_limit=5,
            deliver=False,
            db_path=temp_db,
        )

    assert result["analysis"]["completed"] >= 1
    assert result["analysis"]["failed"] == 0


def test_run_full_pipeline_dedup_reduces_articles(temp_db, sample_sources, mock_entries_a, mock_entries_b):
    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=True,
            analyze=True,
            analysis_limit=10,
            provider=MockLLMProvider(),
            rank=True,
            rank_limit=10,
            deliver=False,
            db_path=temp_db,
        )

    ingestion = result["ingestion"]
    assert ingestion["total_fetched"] == 4
    assert ingestion["total_inserted"] == 3


def test_run_full_pipeline_no_dedup(temp_db, sample_sources, mock_entries_a, mock_entries_b):
    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=False,
            analyze=False,
            rank=False,
            deliver=False,
            db_path=temp_db,
        )

    assert "ingestion" in result
    assert "analysis" not in result
    assert "ranking" not in result
    assert result["ingestion"]["total_inserted"] == 4


def test_run_full_pipeline_no_analysis(temp_db, sample_sources, mock_entries_a, mock_entries_b):
    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=True,
            analyze=False,
            rank=False,
            deliver=False,
            db_path=temp_db,
        )

    assert "analysis" not in result
    assert "ranking" not in result


def test_run_full_pipeline_telegram_disabled_by_default(
    temp_db, sample_sources, mock_entries_a, mock_entries_b, monkeypatch
):
    monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with patch(
        "ingestion.pipeline.fetch_rss_feed",
        side_effect=[mock_entries_a, mock_entries_b],
    ):
        result = run_full_pipeline(
            sources=sample_sources,
            deduplicate=True,
            analyze=True,
            analysis_limit=10,
            provider=MockLLMProvider(),
            rank=True,
            rank_limit=5,
            deliver=True,
            db_path=temp_db,
        )

    assert result["delivery"]["enabled"] is False


def test_run_full_pipeline_empty_sources(temp_db):
    result = run_full_pipeline(
        sources=[],
        deduplicate=True,
        analyze=False,
        rank=False,
        deliver=False,
        db_path=temp_db,
    )
    assert result["ingestion"]["total_sources"] == 0
    assert result["ingestion"]["total_inserted"] == 0


# ---------- Personalization layer (re-rank by topic weight nudge) ----------

from orchestration import _load_topic_weights, apply_personalization


def test_load_topic_weights_from_env(monkeypatch):
    monkeypatch.setenv("TOPIC_WEIGHTS", "agent:1.5,reasoning:1.3,tool:1.0")
    weights = _load_topic_weights()
    assert weights == {"agent": 1.5, "reasoning": 1.3, "tool": 1.0}


def test_load_topic_weights_empty(monkeypatch):
    monkeypatch.delenv("TOPIC_WEIGHTS", raising=False)
    assert _load_topic_weights() == {}
    monkeypatch.setenv("TOPIC_WEIGHTS", "  ,  ")
    assert _load_topic_weights() == {}


def test_load_topic_weights_garbage_safe(monkeypatch):
    monkeypatch.setenv("TOPIC_WEIGHTS", "agent:not-a-number,tool:1.2,broken:2.0")
    weights = _load_topic_weights()
    assert "agent" not in weights
    assert weights["tool"] == 1.2
    assert "broken" in weights  # 2.0 is numeric; garbage only filters true junk


def test_apply_personalization_bonus_boosts_matching_articles():
    articles = [
        {
            "article_id": 1,
            "title": "An agent learns to use the browser",
            "summary": "Paper about autonomous computer-use agents.",
            "final_score": 0.50,
            "published_at": "2026-09-20T10:00:00Z",
        },
        {
            "article_id": 2,
            "title": "A new transformer benchmark",
            "summary": "We compare models on QA tasks.",
            "final_score": 0.50,
            "published_at": "2026-09-20T11:00:00Z",
        },
    ]
    out = apply_personalization(articles, {"agent": 1.5}, bonus_cap=0.20)
    # article 1 contains 'agent' => boosted
    assert out[0]["article_id"] == 1
    assert out[0]["_personalization_bonus"] > 0
    assert out[0]["final_score"] > 0.50
    # article 2 contains no agent-related keyword => no bonus
    assert out[1]["_personalization_bonus"] == 0


def test_apply_personalization_caps_bonus():
    articles = [
        {
            "article_id": 1,
            "title": "agent reasoning tool agent reasoning tool",
            "summary": "agent agent agent reasoning reasoning",
            "final_score": 0.30,
            "published_at": "2026-09-20T10:00:00Z",
        }
    ]
    out = apply_personalization(articles, {"agent": 5.0, "reasoning": 5.0}, bonus_cap=0.20)
    # Bonus capped at 0.20
    assert out[0]["_personalization_bonus"] == pytest.approx(0.20)
    assert out[0]["final_score"] == pytest.approx(0.50)


def test_apply_personalization_no_weights_passthrough():
    articles = [{"article_id": 1, "title": "x", "summary": "y", "final_score": 0.5}]
    out = apply_personalization(articles, {})
    assert out == articles
    out2 = apply_personalization(articles, None)
    assert out2 == articles
