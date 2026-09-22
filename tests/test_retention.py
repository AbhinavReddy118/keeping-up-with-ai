"""Tests for the rolling retention / archival module."""

from datetime import datetime, timedelta, timezone

import pytest

from database import init_db, get_connection
from retention import (
    ARCHIVE_PATH,
    archive_stats,
    prune_articles,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_retention.db"
    archive_file = tmp_path / "test_retention_archive.db"
    init_db(str(db_file))
    return str(db_file), str(archive_file)


def _seed_article(db_path: str, title: str, days_old: int, link: str) -> int:
    published = (
        datetime.now(timezone.utc) - timedelta(days=days_old)
    ).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO articles (title, link, summary, source, published_at, raw_content)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, link, f"summary of {title}", "Test Source", published, "{}"),
        )
        conn.commit()
        return cursor.lastrowid


def test_prune_disabled_when_days_zero(temp_db, monkeypatch):
    db_path, archive_path = temp_db
    monkeypatch.setenv("RETENTION_DAYS", "0")
    result = prune_articles(db_path=db_path, archive_path=archive_path)
    assert result["enabled"] is False
    assert result["archived"] == 0
    assert result["deleted"] == 0


def test_prune_archives_old_articles(temp_db):
    db_path, archive_path = temp_db
    old_id = _seed_article(db_path, "Old Article", days_old=60, link="https://a.com/old")
    new_id = _seed_article(db_path, "New Article", days_old=5, link="https://a.com/new")

    result = prune_articles(db_path=db_path, archive_path=archive_path, days=30)
    assert result["enabled"] is True
    assert result["archived"] == 1
    assert result["deleted"] == 1

    with get_connection(db_path) as conn:
        remaining = conn.execute("SELECT id FROM articles ORDER BY id").fetchall()
        assert [r[0] for r in remaining] == [new_id]
        assert old_id not in [r[0] for r in remaining]

    stats = archive_stats(archive_path=archive_path)
    assert stats["exists"] is True
    assert stats["total"] == 1


def test_prune_no_op_when_nothing_old(temp_db):
    db_path, archive_path = temp_db
    _seed_article(db_path, "Recent", days_old=2, link="https://a.com/r")
    result = prune_articles(db_path=db_path, archive_path=archive_path, days=30)
    assert result["archived"] == 0
    assert result["deleted"] == 0


def test_prune_handles_missing_dates(temp_db):
    db_path, archive_path = temp_db
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO articles (title, link, source, published_at)
            VALUES ('No Date', 'https://a.com/nodate', 'Test', NULL)
            """
        )
        conn.commit()
    result = prune_articles(db_path=db_path, archive_path=archive_path, days=30)
    assert result["archived"] == 0
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        assert count == 1


def test_prune_idempotent_archive(temp_db):
    db_path, archive_path = temp_db
    _seed_article(db_path, "Old", days_old=100, link="https://a.com/old")
    prune_articles(db_path=db_path, archive_path=archive_path, days=30)
    result = prune_articles(db_path=db_path, archive_path=archive_path, days=30)
    assert result["archived"] == 0
    stats = archive_stats(archive_path=archive_path)
    assert stats["total"] == 1


def test_prune_custom_days_override(temp_db):
    db_path, archive_path = temp_db
    _seed_article(db_path, "Article", days_old=10, link="https://a.com/ten")
    result = prune_articles(db_path=db_path, archive_path=archive_path, days=7)
    assert result["archived"] == 1
    assert result["deleted"] == 1


def test_archive_stats_when_missing(tmp_path):
    fake_archive = tmp_path / "does_not_exist.db"
    stats = archive_stats(archive_path=str(fake_archive))
    assert stats["exists"] is False


def test_prune_cascades_analyses(temp_db):
    """Analyses for archived articles should be removed too."""
    db_path, archive_path = temp_db
    article_id = _seed_article(
        db_path, "Analyzed Old", days_old=60, link="https://a.com/analyzed"
    )
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO article_analysis
              (article_id, relevance_score, summary, event_type, novelty, confidence,
               prompt_version, provider, model, status)
            VALUES (?, 0.9, 's', 'model_release', 'significant', 0.95,
                    '1.0', 'mock', 'mock-model', 'completed')
            """,
            (article_id,),
        )
        conn.commit()

    prune_articles(db_path=db_path, archive_path=archive_path, days=30)

    with get_connection(db_path) as conn:
        analyses = conn.execute(
            "SELECT COUNT(*) FROM article_analysis WHERE article_id = ?", (article_id,)
        ).fetchone()[0]
        assert analyses == 0
