"""Rolling retention window. Old articles move to an archive DB and are
deleted from the live one. Analysis rows follow the article via FK.

Default: 30 days, configurable via RETENTION_DAYS env. Set to 0 to disable.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import DATABASE_PATH

ARCHIVE_PATH = Path(DATABASE_PATH).with_name("news_digest_archive.db")


def _connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _retention_days() -> int:
    try:
        return max(0, int(os.environ.get("RETENTION_DAYS", "30")))
    except (TypeError, ValueError):
        return 30


def prune_articles(db_path=None, archive_path=None, days: int | None = None) -> dict:
    """Move articles older than `days` to an archive DB and delete from main.

    Returns a counts dict. Always returns without raising on empty/missing.
    """
    if days is None:
        days = _retention_days()
    if days <= 0:
        return {"enabled": False, "archived": 0, "deleted": 0, "cutoff": None}

    db_path = Path(db_path) if db_path else Path(DATABASE_PATH)
    archive_path = Path(archive_path) if archive_path else ARCHIVE_PATH

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()

    archived = 0
    deleted = 0

    # Open both; create archive table on demand
    with _connect(db_path) as main, _connect(archive_path) as arch:
        # Mirror schema (CREATE IF NOT EXISTS is idempotent)
        arch.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT UNIQUE NOT NULL,
                summary TEXT,
                source TEXT NOT NULL,
                published_at TEXT,
                raw_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        arch.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_pub ON articles(published_at)"
        )

        old = main.execute(
            "SELECT * FROM articles WHERE published_at IS NOT NULL "
            "AND published_at < ?",
            (cutoff,),
        ).fetchall()

        for row in old:
            # Copy row (skip id so the archive assigns fresh autoincrement)
            arch.execute(
                """
                INSERT OR IGNORE INTO articles
                  (title, link, summary, source, published_at, raw_content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["title"],
                    row["link"],
                    row["summary"],
                    row["source"],
                    row["published_at"],
                    row["raw_content"],
                    row["created_at"],
                ),
            )
            # FK isn't ON DELETE CASCADE, so clean analyses first
            main.execute(
                "DELETE FROM article_analysis WHERE article_id = ?", (row["id"],)
            )
            main.execute("DELETE FROM articles WHERE id = ?", (row["id"],))
            archived += 1
            deleted += 1

        main.commit()
        arch.commit()

    return {
        "enabled": True,
        "archived": archived,
        "deleted": deleted,
        "cutoff": cutoff,
        "archive_path": str(archive_path),
    }


def archive_stats(archive_path=None) -> dict:
    p = Path(archive_path) if archive_path else ARCHIVE_PATH
    if not p.exists():
        return {"exists": False, "total": 0}
    with _connect(p) as c:
        total = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        oldest = c.execute(
            "SELECT MIN(published_at) FROM articles"
        ).fetchone()[0]
        newest = c.execute(
            "SELECT MAX(published_at) FROM articles"
        ).fetchone()[0]
    return {"exists": True, "total": total, "oldest": oldest, "newest": newest}
