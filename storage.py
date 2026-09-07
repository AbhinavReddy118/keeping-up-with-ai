"""
SQLite Article Storage & Repository Layer.
Handles article persistence, retrieval, and duplicate URL handling.
"""

import sqlite3
import logging
from database import get_connection

logger = logging.getLogger(__name__)


def save_article(article: dict, conn: sqlite3.Connection = None) -> bool:
    """
    Insert a normalized Article dictionary into the SQLite database.
    
    Args:
        article: Normalized article dictionary containing title, link, summary, source, published_at, raw_content.
        conn: Optional active sqlite3.Connection. If None, a new connection is created.

    Returns:
        bool: True if the article was newly inserted, False if skipped due to duplicate URL link.

    Raises:
        ValueError: If article dictionary is missing required fields.
    """
    required_fields = {"title", "link", "source"}
    if not isinstance(article, dict) or not required_fields.issubset(article.keys()):
        missing = required_fields - set(article.keys()) if isinstance(article, dict) else required_fields
        raise ValueError(f"Article dictionary missing required fields: {sorted(missing)}")

    query = """
        INSERT OR IGNORE INTO articles (title, link, summary, source, published_at, raw_content)
        VALUES (:title, :link, :summary, :source, :published_at, :raw_content)
    """

    payload = {
        "title": article["title"],
        "link": article["link"],
        "summary": article.get("summary"),
        "source": article["source"],
        "published_at": article.get("published_at"),
        "raw_content": article.get("raw_content", "{}"),
    }

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, payload)
        inserted = cursor.rowcount > 0

        if should_close:
            conn.commit()

        if inserted:
            logger.debug(f"Inserted new article: {article['link']}")
        else:
            logger.debug(f"Skipped duplicate article link: {article['link']}")

        return inserted
    finally:
        if should_close:
            conn.close()


def save_articles(articles: list[dict], conn: sqlite3.Connection = None) -> dict:
    """
    Batch insert a list of normalized Article dictionaries.

    Args:
        articles: List of normalized article dictionaries.
        conn: Optional active sqlite3.Connection.

    Returns:
        dict: Summary of operation results e.g. {"inserted": int, "duplicates": int, "total": int}.
    """
    if not isinstance(articles, list):
        raise ValueError("articles must be a list of article dictionaries")

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    inserted_count = 0
    duplicate_count = 0

    try:
        with conn:  # Manage transaction block
            for article in articles:
                if save_article(article, conn=conn):
                    inserted_count += 1
                else:
                    duplicate_count += 1

        return {
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "total": len(articles),
        }
    finally:
        if should_close:
            conn.close()


def get_article_by_link(link: str, conn: sqlite3.Connection = None) -> dict | None:
    """
    Retrieve an article by its exact URL link.

    Args:
        link: Article URL link.
        conn: Optional active sqlite3.Connection.

    Returns:
        dict representation of the database record or None if not found.
    """
    query = "SELECT id, title, link, summary, source, published_at, raw_content, created_at FROM articles WHERE link = ?"

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (link,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            conn.close()


def get_all_articles(limit: int = 100, conn: sqlite3.Connection = None) -> list[dict]:
    """
    Retrieve all stored articles ordered by ID descending.

    Args:
        limit: Maximum number of records to return.
        conn: Optional active sqlite3.Connection.

    Returns:
        List of article dictionary records.
    """
    query = "SELECT id, title, link, summary, source, published_at, raw_content, created_at FROM articles ORDER BY id DESC LIMIT ?"

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        if should_close:
            conn.close()
