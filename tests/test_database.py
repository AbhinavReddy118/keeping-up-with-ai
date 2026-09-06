import sqlite3
from database import init_db, get_connection

def test_init_db():
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles';")
        row = cursor.fetchone()
        assert row is not None
        assert row["name"] == "articles"
