import sqlite3
from config import DATABASE_PATH

def get_connection(db_path=None):
    """Establish and return a connection to the SQLite database."""
    target_path = db_path if db_path is not None else DATABASE_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn

def init_db(db_path=None):
    """Initialize the SQLite database schema for storing ingested articles."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
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
        """)
        conn.commit()
