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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL UNIQUE,
                relevance_score REAL,
                summary TEXT,
                key_points TEXT,
                topics TEXT,
                technologies TEXT,
                companies TEXT,
                people TEXT,
                models TEXT,
                event_type TEXT,
                novelty TEXT,
                confidence REAL,
                entities TEXT,
                prompt_version TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                raw_response TEXT,
                status TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_analysis_status ON article_analysis(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_analysis_article_id ON article_analysis(article_id)")
        conn.commit()

