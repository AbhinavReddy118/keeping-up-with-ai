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
                content TEXT,
                enrichment_status TEXT DEFAULT 'pending',
                enrichment_error TEXT,
                enriched_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: Ensure enrichment columns exist if upgrading an existing database
        cursor.execute("PRAGMA table_info(articles)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        migration_cols = [
            ("content", "TEXT"),
            ("enrichment_status", "TEXT DEFAULT 'pending'"),
            ("enrichment_error", "TEXT"),
            ("enriched_at", "TIMESTAMP"),
        ]
        for col_name, col_def in migration_cols:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_def}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_enrichment_status ON articles(enrichment_status)")
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

        # Continuous monitor tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_state (
                source_name TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'tier2',
                poll_interval_seconds INTEGER NOT NULL,
                last_polled_at TIMESTAMP,
                last_success_at TIMESTAMP,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                next_due_at TIMESTAMP NOT NULL,
                etag TEXT,
                last_modified TEXT,
                last_error TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_pipeline_state (
                article_id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL DEFAULT 'NEW',
                laya_status TEXT,
                laya_prediction TEXT,
                laya_metadata TEXT,
                enrichment_attempts INTEGER NOT NULL DEFAULT 0,
                analysis_attempts INTEGER NOT NULL DEFAULT 0,
                notification_attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_pipeline_stage ON article_pipeline_state(stage)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL UNIQUE,
                channel TEXT NOT NULL DEFAULT 'telegram',
                score REAL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_deliveries_article_id ON article_deliveries(article_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitor_heartbeat (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'running',
                pid INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

