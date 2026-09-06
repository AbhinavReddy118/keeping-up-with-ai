import logging
from database import init_db, get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Initializing AI Intelligence Monitor database...")
    init_db()
    
    # Simple verification check
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles';")
        table_exists = cursor.fetchone()
        
    if table_exists:
        logging.info("Database initialized successfully. Table 'articles' is ready.")
    else:
        logging.error("Database initialization failed. Table 'articles' was not created.")

if __name__ == "__main__":
    main()
