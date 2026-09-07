import logging
from database import init_db
from ingestion.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting AI Intelligence Monitor...")
    init_db()
    summary = run_pipeline()
    return summary

if __name__ == "__main__":
    main()
