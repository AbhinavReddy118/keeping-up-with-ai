import logging
from database import init_db, get_connection
from sources import get_sources
from ingestion.rss_reader import fetch_rss_feed
from ingestion.normalizer import normalize_article
from storage import save_articles, get_all_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Initializing database...")
    init_db()

    logging.info("Loading enabled RSS sources...")
    sources = get_sources(enabled_only=True)
    if not sources:
        logging.warning("No enabled sources found.")
        return

    total_inserted = 0
    total_duplicates = 0

    for source in sources:
        logging.info(f"Fetching RSS feed from: {source['name']} ({source['url']})")
        try:
            raw_entries = fetch_rss_feed(source['url'], timeout=10)
            logging.info(f"Fetched {len(raw_entries)} raw items.")

            # Normalize entries
            normalized_articles = [normalize_article(entry, source) for entry in raw_entries]

            # Save to SQLite database
            result = save_articles(normalized_articles)
            logging.info(f"Storage Result for {source['name']}: {result['inserted']} inserted, {result['duplicates']} duplicates skipped.")

            total_inserted += result["inserted"]
            total_duplicates += result["duplicates"]

        except Exception as e:
            logging.error(f"Error processing source '{source['name']}': {e}")

    print("\n==================================================")
    print("STORAGE SUMMARY")
    print("==================================================")
    print(f"Total New Articles Inserted : {total_inserted}")
    print(f"Total Duplicates Skipped    : {total_duplicates}")

    # Inspect records from SQLite database
    stored_articles = get_all_articles(limit=5)
    print(f"\nTotal Records in Database   : {len(stored_articles)}")
    print("--- Most Recent 3 Stored Articles in Database ---")
    for idx, art in enumerate(stored_articles[:3], 1):
        print(f"\n[{idx}] ID: {art['id']} | Source: {art['source']}")
        print(f"    Title       : {art['title']}")
        print(f"    Link        : {art['link']}")
        print(f"    Published At: {art['published_at']}")

if __name__ == "__main__":
    main()
