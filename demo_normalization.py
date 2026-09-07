import json
import logging
from sources import get_sources
from ingestion.rss_reader import fetch_rss_feed
from ingestion.normalizer import normalize_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Loading enabled RSS sources...")
    sources = get_sources(enabled_only=True)
    if not sources:
        logging.warning("No enabled sources found.")
        return

    # Select the first configured source for demonstration
    source = sources[0]
    logging.info(f"Fetching raw entries from source: '{source['name']}' ({source['url']})")

    try:
        raw_entries = fetch_rss_feed(source['url'], timeout=10)
        logging.info(f"Fetched {len(raw_entries)} raw entries.")

        if not raw_entries:
            logging.warning("No entries found in feed.")
            return

        # Pick first raw entry and normalize it
        sample_raw = raw_entries[0]
        normalized = normalize_article(sample_raw, source)

        print("\n==================================================")
        print("RAW RSS ENTRY (Before Normalization):")
        print("==================================================")
        print(json.dumps(sample_raw, indent=2))

        print("\n==================================================")
        print("NORMALIZED ARTICLE (After Normalization):")
        print("==================================================")
        print(f"Title       : {normalized['title']}")
        print(f"Link        : {normalized['link']}")
        print(f"Source      : {normalized['source']}")
        print(f"Published At: {normalized['published_at']}")
        print(f"Summary     : {normalized['summary'][:150]}..." if normalized['summary'] else "Summary     : None")
        print(f"Raw Content : {normalized['raw_content'][:100]}... (JSON payload preserved)")
        print("==================================================")

    except Exception as e:
        logging.error(f"Normalization demo failed: {e}")

if __name__ == "__main__":
    main()
