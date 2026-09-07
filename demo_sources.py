import logging
from sources import get_sources
from ingestion.rss_reader import fetch_rss_feed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Loading enabled RSS sources from configuration...")
    sources = get_sources(enabled_only=True)
    logging.info(f"Loaded {len(sources)} enabled sources.")

    for source in sources:
        print(f"\n========================================")
        print(f"Source Name : {source['name']}")
        print(f"Source URL  : {source['url']}")
        print(f"Source Type : {source['type']}")
        print(f"Enabled     : {source['enabled']}")
        print(f"========================================")
        
        try:
            entries = fetch_rss_feed(source['url'], timeout=10)
            print(f"Fetched {len(entries)} items. Sample headline:")
            if entries:
                print(f"  -> Title: {entries[0]['title']}")
                print(f"  -> Link : {entries[0]['link']}")
        except Exception as e:
            print(f"  Failed to fetch: {e}")

if __name__ == "__main__":
    main()
