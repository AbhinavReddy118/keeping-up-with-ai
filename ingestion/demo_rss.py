import os
import sys
import logging

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingestion.rss_reader import fetch_rss_feed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Example real RSS feed (Hacker News AI query feed)
DEMO_FEED_URL = "https://hnrss.org/newest?q=AI"

def main():
    logging.info(f"Starting manual RSS fetch test using feed: {DEMO_FEED_URL}")
    try:
        entries = fetch_rss_feed(DEMO_FEED_URL, timeout=10)
        logging.info(f"Successfully fetched {len(entries)} entries from feed.")
        
        print("\n--- First 3 Sample Feed Entries ---")
        for idx, entry in enumerate(entries[:3], 1):
            print(f"\n[{idx}] Title: {entry['title']}")
            print(f"    Link: {entry['link']}")
            print(f"    Published: {entry['published_at']}")
            print(f"    GUID: {entry['guid']}")
            print(f"    Summary Snippet: {entry['summary'][:120]}...")
            
    except Exception as e:
        logging.error(f"Manual RSS test failed: {e}")

if __name__ == "__main__":
    main()
