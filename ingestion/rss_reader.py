import logging
import urllib.request
import urllib.error
import feedparser

logger = logging.getLogger(__name__)

USER_AGENT = "AI-News-Digest/1.0 (+https://github.com/AbhinavReddy118/Ai_news_digest)"

def parse_rss_content(content: str | bytes) -> list[dict]:
    """
    Parse raw RSS/Atom XML content and return structured entry dictionaries.
    
    Args:
        content: String or bytes containing XML feed data.

    Returns:
        List of dictionaries containing extracted fields for each entry.
    """
    feed = feedparser.parse(content)
    
    if feed.bozo and feed.bozo_exception:
        logger.warning(f"Feed parsing completed with warnings/bozo exception: {feed.bozo_exception}")

    entries = []
    for entry in feed.entries:
        parsed_entry = {
            "guid": entry.get("id") or entry.get("guid") or entry.get("link", ""),
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", "").strip(),
            "summary": entry.get("summary") or entry.get("description", ""),
            "published_at": entry.get("published") or entry.get("updated", ""),
        }
        entries.append(parsed_entry)

    return entries


def fetch_rss_feed(feed_url: str, timeout: int = 10) -> list[dict]:
    """
    Fetch an RSS/Atom feed from a URL and return parsed entries.

    Args:
        feed_url: Absolute HTTP/HTTPS URL of the RSS feed.
        timeout: Network request timeout in seconds (default: 10s).

    Returns:
        List of entry dictionaries extracted from the feed.

    Raises:
        ValueError: If feed_url is invalid or missing scheme.
        RuntimeError: If network request fails or times out.
    """
    if not (feed_url.startswith("http://") or feed_url.startswith("https://")):
        raise ValueError(f"Invalid feed URL scheme: '{feed_url}'. URL must start with http:// or https://")

    req = urllib.request.Request(
        feed_url,
        headers={"User-Agent": USER_AGENT}
    )

    try:
        logger.info(f"Fetching RSS feed from: {feed_url}")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_content = response.read()
            return parse_rss_content(xml_content)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error ({e.code}) while fetching feed '{feed_url}': {e.reason}")
        raise RuntimeError(f"HTTP {e.code} error fetching feed: {e.reason}") from e

    except urllib.error.URLError as e:
        logger.error(f"Network error while fetching feed '{feed_url}': {e.reason}")
        raise RuntimeError(f"Failed to reach feed server: {e.reason}") from e

    except TimeoutError as e:
        logger.error(f"Timeout ({timeout}s) exceeded while fetching feed '{feed_url}'")
        raise RuntimeError(f"Request timed out after {timeout} seconds") from e

    except Exception as e:
        logger.error(f"Unexpected error while fetching feed '{feed_url}': {str(e)}")
        raise RuntimeError(f"Unexpected feed fetching error: {str(e)}") from e
