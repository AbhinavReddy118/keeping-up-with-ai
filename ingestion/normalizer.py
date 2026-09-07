"""
Article Normalization Module.
Converts heterogeneous raw RSS/Atom entry dictionaries into standard Article representations.
"""

import html
import json
import logging
from datetime import datetime, timezone
import email.utils
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """HTML parser that extracts text content while maintaining spacing for block elements."""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data: str):
        self.text_parts.append(data)

    def handle_endtag(self, tag: str):
        # Insert space for block-level elements to prevent words from running together
        if tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "article", "section"}:
            self.text_parts.append(" ")


def clean_html(raw_html: str | None) -> str | None:
    """
    Strip HTML tags and unescape HTML entities from a string.
    Returns clean text string or None if empty/missing.
    """
    if not raw_html or not isinstance(raw_html, str):
        return None

    stripped = raw_html.strip()
    if not stripped:
        return None

    try:
        parser = HTMLTextExtractor()
        parser.feed(stripped)
        extracted = "".join(parser.text_parts)
        unescaped = html.unescape(extracted)
        cleaned = " ".join(unescaped.split())
        return cleaned if cleaned else None
    except Exception as e:
        logger.warning(f"Failed to parse HTML string during cleaning: {e}")
        unescaped = html.unescape(stripped)
        cleaned = " ".join(unescaped.split())
        return cleaned if cleaned else None


def normalize_datetime(date_str: str | None) -> str | None:
    """
    Convert heterogeneous date strings into standard ISO-8601 UTC string format (YYYY-MM-DDTHH:MM:SSZ).
    Returns None if date_str is missing, empty, or unparseable.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    stripped = date_str.strip()
    if not stripped:
        return None

    dt = None

    # 1. Try ISO-8601 format (e.g. 2026-09-06T10:00:00Z or 2026-09-06T10:00:00+05:30)
    try:
        dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        pass

    # 2. Try RFC 822 / RFC 2822 format (e.g. Sun, 06 Sep 2026 12:00:00 GMT)
    if dt is None:
        try:
            dt = email.utils.parsedate_to_datetime(stripped)
        except Exception:
            pass

    if dt is not None:
        # Convert to UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.debug(f"Could not parse date string into ISO-8601: '{date_str}'")
    return None


def normalize_article(raw_entry: dict, source_info: dict | str) -> dict:
    """
    Normalize a raw RSS/Atom entry dictionary into a standard Article dictionary.

    Args:
        raw_entry: Raw dictionary extracted from RSS/Atom parser.
        source_info: Configured source metadata dict (from sources.py) or source name string.

    Returns:
        Standardized Article dictionary matching SQLite schema fields:
        {
            "title": str,
            "link": str,
            "summary": str | None,
            "source": str,
            "published_at": str | None,
            "raw_content": str (JSON)
        }

    Raises:
        ValueError: If required fields (title, link, source) are missing or invalid.
    """
    if not isinstance(raw_entry, dict):
        raise ValueError(f"raw_entry must be a dictionary, got {type(raw_entry).__name__}")

    # Extract source name
    source_name = None
    if isinstance(source_info, dict):
        source_name = source_info.get("name")
    elif isinstance(source_info, str):
        source_name = source_info.strip()

    if not source_name or not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("Article source is missing or empty")

    source_name = source_name.strip()

    # Validate required title
    title = raw_entry.get("title")
    if not title or not isinstance(title, str) or not title.strip():
        raise ValueError("Article title is missing or empty")
    title = title.strip()

    # Validate required link
    link = raw_entry.get("link")
    if not link or not isinstance(link, str) or not link.strip():
        raise ValueError("Article link is missing or empty")
    link = link.strip()

    if not (link.startswith("http://") or link.startswith("https://")):
        raise ValueError(f"Invalid article link URL scheme: '{link}'")

    # Clean optional summary
    raw_summary = raw_entry.get("summary") or raw_entry.get("description")
    summary = clean_html(raw_summary)

    # Normalize publication date
    raw_date = raw_entry.get("published_at") or raw_entry.get("published") or raw_entry.get("updated")
    published_at = normalize_datetime(raw_date)

    # Preserve original raw entry and source context as a JSON string
    raw_content_payload = {
        "raw_entry": raw_entry,
        "source_info": source_info if isinstance(source_info, (dict, str)) else str(source_info),
    }
    raw_content = json.dumps(raw_content_payload, default=str, sort_keys=True)

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "source": source_name,
        "published_at": published_at,
        "raw_content": raw_content,
    }
