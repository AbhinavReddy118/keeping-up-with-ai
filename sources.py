"""
Centralized RSS source configuration and validation for the AI Intelligence Monitor.
"""

REQUIRED_KEYS = {"name", "url", "type", "enabled"}
VALID_TYPES = {"rss", "atom"}

# Default configured AI news RSS sources
DEFAULT_SOURCES = [
    {
        "name": "Hacker News AI",
        "url": "https://hnrss.org/newest?q=AI",
        "type": "rss",
        "enabled": True,
    },
    {
        "name": "MIT Tech Review AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "type": "rss",
        "enabled": True,
    },
    {
        "name": "arXiv CS.AI Papers",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "type": "rss",
        "enabled": True,
    },
    {
        "name": "Sample Disabled Feed",
        "url": "https://example.com/disabled-feed.xml",
        "type": "rss",
        "enabled": False,
    },
]


def validate_source_config(source: dict) -> None:
    """
    Validate a single source configuration dictionary.

    Raises:
        ValueError: If any required key is missing, invalid type, or malformed URL.
    """
    if not isinstance(source, dict):
        raise ValueError(f"Source configuration must be a dictionary, got {type(source).__name__}")

    missing_keys = REQUIRED_KEYS - set(source.keys())
    if missing_keys:
        raise ValueError(f"Source config missing required keys: {sorted(missing_keys)}")

    if not isinstance(source["name"], str) or not source["name"].strip():
        raise ValueError("Source 'name' must be a non-empty string")

    if not isinstance(source["url"], str) or not (
        source["url"].startswith("http://") or source["url"].startswith("https://")
    ):
        raise ValueError(f"Source 'url' must be a valid HTTP/HTTPS URL, got: '{source.get('url')}'")

    if source["type"] not in VALID_TYPES:
        raise ValueError(f"Source 'type' must be one of {sorted(VALID_TYPES)}, got: '{source.get('type')}'")

    if not isinstance(source["enabled"], bool):
        raise ValueError(f"Source 'enabled' must be a boolean (True/False), got: {type(source['enabled']).__name__}")


def get_sources(sources_list: list[dict] = None, enabled_only: bool = True) -> list[dict]:
    """
    Retrieve and validate configured RSS sources.

    Args:
        sources_list: Optional list of source dictionaries (defaults to DEFAULT_SOURCES).
        enabled_only: If True, returns only sources where enabled is True.

    Returns:
        List of validated source configuration dictionaries.
    """
    if sources_list is None:
        sources_list = DEFAULT_SOURCES

    if not isinstance(sources_list, list):
        raise ValueError(f"Sources list must be a list of dictionaries, got {type(sources_list).__name__}")

    validated_sources = []
    for idx, source in enumerate(sources_list):
        try:
            validate_source_config(source)
        except ValueError as e:
            raise ValueError(f"Malformed source at index {idx}: {e}") from e

        if not enabled_only or source["enabled"]:
            validated_sources.append(source)

    return validated_sources
