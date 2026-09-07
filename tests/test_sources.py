import pytest
from sources import get_sources, validate_source_config, DEFAULT_SOURCES

def test_get_sources_default_enabled_only():
    sources = get_sources(enabled_only=True)
    assert len(sources) == 3
    for source in sources:
        assert source["enabled"] is True


def test_get_sources_all_includes_disabled():
    sources = get_sources(enabled_only=False)
    assert len(sources) == 4
    disabled_sources = [s for s in sources if not s["enabled"]]
    assert len(disabled_sources) == 1
    assert disabled_sources[0]["name"] == "Sample Disabled Feed"


def test_source_required_fields():
    sources = get_sources(enabled_only=False)
    required_keys = {"name", "url", "type", "enabled"}
    for source in sources:
        assert required_keys.issubset(source.keys())
        assert isinstance(source["name"], str)
        assert isinstance(source["url"], str)
        assert source["url"].startswith(("http://", "https://"))
        assert source["type"] in {"rss", "atom"}
        assert isinstance(source["enabled"], bool)


def test_validate_source_config_missing_key():
    invalid_source = {
        "name": "Missing URL Source",
        "type": "rss",
        "enabled": True,
    }
    with pytest.raises(ValueError, match="missing required keys"):
        validate_source_config(invalid_source)


def test_validate_source_config_invalid_url():
    invalid_source = {
        "name": "Invalid URL",
        "url": "ftp://example.com/rss",
        "type": "rss",
        "enabled": True,
    }
    with pytest.raises(ValueError, match="must be a valid HTTP/HTTPS URL"):
        validate_source_config(invalid_source)


def test_validate_source_config_invalid_type():
    invalid_source = {
        "name": "Invalid Type",
        "url": "https://example.com/feed",
        "type": "json",
        "enabled": True,
    }
    with pytest.raises(ValueError, match="must be one of"):
        validate_source_config(invalid_source)


def test_validate_source_config_invalid_enabled_boolean():
    invalid_source = {
        "name": "String Enabled",
        "url": "https://example.com/feed",
        "type": "rss",
        "enabled": "true",  # String instead of boolean
    }
    with pytest.raises(ValueError, match="must be a boolean"):
        validate_source_config(invalid_source)
