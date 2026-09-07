import json
import pytest
from ingestion.normalizer import normalize_article, clean_html, normalize_datetime


@pytest.fixture
def sample_source():
    return {
        "name": "MIT Tech Review AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "type": "rss",
        "enabled": True,
    }


def test_normalize_normal_rss_entry(sample_source):
    raw_entry = {
        "guid": "guid-123",
        "title": "Architecting memory and storage in the AI era",
        "link": "https://www.technologyreview.com/2026/09/04/1140872/article/",
        "summary": "Overview of memory systems designed for modern LLMs.",
        "published_at": "Fri, 04 Sep 2026 14:00:00 GMT",
    }

    article = normalize_article(raw_entry, sample_source)

    assert article["title"] == "Architecting memory and storage in the AI era"
    assert article["link"] == "https://www.technologyreview.com/2026/09/04/1140872/article/"
    assert article["summary"] == "Overview of memory systems designed for modern LLMs."
    assert article["source"] == "MIT Tech Review AI"
    assert article["published_at"] == "2026-09-04T14:00:00Z"

    # Verify raw content is preserved as valid JSON
    raw_payload = json.loads(article["raw_content"])
    assert raw_payload["raw_entry"]["guid"] == "guid-123"
    assert raw_payload["source_info"]["name"] == "MIT Tech Review AI"


def test_normalize_missing_optional_summary(sample_source):
    raw_entry = {
        "title": "Headline without summary",
        "link": "https://example.com/no-summary",
        "summary": "",
        "published_at": "2026-09-06T10:00:00Z",
    }

    article = normalize_article(raw_entry, sample_source)

    assert article["summary"] is None


def test_normalize_html_inside_summary(sample_source):
    raw_entry = {
        "title": "HTML Summary Article",
        "link": "https://example.com/html-summary",
        "summary": "<p>Article URL: <a href='https://example.com'>link</a></p><p>Summary text with &amp; entity.</p>",
        "published_at": "2026-09-06T10:00:00Z",
    }

    article = normalize_article(raw_entry, sample_source)

    assert "<p>" not in article["summary"]
    assert "<a>" not in article["summary"]
    assert "Summary text with & entity." in article["summary"]


def test_normalize_valid_publication_date_iso_and_rfc(sample_source):
    # RFC 822
    entry_rfc = {
        "title": "RFC Date Article",
        "link": "https://example.com/rfc-date",
        "published_at": "Sun, 06 Sep 2026 12:30:00 GMT",
    }
    art_rfc = normalize_article(entry_rfc, sample_source)
    assert art_rfc["published_at"] == "2026-09-06T12:30:00Z"

    # ISO 8601 with timezone offset
    entry_iso = {
        "title": "ISO Date Article",
        "link": "https://example.com/iso-date",
        "published_at": "2026-09-06T18:00:00+05:30",
    }
    art_iso = normalize_article(entry_iso, sample_source)
    assert art_iso["published_at"] == "2026-09-06T12:30:00Z"


def test_normalize_missing_publication_date(sample_source):
    raw_entry = {
        "title": "No Date Article",
        "link": "https://example.com/no-date",
        "published_at": None,
    }

    article = normalize_article(raw_entry, sample_source)

    assert article["published_at"] is None


def test_normalize_malformed_publication_date(sample_source):
    raw_entry = {
        "title": "Malformed Date Article",
        "link": "https://example.com/bad-date",
        "published_at": "Not A Real Date String 12345",
    }

    article = normalize_article(raw_entry, sample_source)

    assert article["published_at"] is None


def test_normalize_missing_required_title(sample_source):
    raw_entry = {
        "title": "   ",  # Whitespace only
        "link": "https://example.com/no-title",
    }

    with pytest.raises(ValueError, match="Article title is missing or empty"):
        normalize_article(raw_entry, sample_source)


def test_normalize_missing_required_link(sample_source):
    raw_entry = {
        "title": "Valid Title",
        "link": "",  # Empty string
    }

    with pytest.raises(ValueError, match="Article link is missing or empty"):
        normalize_article(raw_entry, sample_source)


def test_normalize_invalid_url_scheme(sample_source):
    raw_entry = {
        "title": "Valid Title",
        "link": "ftp://invalid-scheme.com/file",
    }

    with pytest.raises(ValueError, match="Invalid article link URL scheme"):
        normalize_article(raw_entry, sample_source)


def test_normalize_source_metadata_attached():
    raw_entry = {
        "title": "Source Test Article",
        "link": "https://example.com/source-test",
    }

    # Test source as dictionary
    art_dict = normalize_article(raw_entry, {"name": "Hacker News AI"})
    assert art_dict["source"] == "Hacker News AI"

    # Test source as string
    art_str = normalize_article(raw_entry, "arXiv CS.AI")
    assert art_str["source"] == "arXiv CS.AI"

    # Test missing source raises ValueError
    with pytest.raises(ValueError, match="Article source is missing or empty"):
        normalize_article(raw_entry, "")


def test_normalize_preservation_of_raw_content(sample_source):
    raw_entry = {
        "guid": "unique-guid-999",
        "title": "Raw Preservation Test",
        "link": "https://example.com/raw-test",
        "summary": "Original summary",
        "extra_custom_field": "Preserved data",
    }

    article = normalize_article(raw_entry, sample_source)

    raw_json = json.loads(article["raw_content"])
    assert raw_json["raw_entry"]["extra_custom_field"] == "Preserved data"
    assert raw_json["raw_entry"]["guid"] == "unique-guid-999"
