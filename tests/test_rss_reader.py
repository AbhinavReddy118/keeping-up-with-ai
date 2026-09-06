import pytest
from unittest.mock import MagicMock, patch
import urllib.error
from ingestion.rss_reader import parse_rss_content, fetch_rss_feed

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
    <title>AI News Sample Feed</title>
    <link>https://example.com/ai-news</link>
    <description>Latest updates in AI</description>
    <item>
        <title>New Breakthrough in LLM Training</title>
        <link>https://example.com/ai-news/llm-breakthrough</link>
        <description>Researchers present a novel approach to training language models efficiently.</description>
        <pubDate>Sun, 06 Sep 2026 12:00:00 GMT</pubDate>
        <guid>https://example.com/ai-news/llm-breakthrough</guid>
    </item>
    <item>
        <title>Open Source AI Agent Framework</title>
        <link>https://example.com/ai-news/agent-framework</link>
        <description>An open-source library for building autonomous tools was released today.</description>
        <pubDate>Sat, 05 Sep 2026 09:30:00 GMT</pubDate>
        <guid>guid-item-102</guid>
    </item>
</channel>
</rss>
"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>AI Research Atom Feed</title>
    <link href="https://example.com/atom"/>
    <updated>2026-09-06T10:00:00Z</updated>
    <id>urn:uuid:60a76c80-d399-11d9-b93c-00059a3c7a00</id>
    <entry>
        <title>Atom Entry: Computer Vision Progress</title>
        <link href="https://example.com/atom/cv-progress"/>
        <id>urn:uuid:12345678-1234-5678-1234-567812345678</id>
        <updated>2026-09-06T10:00:00Z</updated>
        <summary>Overview of recent state-of-the-art vision models.</summary>
    </entry>
</feed>
"""

def test_parse_rss_content_rss2():
    entries = parse_rss_content(SAMPLE_RSS_XML)
    assert len(entries) == 2
    
    first = entries[0]
    assert first["title"] == "New Breakthrough in LLM Training"
    assert first["link"] == "https://example.com/ai-news/llm-breakthrough"
    assert first["summary"] == "Researchers present a novel approach to training language models efficiently."
    assert first["published_at"] == "Sun, 06 Sep 2026 12:00:00 GMT"
    assert first["guid"] == "https://example.com/ai-news/llm-breakthrough"

    second = entries[1]
    assert second["guid"] == "guid-item-102"


def test_parse_rss_content_atom():
    entries = parse_rss_content(SAMPLE_ATOM_XML)
    assert len(entries) == 1
    
    entry = entries[0]
    assert entry["title"] == "Atom Entry: Computer Vision Progress"
    assert entry["link"] == "https://example.com/atom/cv-progress"
    assert entry["summary"] == "Overview of recent state-of-the-art vision models."
    assert entry["guid"] == "urn:uuid:12345678-1234-5678-1234-567812345678"


def test_fetch_rss_feed_invalid_url():
    with pytest.raises(ValueError, match="Invalid feed URL scheme"):
        fetch_rss_feed("ftp://invalid-url.com/rss")


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_mocked_network(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_RSS_XML.encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    entries = fetch_rss_feed("https://example.com/rss")
    assert len(entries) == 2
    assert entries[0]["title"] == "New Breakthrough in LLM Training"


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_http_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://example.com/rss", code=404, msg="Not Found", hdrs={}, fp=None
    )
    with pytest.raises(RuntimeError, match="HTTP 404 error fetching feed"):
        fetch_rss_feed("https://example.com/rss")
