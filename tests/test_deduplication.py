import pytest
from deduplication import (
    normalize_url,
    normalize_title_for_dedup,
    content_fingerprint,
    calculate_title_similarity,
    analyze_duplicates,
)


def test_normalize_url():
    # Fragment stripping
    assert normalize_url("https://example.com/post#section2") == "https://example.com/post"

    # Tracking parameter stripping
    url_with_params = "https://example.com/post?utm_source=twitter&utm_medium=social&page=2&ref=hackernews"
    assert normalize_url(url_with_params) == "https://example.com/post?page=2"

    # Trailing slash stripping
    assert normalize_url("https://example.com/blog/article/") == "https://example.com/blog/article"
    assert normalize_url("https://example.com/") == "https://example.com/"

    # Scheme and netloc lowercasing
    assert normalize_url("HTTPS://EXAMPLE.COM/Page") == "https://example.com/Page"

    # Empty and None handling
    assert normalize_url("") == ""
    assert normalize_url(None) == ""


def test_normalize_title_for_dedup():
    assert normalize_title_for_dedup("OpenAI Releases GPT-6!") == "openai releases gpt 6"
    assert normalize_title_for_dedup("  OpenAI   releases  GPT-6  ") == "openai releases gpt 6"
    assert normalize_title_for_dedup("Special @#$% Characters") == "special characters"
    assert normalize_title_for_dedup("") == ""
    assert normalize_title_for_dedup(None) == ""


def test_content_fingerprint_equality():
    fp1 = content_fingerprint("OpenAI Announces GPT-5", "Summary of GPT-5 features.")
    fp2 = content_fingerprint("openai announces gpt 5!", "Summary of GPT-5 features...")
    assert fp1 == fp2

    fp3 = content_fingerprint("Different Title", "Summary of GPT-5 features.")
    assert fp1 != fp3


def test_title_similarity_identical():
    sim = calculate_title_similarity("OpenAI releases GPT-5", "OpenAI Releases GPT-5!")
    assert sim["ratio"] == 1.0
    assert sim["jaccard"] == 1.0
    assert sim["combined"] == 1.0


def test_title_similarity_similar():
    # Minor word change / addition
    sim = calculate_title_similarity(
        "OpenAI announces new frontier GPT model",
        "OpenAI unveils new frontier GPT model"
    )
    assert sim["ratio"] >= 0.80
    assert sim["jaccard"] >= 0.60


def test_title_similarity_unrelated():
    sim = calculate_title_similarity(
        "Breakthrough in Quantum Computing Algorithms",
        "Recipe for chocolate chip cookies"
    )
    assert sim["ratio"] < 0.30
    assert sim["jaccard"] == 0.0


def test_analyze_duplicates_empty_dataset():
    res = analyze_duplicates([])
    assert res["total_articles"] == 0
    assert res["candidate_groups"] == []
    assert res["exact_url_duplicates"] == 0


def test_analyze_duplicates_exact_and_cross_source():
    articles = [
        # Group 1: Exact title match across 2 different sources
        {
            "id": 1,
            "title": "OpenAI Releases GPT-6",
            "link": "https://openai.com/news/gpt-6",
            "summary": "Official announcement.",
            "source": "OpenAI News",
            "published_at": "2026-09-07T10:00:00Z",
        },
        {
            "id": 2,
            "title": "OpenAI Releases GPT-6!",
            "link": "https://news.ycombinator.com/item?id=12345",
            "summary": "Hacker news thread.",
            "source": "Hacker News AI",
            "published_at": "2026-09-07T10:30:00Z",
        },
        # Group 2: High lexical similarity (probable duplicate) within same source
        {
            "id": 3,
            "title": "Introducing o3 reasoning system card",
            "link": "https://openai.com/news/o3-card",
            "summary": "Details on o3.",
            "source": "OpenAI News",
            "published_at": "2026-09-06T12:00:00Z",
        },
        {
            "id": 4,
            "title": "Introducing o3 and o4 reasoning system card",
            "link": "https://openai.com/news/o3-o4-card",
            "summary": "Details on o3 and o4.",
            "source": "OpenAI News",
            "published_at": "2026-09-06T14:00:00Z",
        },
        # Group 3: Completely unrelated article
        {
            "id": 5,
            "title": "Novel Approach to Quantum State Tomography",
            "link": "https://arxiv.org/abs/2609.0001",
            "summary": None,  # Test null summary
            "source": "arXiv CS.AI Papers",
            "published_at": "2026-09-07T08:00:00Z",
        },
    ]

    res = analyze_duplicates(articles)

    assert res["total_articles"] == 5

    # Should detect 1 exact title match across sources
    assert len(res["exact_title_groups"]) == 1
    assert len(res["cross_source_groups"]) >= 1

    cross_group = res["cross_source_groups"][0]
    assert cross_group["match_type"] == "exact_title_duplicate"
    assert cross_group["article_a"]["source"] != cross_group["article_b"]["source"]

    # Unrelated article should not match anything
    matched_ids = set()
    for g in res["candidate_groups"]:
        matched_ids.add(g["article_a"]["id"])
        matched_ids.add(g["article_b"]["id"])
    assert 5 not in matched_ids
