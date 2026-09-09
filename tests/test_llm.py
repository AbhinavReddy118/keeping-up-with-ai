"""
Offline unit tests for the LLM Analysis Foundation (Task 2.9.1).
No real network or external API calls are made.
"""

import pytest
from llm import (
    AnalysisRequest,
    AnalysisResult,
    MockLLMProvider,
    get_llm_provider,
    prepare_article_input,
    analyze_article,
    ANALYSIS_PROMPT_VERSION,
    build_analysis_prompt,
    LLMError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMMalformedResponseError,
    LLMSchemaValidationError,
)


@pytest.fixture
def sample_article():
    return {
        "id": 42,
        "title": "OpenAI Introduces GPT-4o for Advanced Multimodal Reasoning",
        "link": "https://openai.com/news/gpt-4o",
        "source": "OpenAI News",
        "published_at": "2026-05-13T10:00:00Z",
        "summary": "GPT-4o delivers real-time voice, vision, and text processing with improved speed and lower cost.",
        "raw_content": '{"extra": "data"}',
    }


def test_valid_article_to_valid_analysis_result(sample_article):
    """1. Valid article -> valid AnalysisResult."""
    provider = MockLLMProvider()
    result = analyze_article(sample_article, provider=provider)

    assert isinstance(result, AnalysisResult)
    assert 0.0 <= result.relevance_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.summary, str) and len(result.summary) > 0
    assert isinstance(result.key_points, list)
    assert "OpenAI" in result.companies
    assert "GPT-4" in result.models or "GPT-4o" in result.summary
    assert result.event_type in {"model_release", "industry_news", "research_paper"}
    assert result.prompt_version == ANALYSIS_PROMPT_VERSION


def test_missing_summary_produces_valid_request(sample_article):
    """2. Missing summary -> analyzer still produces a valid request and result."""
    article_no_summary = dict(sample_article)
    article_no_summary["summary"] = None

    request = prepare_article_input(article_no_summary)
    assert request.summary is None
    assert request.title == sample_article["title"]

    # Analyzer succeeds even without summary
    provider = MockLLMProvider()
    result = analyze_article(article_no_summary, provider=provider)
    assert isinstance(result, AnalysisResult)
    assert result.relevance_score > 0.5


def test_mock_provider_returns_structured_result(sample_article):
    """3. Mock provider returns structured result conforming to schema."""
    custom_data = {
        "relevance_score": 0.95,
        "summary": "Custom deterministic test summary.",
        "key_points": ["Point 1", "Point 2"],
        "topics": ["Artificial Intelligence", "NLP"],
        "technologies": ["Transformers"],
        "companies": ["OpenAI"],
        "people": [],
        "models": ["GPT-4o"],
        "event_type": "model_release",
        "novelty": "breakthrough",
        "confidence": 0.98,
    }
    provider = MockLLMProvider(custom_response=custom_data)
    result = analyze_article(sample_article, provider=provider)

    assert result.relevance_score == 0.95
    assert result.novelty == "breakthrough"
    assert result.event_type == "model_release"
    assert result.companies == ["OpenAI"]
    assert result.entities["companies"] == ["OpenAI"]

    # Test conversion to dict
    res_dict = result.to_dict()
    assert res_dict["relevance_score"] == 0.95
    assert res_dict["summary"] == "Custom deterministic test summary."


def test_invalid_provider_output_is_rejected(sample_article):
    """4. Invalid provider output is rejected via schema validation error."""
    # Test provider simulating malformed non-JSON
    malformed_provider = MockLLMProvider(simulate_error="malformed")
    with pytest.raises(LLMMalformedResponseError, match="invalid non-JSON output"):
        analyze_article(sample_article, provider=malformed_provider)

    # Test provider simulating schema validation failure (missing required keys)
    schema_err_provider = MockLLMProvider(simulate_error="schema")
    with pytest.raises(LLMSchemaValidationError, match="missing required keys"):
        analyze_article(sample_article, provider=schema_err_provider)

    # Test direct AnalysisResult schema validation with invalid score
    invalid_data = {
        "relevance_score": 1.5,  # Out of bounds (> 1.0)
        "summary": "Invalid test summary",
        "key_points": [],
        "topics": [],
        "technologies": [],
        "companies": [],
        "people": [],
        "models": [],
        "event_type": "model_release",
        "novelty": "incremental",
        "confidence": 0.9,
    }
    with pytest.raises(LLMSchemaValidationError, match="out of bounds"):
        AnalysisResult.from_dict(invalid_data)


def test_missing_or_unsupported_api_configuration():
    """5. Missing/unsupported API configuration is handled correctly."""
    # Attempting to use an unconfigured live provider raises LLMConfigurationError
    with pytest.raises(LLMConfigurationError, match="deferred to a future task"):
        get_llm_provider("openai")

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        get_llm_provider("nonexistent_provider")


def test_provider_failure_does_not_become_fake_successful_result(sample_article):
    """6. Provider failure raises typed exception rather than silently succeeding."""
    timeout_provider = MockLLMProvider(simulate_error="timeout")
    with pytest.raises(LLMTimeoutError, match="timed out"):
        analyze_article(sample_article, provider=timeout_provider)

    unavailable_provider = MockLLMProvider(simulate_error="unavailable")
    with pytest.raises(LLMProviderUnavailableError, match="service temporarily unavailable"):
        analyze_article(sample_article, provider=unavailable_provider)

    rate_limit_provider = MockLLMProvider(simulate_error="rate_limit")
    with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
        analyze_article(sample_article, provider=rate_limit_provider)


def test_prompt_version_is_present():
    """7. Prompt version is present and tracks contract."""
    assert ANALYSIS_PROMPT_VERSION == "1.0"
    prompt = build_analysis_prompt("Test Title", "Test Source", "2026-09-09", "Test Summary")
    assert "Test Title" in prompt
    assert "relevance_score" in prompt
    assert "event_type" in prompt


def test_article_input_sanitization():
    """8. Article input sanitizes inputs, handles truncated lengths, and enforces required fields."""
    # Missing title raises ValueError
    with pytest.raises(ValueError, match="missing required 'title'"):
        prepare_article_input({"source": "OpenAI", "summary": "No title"})

    # Missing source raises ValueError
    with pytest.raises(ValueError, match="missing required 'source'"):
        prepare_article_input({"title": "A Title", "summary": "No source"})

    # Very long summary is truncated safely
    huge_summary = "A" * 3000
    req = prepare_article_input(
        {"title": "Valid Title", "source": "Valid Source", "summary": huge_summary},
        max_summary_chars=500,
    )
    assert len(req.summary) < 550
    assert "[truncated]" in req.summary


def test_no_real_network_call_during_offline_analysis(sample_article, monkeypatch):
    """9. No real network/API call occurs during offline analysis."""
    # Block socket connections to guarantee zero external network calls
    import socket
    def guard(*args, **kwargs):
        raise AssertionError("Unexpected network call attempted during unit test!")
    monkeypatch.setattr(socket, "socket", guard)

    provider = MockLLMProvider()
    result = analyze_article(sample_article, provider=provider)
    assert result is not None
    assert result.summary is not None
