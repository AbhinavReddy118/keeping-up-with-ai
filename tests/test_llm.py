"""
Offline unit tests for the LLM Analysis Foundation & Live Provider Integration (Tasks 2.9.1 & 2.9.2).
No real network or external API calls are made.
"""

import json
import urllib.error
from unittest.mock import patch, MagicMock
import pytest
from llm import (
    AnalysisRequest,
    AnalysisResult,
    MockLLMProvider,
    OpenAICompatibleProvider,
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


@pytest.fixture
def valid_llm_output_dict():
    return {
        "relevance_score": 0.95,
        "summary": "OpenAI launched GPT-4o, an omni-model integrating voice, vision, and text capabilities.",
        "key_points": ["Real-time multimodal reasoning", "Lower latency and cost than GPT-4 Turbo"],
        "topics": ["Artificial Intelligence", "Multimodal AI"],
        "technologies": ["Transformers", "Audio-Vision Integration"],
        "companies": ["OpenAI"],
        "people": [],
        "models": ["GPT-4o"],
        "event_type": "model_release",
        "novelty": "breakthrough",
        "confidence": 0.98,
    }


# ==============================================================================
# EXISTING TASK 2.9.1 TESTS (PRESERVED)
# ==============================================================================

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

    res_dict = result.to_dict()
    assert res_dict["relevance_score"] == 0.95
    assert res_dict["summary"] == "Custom deterministic test summary."


def test_invalid_provider_output_is_rejected(sample_article):
    """4. Invalid provider output is rejected via schema validation error."""
    malformed_provider = MockLLMProvider(simulate_error="malformed")
    with pytest.raises(LLMMalformedResponseError, match="invalid non-JSON output"):
        analyze_article(sample_article, provider=malformed_provider)

    schema_err_provider = MockLLMProvider(simulate_error="schema")
    with pytest.raises(LLMSchemaValidationError, match="missing required keys"):
        analyze_article(sample_article, provider=schema_err_provider)

    invalid_data = {
        "relevance_score": 1.5,
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
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY is required"):
        get_llm_provider("openai", api_key="")

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
    with pytest.raises(ValueError, match="missing required 'title'"):
        prepare_article_input({"source": "OpenAI", "summary": "No title"})

    with pytest.raises(ValueError, match="missing required 'source'"):
        prepare_article_input({"title": "A Title", "summary": "No source"})

    huge_summary = "A" * 3000
    req = prepare_article_input(
        {"title": "Valid Title", "source": "Valid Source", "summary": huge_summary},
        max_summary_chars=500,
    )
    assert len(req.summary) < 550
    assert "[truncated]" in req.summary


def test_no_real_network_call_during_offline_analysis(sample_article, monkeypatch):
    """9. No real network/API call occurs during offline analysis."""
    import socket
    def guard(*args, **kwargs):
        raise AssertionError("Unexpected network call attempted during unit test!")
    monkeypatch.setattr(socket, "socket", guard)

    provider = MockLLMProvider()
    result = analyze_article(sample_article, provider=provider)
    assert result is not None
    assert result.summary is not None


# ==============================================================================
# TASK 2.9.2: LIVE HTTP PROVIDER INTEGRATION TESTS (MOCKED HTTP)
# ==============================================================================

def test_openai_compatible_provider_successful_response(sample_article, valid_llm_output_dict):
    """1. Successful HTTP provider response parses and validates correctly."""
    mock_http_response = json.dumps({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(valid_llm_output_dict),
                }
            }
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_http_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenAICompatibleProvider(api_key="test-key", model="gpt-4o-mini")
        result = analyze_article(sample_article, provider=provider)

        assert isinstance(result, AnalysisResult)
        assert result.relevance_score == 0.95
        assert result.companies == ["OpenAI"]
        assert result.models == ["GPT-4o"]
        assert result.event_type == "model_release"
        assert result.novelty == "breakthrough"
        assert result.raw_response is not None


def test_openai_compatible_provider_malformed_json_response(sample_article):
    """2. Malformed non-JSON response raises LLMMalformedResponseError."""
    mock_http_response = b"<!DOCTYPE html><html><body>Error Gateway 502</body></html>"
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_http_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMMalformedResponseError, match="not valid JSON"):
            analyze_article(sample_article, provider=provider)


def test_openai_compatible_provider_schema_violation(sample_article):
    """3. JSON response missing required schema fields raises LLMSchemaValidationError."""
    incomplete_dict = {
        "summary": "Missing other required fields",
    }
    mock_http_response = json.dumps({
        "choices": [{"message": {"content": json.dumps(incomplete_dict)}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_http_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMSchemaValidationError, match="missing required keys"):
            analyze_article(sample_article, provider=provider)


def test_openai_compatible_provider_http_401_auth_error(sample_article):
    """4. HTTP 401 Unauthorized raises LLMConfigurationError without leaking secrets."""
    secret_key = "super-secret-key-12345"
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        provider = OpenAICompatibleProvider(api_key=secret_key)
        with pytest.raises(LLMConfigurationError) as exc_info:
            analyze_article(sample_article, provider=provider)

        # Confirm that API key is NEVER leaked in exception messages or logs
        assert secret_key not in str(exc_info.value)
        assert "Authentication failed" in str(exc_info.value)


def test_openai_compatible_provider_http_429_rate_limit(sample_article):
    """5. HTTP 429 Rate Limit raises LLMRateLimitError."""
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMRateLimitError, match="rate limit or quota exceeded"):
            analyze_article(sample_article, provider=provider)


def test_openai_compatible_provider_http_503_server_error(sample_article):
    """6. HTTP 503 Server Error raises LLMProviderUnavailableError."""
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMProviderUnavailableError, match="service temporarily unavailable"):
            analyze_article(sample_article, provider=provider)


def test_openai_compatible_provider_timeout(sample_article):
    """7. Network timeout raises LLMTimeoutError."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out after 30.0s")):
        provider = OpenAICompatibleProvider(api_key="test-key", timeout=30.0)
        with pytest.raises(LLMTimeoutError, match="timed out"):
            analyze_article(sample_article, provider=provider)


def test_provider_factory_selection():
    """8. Provider factory correctly instantiates mock, openai, and openrouter providers."""
    # Mock provider
    p_mock = get_llm_provider("mock")
    assert p_mock.name == "mock"
    assert isinstance(p_mock, MockLLMProvider)

    # OpenAI provider
    p_openai = get_llm_provider("openai", api_key="sk-test", model="gpt-4o")
    assert p_openai.name == "openai"
    assert isinstance(p_openai, OpenAICompatibleProvider)
    assert p_openai.model == "gpt-4o"
    assert p_openai.base_url == "https://api.openai.com/v1"

    # OpenRouter provider
    p_openrouter = get_llm_provider("openrouter", api_key="sk-test")
    assert p_openrouter.name == "openrouter"
    assert isinstance(p_openrouter, OpenAICompatibleProvider)
    assert "openrouter.ai" in p_openrouter.base_url


def test_openai_compatible_provider_empty_api_key():
    """9. Empty API key during direct initialization raises LLMConfigurationError."""
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY is required"):
        OpenAICompatibleProvider(api_key="")

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY is required"):
        OpenAICompatibleProvider(api_key="   ")


def test_openai_compatible_provider_missing_choices_or_content(sample_article):
    """10. Responses missing choices or message content raise LLMMalformedResponseError."""
    # Missing choices
    resp_no_choices = json.dumps({"id": "chatcmpl-123"}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = resp_no_choices
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMMalformedResponseError, match="missing 'choices' array"):
            analyze_article(sample_article, provider=provider)

    # Missing message content
    resp_no_content = json.dumps({"choices": [{"message": {}}]}).encode("utf-8")
    mock_resp2 = MagicMock()
    mock_resp2.read.return_value = resp_no_content
    mock_resp2.__enter__.return_value = mock_resp2

    with patch("urllib.request.urlopen", return_value=mock_resp2):
        provider = OpenAICompatibleProvider(api_key="test-key")
        with pytest.raises(LLMMalformedResponseError, match="missing message content"):
            analyze_article(sample_article, provider=provider)


def test_openai_compatible_provider_invalid_model_name():
    """11. Unverified OpenAI model names raise LLMConfigurationError clearly."""
    with pytest.raises(LLMConfigurationError, match="Unsupported or unverified OpenAI model"):
        OpenAICompatibleProvider(api_key="test-key", model="gpt-5.6-luna")


# ==============================================================================
# OLLAMA LOCAL PROVIDER TESTS (TASK 1.3)
# ==============================================================================

def test_ollama_provider_selection_and_defaults():
    """Ollama provider can be instantiated via factory with proper defaults."""
    provider = get_llm_provider("ollama")
    assert provider.name == "ollama"
    assert provider.model == "qwen3:4b"
    assert provider.base_url == "http://localhost:11434"

    custom_provider = get_llm_provider(
        "ollama",
        model="custom-qwen",
        base_url="http://127.0.0.1:11434",
        timeout=45.0,
    )
    assert custom_provider.model == "custom-qwen"
    assert custom_provider.base_url == "http://127.0.0.1:11434"
    assert custom_provider.timeout == 45.0


def test_ollama_provider_successful_analysis(sample_article, valid_llm_output_dict):
    """Successful Ollama response returns validated AnalysisResult."""
    ollama_envelope = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": json.dumps(valid_llm_output_dict)},
        "done": True,
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(ollama_envelope).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider = get_llm_provider("ollama")
        result = analyze_article(sample_article, provider=provider)

        assert isinstance(result, AnalysisResult)
        assert result.relevance_score == 0.95
        assert "GPT-4o" in result.models
        assert "OpenAI" in result.companies
        assert result.event_type == "model_release"
        assert result.novelty == "breakthrough"

        # Verify request URL and payload
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == "http://localhost:11434/api/chat"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "qwen3:4b"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert payload["think"] is False


def test_ollama_provider_handles_thinking_tags(sample_article, valid_llm_output_dict):
    """Reasoning models outputting <think>...</think> before JSON are stripped cleanly."""
    raw_with_think = f"<think>\nAnalyzing the input article...\n</think>\n{json.dumps(valid_llm_output_dict)}"
    ollama_envelope = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": raw_with_think},
        "done": True,
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(ollama_envelope).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = get_llm_provider("ollama")
        result = analyze_article(sample_article, provider=provider)
        assert isinstance(result, AnalysisResult)
        assert result.relevance_score == 0.95


def test_ollama_provider_malformed_envelope_response(sample_article):
    """Non-JSON response from Ollama server raises LLMMalformedResponseError."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html>Server Error</html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMMalformedResponseError, match="not valid JSON"):
            analyze_article(sample_article, provider=provider)


def test_ollama_provider_missing_content(sample_article):
    """Ollama envelope missing content raises LLMMalformedResponseError."""
    ollama_envelope = {"model": "qwen3:4b", "message": {}}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(ollama_envelope).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMMalformedResponseError, match="missing message content"):
            analyze_article(sample_article, provider=provider)


def test_ollama_provider_invalid_json_model_output(sample_article):
    """Model message content that is not valid JSON raises LLMMalformedResponseError."""
    ollama_envelope = {
        "model": "qwen3:4b",
        "message": {"content": "I am unable to analyze this article."},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(ollama_envelope).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMMalformedResponseError, match="Model output content is not valid JSON"):
            analyze_article(sample_article, provider=provider)


def test_ollama_provider_http_failure(sample_article):
    """HTTP 404 or 500 errors from Ollama raise LLMProviderUnavailableError."""
    # 404 Not Found (e.g. model not pulled)
    err_404 = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=err_404):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMProviderUnavailableError, match="404"):
            analyze_article(sample_article, provider=provider)

    # 503 Service Unavailable
    err_503 = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=err_503):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMProviderUnavailableError, match="503"):
            analyze_article(sample_article, provider=provider)


def test_ollama_provider_timeout(sample_article):
    """Network / socket timeout raises LLMTimeoutError."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMTimeoutError, match="timed out"):
            analyze_article(sample_article, provider=provider)


def test_ollama_provider_schema_validation_failure(sample_article):
    """JSON output that violates AnalysisResult schema raises LLMSchemaValidationError."""
    invalid_schema_dict = {
        "relevance_score": 2.5,  # Out of bounds (> 1.0)
        "summary": "Summary text",
        "key_points": ["point 1"],
        "topics": [],
        "technologies": [],
        "companies": [],
        "people": [],
        "models": [],
        "event_type": "invalid_event_type",
        "novelty": "unknown",
        "confidence": 0.9,
    }
    ollama_envelope = {
        "model": "qwen3:4b",
        "message": {"content": json.dumps(invalid_schema_dict)},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(ollama_envelope).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = get_llm_provider("ollama")
        with pytest.raises(LLMSchemaValidationError):
            analyze_article(sample_article, provider=provider)
