"""
Provider abstraction and implementations for LLM article analysis.
"""

from abc import ABC, abstractmethod
import json
import logging
import re
import urllib.request
import urllib.error
from typing import Optional
from llm.models import AnalysisRequest, AnalysisResult
from llm.exceptions import (
    LLMConfigurationError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMMalformedResponseError,
    LLMSchemaValidationError,
)
from llm.prompts import ANALYSIS_PROMPT_VERSION, SYSTEM_PROMPT, build_analysis_prompt

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'mock', 'openai', 'openrouter')."""
        pass

    @abstractmethod
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Execute article analysis and return a validated AnalysisResult.

        Raises:
            LLMError subclasses on failure.
        """
        pass


class MockLLMProvider(LLMProvider):
    """
    Deterministic offline mock provider for automated tests and local development.
    Does not make any network or external API calls.
    """

    def __init__(
        self,
        custom_response: Optional[dict] = None,
        simulate_error: Optional[str] = None,
    ):
        """
        Args:
            custom_response: Optional explicit dictionary to return instead of generated response.
            simulate_error: Optional error type to simulate:
                'timeout', 'rate_limit', 'unavailable', 'malformed', 'schema', 'config'
        """
        self.custom_response = custom_response
        self.simulate_error = simulate_error

    @property
    def name(self) -> str:
        return "mock"

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Simulate analysis deterministically based on request metadata."""
        # 1. Handle error simulation
        if self.simulate_error == "timeout":
            raise LLMTimeoutError("Mock provider request timed out after 10.0 seconds.")
        elif self.simulate_error == "rate_limit":
            raise LLMRateLimitError("Mock provider rate limit exceeded (HTTP 429).")
        elif self.simulate_error == "unavailable":
            raise LLMProviderUnavailableError("Mock provider service temporarily unavailable (HTTP 503).")
        elif self.simulate_error == "malformed":
            raise LLMMalformedResponseError("Mock provider returned invalid non-JSON output: 'ERROR: Internal Server'")
        elif self.simulate_error == "schema":
            bad_data = {"some_irrelevant_key": "data"}
            raise LLMSchemaValidationError(f"Analysis result missing required keys: {bad_data}")
        elif self.simulate_error == "config":
            raise LLMConfigurationError("Mock provider misconfigured: missing credentials.")

        # 2. Use custom response if provided
        if self.custom_response is not None:
            return AnalysisResult.from_dict(
                self.custom_response,
                prompt_version=request.prompt_version,
                raw_response=json.dumps(self.custom_response),
            )

        # 3. Generate deterministic result based on input text
        title_lower = request.title.lower()
        summary_text = (request.summary or "").strip()

        topics = ["Artificial Intelligence"]
        technologies = []
        companies = []
        models = []
        people = []
        event_type = "industry_news"
        novelty = "incremental"

        if "llm" in title_lower or "language model" in title_lower or "gpt" in title_lower:
            topics.append("Natural Language Processing")
            technologies.append("Large Language Models")

        if "vision" in title_lower or "image" in title_lower or "diffusion" in title_lower:
            topics.append("Computer Vision")
            technologies.append("Diffusion Models")

        if "research" in title_lower or "paper" in title_lower or "arxiv" in request.source.lower():
            event_type = "research_paper"

        if "release" in title_lower or "introducing" in title_lower or "announcing" in title_lower or "launches" in title_lower:
            event_type = "model_release"
            novelty = "significant"

        if "openai" in title_lower or "openai" in request.source.lower():
            companies.append("OpenAI")
        if "google" in title_lower or "deepmind" in title_lower or "deepmind" in request.source.lower():
            companies.append("Google DeepMind")
        if "microsoft" in title_lower or "microsoft" in request.source.lower():
            companies.append("Microsoft")
        if "hugging face" in title_lower or "hugging face" in request.source.lower():
            companies.append("Hugging Face")

        if "gpt-4" in title_lower:
            models.append("GPT-4")
        if "gemma" in title_lower:
            models.append("Gemma")
        if "claude" in title_lower:
            models.append("Claude")

        ai_signals = ["ai", "model", "learning", "gpt", "neural", "robotics", "intelligence", "agent", "data", "training", "research"]
        matched_signals = sum(1 for s in ai_signals if s in title_lower)
        relevance_score = min(1.0, max(0.6, 0.5 + (matched_signals * 0.1)))

        summary_preview = summary_text.strip() if summary_text else request.title.strip()
        clean_summary = summary_preview if summary_preview else f"Overview of {request.title}."

        mock_data = {
            "relevance_score": round(relevance_score, 2),
            "summary": clean_summary,
            "key_points": [
                f"Reports on developments regarding {request.title}.",
                f"Covers key technical topics including {topics[-1]}.",
            ],
            "topics": topics,
            "technologies": technologies if technologies else ["Machine Learning"],
            "companies": companies,
            "people": people,
            "models": models,
            "event_type": event_type,
            "novelty": novelty,
            "confidence": 0.95,
        }

        return AnalysisResult.from_dict(
            mock_data,
            prompt_version=request.prompt_version,
            raw_response=json.dumps(mock_data),
        )


VALID_OPENAI_MODEL_PREFIXES = (
    "gpt-4o",
    "gpt-4-",
    "gpt-4",
    "gpt-3.5",
    "o1-",
    "o1",
    "o3-",
    "o3",
    "chatgpt-4o",
)


def validate_model_identifier(provider_name: str, base_url: str, model: str) -> None:
    """Validate model identifier against recognized provider model formats."""
    clean_p = (provider_name or "").strip().lower()
    clean_m = (model or "").strip().lower()
    if clean_p == "openai" and "api.openai.com" in base_url:
        if not any(clean_m == p or clean_m.startswith(p) for p in VALID_OPENAI_MODEL_PREFIXES):
            raise LLMConfigurationError(
                f"Unsupported or unverified OpenAI model: '{model}'. "
                f"Expected standard OpenAI model such as 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', or 'o3-mini'."
            )


class OpenAICompatibleProvider(LLMProvider):
    """
    HTTP REST provider for OpenAI and OpenAI-compatible endpoints (OpenRouter, Groq, local Ollama, etc.).
    Avoids third-party vendor SDK dependencies by using standard library urllib.request.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        provider_name: str = "openai",
    ):
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise LLMConfigurationError("LLM_API_KEY is required and cannot be empty.")

        self._api_key = api_key.strip()
        self.model = model.strip() if model and model.strip() else "gpt-4o-mini"
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.timeout = float(timeout)
        self._provider_name = provider_name.strip().lower()

        # Validate model identifier consistency (Step 5)
        validate_model_identifier(self._provider_name, self.base_url, self.model)

    @property
    def name(self) -> str:
        return self._provider_name

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Execute live HTTP request to chat completions endpoint with structured JSON enforcement.
        """
        user_prompt = build_analysis_prompt(
            title=request.title,
            source=request.source,
            published_at=request.published_at,
            summary=request.summary,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        endpoint = f"{self.base_url}/chat/completions"
        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "AI-News-Digest/1.0",
            },
            method="POST",
        )

        try:
            logger.info(f"Sending LLM analysis request to {self.name} ({self.model})...")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                resp_text = resp_bytes.decode("utf-8")

        except urllib.error.HTTPError as e:
            # Mask credentials completely; never leak API keys in logs or exceptions
            if e.code in (401, 403):
                logger.error(f"Authentication failed with {self.name} API (HTTP {e.code}).")
                raise LLMConfigurationError(
                    f"Authentication failed with {self.name} API (HTTP {e.code}). Please verify LLM_API_KEY."
                ) from e
            elif e.code == 429:
                logger.error(f"{self.name} API rate limit or quota exceeded (HTTP 429).")
                raise LLMRateLimitError(
                    f"{self.name} API rate limit or quota exceeded (HTTP 429)."
                ) from e
            elif e.code in (500, 502, 503, 504):
                logger.error(f"{self.name} API server error (HTTP {e.code}).")
                raise LLMProviderUnavailableError(
                    f"{self.name} API service temporarily unavailable (HTTP {e.code})."
                ) from e
            else:
                logger.error(f"{self.name} API HTTP error: {e.code}")
                raise LLMProviderUnavailableError(
                    f"{self.name} API returned HTTP error {e.code}: {e.reason}"
                ) from e

        except (urllib.error.URLError, TimeoutError) as e:
            # Handle socket / request timeouts
            err_str = str(e).lower()
            if "timed out" in err_str or isinstance(e, TimeoutError):
                logger.error(f"{self.name} request timed out after {self.timeout}s.")
                raise LLMTimeoutError(f"Request to {self.name} API timed out after {self.timeout}s.") from e
            else:
                logger.error(f"Network error connecting to {self.name} API: {e}")
                raise LLMProviderUnavailableError(
                    f"Network error connecting to {self.name} API: {getattr(e, 'reason', e)}"
                ) from e

        except Exception as e:
            logger.error(f"Unexpected error communicating with {self.name} API: {e}")
            raise LLMProviderUnavailableError(
                f"Unexpected error communicating with {self.name} API: {e}"
            ) from e

        # Parse response JSON
        try:
            resp_json = json.loads(resp_text)
        except json.JSONDecodeError as e:
            raise LLMMalformedResponseError(f"{self.name} response is not valid JSON: {e}") from e

        choices = resp_json.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMMalformedResponseError(f"{self.name} response missing 'choices' array.")

        message = choices[0].get("message", {})
        raw_content = message.get("content")
        if not raw_content or not isinstance(raw_content, str):
            raise LLMMalformedResponseError(f"{self.name} response missing message content.")

        # Parse extracted JSON content
        try:
            parsed_analysis = json.loads(raw_content.strip())
        except json.JSONDecodeError as e:
            raise LLMMalformedResponseError(
                f"Model output content is not valid JSON: {raw_content[:200]}"
            ) from e

        # Validate through schema
        return AnalysisResult.from_dict(
            parsed_analysis,
            prompt_version=request.prompt_version,
            raw_response=raw_content,
        )


class OllamaProvider(LLMProvider):
    """
    Local LLM provider using the Ollama HTTP API (Task 1.3).
    Communicates with local Ollama daemon (e.g. running Qwen3 4B) over HTTP without paid API dependencies.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        provider_name: str = "ollama",
    ):
        self.model = model.strip() if model and model.strip() else "qwen3:4b"
        self.base_url = base_url.rstrip("/") if base_url else "http://localhost:11434"
        self.timeout = float(timeout)
        self._provider_name = provider_name.strip().lower()

    @property
    def name(self) -> str:
        return self._provider_name

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Execute structured LLM analysis via Ollama /api/chat with format='json'.
        """
        user_prompt = build_analysis_prompt(
            title=request.title,
            source=request.source,
            published_at=request.published_at,
            summary=request.summary,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": min(request.max_tokens or 800, 1000),
            },
        }

        endpoint = f"{self.base_url}/api/chat"
        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AI-News-Digest/1.0",
            },
            method="POST",
        )

        try:
            logger.info(f"Sending LLM analysis request to {self.name} ({self.model})...")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                resp_text = resp_bytes.decode("utf-8")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.error(f"Ollama model or endpoint not found (HTTP 404): {e.reason}")
                raise LLMProviderUnavailableError(
                    f"Ollama model '{self.model}' or endpoint '{endpoint}' not found (HTTP 404)."
                ) from e
            elif e.code in (500, 502, 503, 504):
                logger.error(f"Ollama server error (HTTP {e.code}): {e.reason}")
                raise LLMProviderUnavailableError(
                    f"Ollama service temporarily unavailable (HTTP {e.code})."
                ) from e
            else:
                logger.error(f"Ollama HTTP error {e.code}: {e.reason}")
                raise LLMProviderUnavailableError(
                    f"Ollama returned HTTP error {e.code}: {e.reason}"
                ) from e

        except (urllib.error.URLError, TimeoutError) as e:
            err_str = str(e).lower()
            if "timed out" in err_str or isinstance(e, TimeoutError):
                logger.error(f"Ollama request timed out after {self.timeout}s.")
                raise LLMTimeoutError(f"Request to Ollama API timed out after {self.timeout}s.") from e
            else:
                logger.error(f"Network error connecting to Ollama daemon at {self.base_url}: {e}")
                raise LLMProviderUnavailableError(
                    f"Could not connect to Ollama daemon at {self.base_url}: {getattr(e, 'reason', e)}"
                ) from e

        except Exception as e:
            logger.error(f"Unexpected error communicating with Ollama: {e}")
            raise LLMProviderUnavailableError(f"Unexpected error communicating with Ollama: {e}") from e

        # Parse Ollama envelope JSON
        try:
            resp_json = json.loads(resp_text)
        except json.JSONDecodeError as e:
            raise LLMMalformedResponseError(f"Ollama response is not valid JSON: {e}") from e

        message = resp_json.get("message", {})
        raw_content = message.get("content")
        if not raw_content or not isinstance(raw_content, str):
            raise LLMMalformedResponseError("Ollama response missing message content.")

        # Clean any thinking block (e.g. <think>...</think>) from reasoning models
        cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # Parse extracted JSON content
        try:
            parsed_analysis = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise LLMMalformedResponseError(
                f"Model output content is not valid JSON: {cleaned_content[:200]}"
            ) from e

        # Validate through schema
        return AnalysisResult.from_dict(
            parsed_analysis,
            prompt_version=request.prompt_version,
            raw_response=raw_content,
        )


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to retrieve configured LLMProvider.

    Args:
        provider_name: Provider identifier (defaults to config.LLM_PROVIDER).
        api_key: Optional API key override (defaults to config.LLM_API_KEY).
        model: Optional model name override (defaults to config.LLM_MODEL).
        base_url: Optional base URL override (defaults to config.LLM_BASE_URL).
        timeout: Network request timeout in seconds.

    Returns:
        LLMProvider instance.

    Raises:
        LLMConfigurationError: If requested provider is unsupported or missing credentials.
    """
    from config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

    selected = (provider_name or LLM_PROVIDER or "mock").strip().lower()

    if selected == "mock":
        return MockLLMProvider(**kwargs)

    if selected == "ollama":
        if model is not None:
            resolved_model = model.strip()
        elif LLM_PROVIDER == "ollama" and LLM_MODEL:
            resolved_model = LLM_MODEL.strip()
        else:
            resolved_model = "qwen3:4b"

        if base_url is not None:
            resolved_base = base_url.strip()
        elif LLM_PROVIDER == "ollama" and LLM_BASE_URL:
            resolved_base = LLM_BASE_URL.strip()
        else:
            resolved_base = "http://localhost:11434"

        # Allow generous timeout for local model inference if default 30s was passed
        resolved_timeout = max(timeout, 180.0) if timeout == 30.0 else timeout

        return OllamaProvider(
            model=resolved_model,
            base_url=resolved_base,
            timeout=resolved_timeout,
            provider_name=selected,
        )

    if selected in {"openai", "openrouter", "openai_compatible"}:
        resolved_key = (api_key if api_key is not None else LLM_API_KEY or "").strip()
        if not resolved_key:
            raise LLMConfigurationError(
                f"LLM_API_KEY is required for provider '{selected}'. "
                "Please configure LLM_API_KEY in your .env file or environment."
            )

        # Provider-specific defaults
        if selected == "openrouter":
            if base_url is not None:
                resolved_base = base_url.strip()
            elif LLM_PROVIDER == "openrouter" and LLM_BASE_URL:
                resolved_base = LLM_BASE_URL.strip()
            else:
                resolved_base = "https://openrouter.ai/api/v1"

            if model is not None:
                resolved_model = model.strip()
            elif LLM_PROVIDER == "openrouter" and LLM_MODEL:
                resolved_model = LLM_MODEL.strip()
            else:
                resolved_model = "openai/gpt-4o-mini"
        else:
            if base_url is not None:
                resolved_base = base_url.strip()
            elif (LLM_PROVIDER in {"openai", "openai_compatible"}) and LLM_BASE_URL:
                resolved_base = LLM_BASE_URL.strip()
            else:
                resolved_base = "https://api.openai.com/v1"

            if model is not None:
                resolved_model = model.strip()
            elif (LLM_PROVIDER in {"openai", "openai_compatible"}) and LLM_MODEL:
                resolved_model = LLM_MODEL.strip()
            else:
                resolved_model = "gpt-4o-mini"

        return OpenAICompatibleProvider(
            api_key=resolved_key,
            model=resolved_model,
            base_url=resolved_base,
            timeout=timeout,
            provider_name=selected,
        )


    raise LLMConfigurationError(
        f"Unsupported LLM provider: '{selected}'. Supported providers: 'mock', 'ollama', 'openai', 'openrouter'."
    )
