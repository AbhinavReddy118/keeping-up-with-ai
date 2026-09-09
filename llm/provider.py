"""
Provider abstraction and implementations for LLM article analysis.
"""

from abc import ABC, abstractmethod
import json
import logging
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
from llm.prompts import ANALYSIS_PROMPT_VERSION

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'mock', 'openai', 'anthropic')."""
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
            # Return JSON missing required keys
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

        # Identify simple keywords for realistic mock analysis
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

        # Determine relevance score
        ai_signals = ["ai", "model", "learning", "gpt", "neural", "robotics", "intelligence", "agent", "data", "training", "research"]
        matched_signals = sum(1 for s in ai_signals if s in title_lower)
        relevance_score = min(1.0, max(0.6, 0.5 + (matched_signals * 0.1)))

        summary_preview = summary_text[:150] if summary_text else request.title
        clean_summary = f"Analysis of '{request.title}'. {summary_preview}"

        mock_data = {
            "relevance_score": round(relevance_score, 2),
            "summary": clean_summary,
            "key_points": [
                f"Article from {request.source} titled '{request.title}'.",
                f"Primary topic identified as {topics[-1]}.",
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


def get_llm_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to retrieve configured LLMProvider.

    Args:
        provider_name: Provider identifier (defaults to config.LLM_PROVIDER).

    Returns:
        LLMProvider instance.

    Raises:
        LLMConfigurationError: If requested provider is unsupported or missing configuration.
    """
    from config import LLM_PROVIDER

    selected = (provider_name or LLM_PROVIDER or "mock").strip().lower()

    if selected == "mock":
        return MockLLMProvider(**kwargs)

    if selected in {"openai", "anthropic", "gemini", "openrouter"}:
        raise LLMConfigurationError(
            f"Provider '{selected}' is recognized but live API integration is deferred to a future task. "
            "Task 2.9.1 provides the architectural foundation. Use 'mock' provider for current testing."
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider: '{selected}'. Supported providers for Task 2.9.1: 'mock'."
    )
