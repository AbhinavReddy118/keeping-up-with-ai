"""
LLM Analysis Foundation Package.
Provides provider-agnostic interfaces, structured schema validation, and article analysis.
"""

from llm.models import AnalysisRequest, AnalysisResult
from llm.exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMMalformedResponseError,
    LLMSchemaValidationError,
)
from llm.prompts import ANALYSIS_PROMPT_VERSION, SYSTEM_PROMPT, build_analysis_prompt
from llm.provider import LLMProvider, MockLLMProvider, get_llm_provider
from llm.analyzer import prepare_article_input, analyze_article

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "LLMError",
    "LLMConfigurationError",
    "LLMProviderUnavailableError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMMalformedResponseError",
    "LLMSchemaValidationError",
    "ANALYSIS_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_analysis_prompt",
    "LLMProvider",
    "MockLLMProvider",
    "get_llm_provider",
    "prepare_article_input",
    "analyze_article",
]
