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
from llm.provider import LLMProvider, MockLLMProvider, OpenAICompatibleProvider, get_llm_provider
from llm.analyzer import prepare_article_input, analyze_article
from llm.batch import (
    BatchReport,
    BatchProcessor,
    run_batch_analysis,
    get_unprocessed_articles,
    save_analysis_success,
    save_analysis_failure,
    get_analysis_by_article_id,
    get_analysis_stats,
    is_transient_error,
)

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
    "OpenAICompatibleProvider",
    "get_llm_provider",
    "prepare_article_input",
    "analyze_article",
    "BatchReport",
    "BatchProcessor",
    "run_batch_analysis",
    "get_unprocessed_articles",
    "save_analysis_success",
    "save_analysis_failure",
    "get_analysis_by_article_id",
    "get_analysis_stats",
    "is_transient_error",
]

