"""
High-level Article Analyzer orchestration module.
Prepares article inputs, resolves providers, and executes structured analysis.
"""

import logging
from typing import Optional
from llm.models import AnalysisRequest, AnalysisResult
from llm.provider import LLMProvider, get_llm_provider
from llm.exceptions import LLMError

logger = logging.getLogger(__name__)


def prepare_article_input(article: dict, max_summary_chars: int = 4000) -> AnalysisRequest:
    """
    Sanitize and validate an article dictionary into a typed AnalysisRequest.

    Args:
        article: Database or normalized article dictionary.
        max_summary_chars: Maximum character limit for summary to protect prompt token size.

    Returns:
        AnalysisRequest instance.

    Raises:
        ValueError: If required article metadata (title or source) is missing.
    """
    if not isinstance(article, dict):
        raise ValueError(f"Expected article to be a dictionary, got {type(article).__name__}.")

    title = article.get("title")
    if not title or not isinstance(title, str) or not title.strip():
        raise ValueError("Article missing required 'title'.")

    source = article.get("source")
    if not source or not isinstance(source, str) or not source.strip():
        raise ValueError("Article missing required 'source'.")

    link = article.get("link", "")
    content = article.get("content")
    summary = article.get("summary")
    # Prefer enriched article content over RSS summary if available
    text_payload = content if (content and isinstance(content, str) and content.strip()) else summary
    if text_payload and isinstance(text_payload, str):
        text_payload = text_payload.strip()
        if len(text_payload) > max_summary_chars:
            text_payload = text_payload[:max_summary_chars] + "... [truncated]"
    else:
        text_payload = None

    published_at = article.get("published_at")
    article_id = article.get("id")

    return AnalysisRequest(
        article_id=article_id,
        title=title.strip(),
        link=link.strip() if isinstance(link, str) else "",
        source=source.strip(),
        summary=text_payload,
        published_at=published_at,
    )


def analyze_article(article: dict, provider: Optional[LLMProvider] = None) -> AnalysisResult:
    """
    Analyze a single article using the configured or supplied LLM provider.

    Args:
        article: Article dictionary containing title, source, summary, etc.
        provider: Optional LLMProvider instance. Defaults to get_llm_provider().

    Returns:
        Validated AnalysisResult instance.

    Raises:
        ValueError: If article input is malformed.
        LLMError: Subclasses for configuration, timeout, or schema failures.
    """
    request = prepare_article_input(article)

    active_provider = provider if provider is not None else get_llm_provider()

    logger.debug(f"Executing LLM analysis for article '{request.title}' via provider '{active_provider.name}'")

    try:
        result = active_provider.analyze(request)
        return result
    except LLMError as e:
        logger.error(f"LLM analysis failed for article '{request.title}': {e}")
        raise
