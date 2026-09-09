"""
Typed exceptions for the LLM analysis layer.
"""


class LLMError(Exception):
    """Base exception for all LLM analysis errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM provider configuration, credentials, or models are invalid or missing."""
    pass


class LLMProviderUnavailableError(LLMError):
    """Raised when an LLM provider service is unreachable or returns server errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM analysis request exceeds the allotted timeout."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits or quotas are exceeded."""
    pass


class LLMMalformedResponseError(LLMError):
    """Raised when a provider returns unparseable, incomplete, or non-JSON output."""
    pass


class LLMSchemaValidationError(LLMError):
    """Raised when parsed LLM output fails schema validation constraints."""
    pass
