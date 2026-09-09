"""
Structured analysis contract and data models for article analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from llm.exceptions import LLMSchemaValidationError
from llm.prompts import ANALYSIS_PROMPT_VERSION

VALID_EVENT_TYPES = {
    "model_release",
    "research_paper",
    "tool_release",
    "business_partnership",
    "regulation_policy",
    "industry_news",
    "opinion_editorial",
    "unknown",
}

VALID_NOVELTY_LEVELS = {
    "breakthrough",
    "significant",
    "incremental",
    "unknown",
}


@dataclass
class AnalysisRequest:
    """Standardized input payload sent to an LLM provider."""
    title: str
    link: str
    source: str
    summary: Optional[str] = None
    published_at: Optional[str] = None
    article_id: Optional[int] = None
    prompt_version: str = ANALYSIS_PROMPT_VERSION
    model: str = "mock-model"
    temperature: float = 0.0
    max_tokens: int = 1000

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError("AnalysisRequest title cannot be empty.")
        if not self.source or not self.source.strip():
            raise ValueError("AnalysisRequest source cannot be empty.")


@dataclass
class AnalysisResult:
    """
    Validated structured result of an LLM article analysis.
    """
    relevance_score: float
    summary: str
    key_points: list[str]
    topics: list[str]
    technologies: list[str]
    companies: list[str]
    people: list[str]
    models: list[str]
    event_type: str
    novelty: str
    confidence: float
    entities: dict[str, list[str]] = field(default_factory=dict)
    prompt_version: str = ANALYSIS_PROMPT_VERSION
    raw_response: Optional[str] = None

    def __post_init__(self):
        # 1. Validate relevance_score
        try:
            self.relevance_score = float(self.relevance_score)
        except (ValueError, TypeError):
            raise LLMSchemaValidationError(f"Invalid relevance_score: {self.relevance_score}. Must be a float.")

        if not (0.0 <= self.relevance_score <= 1.0):
            raise LLMSchemaValidationError(
                f"relevance_score {self.relevance_score} is out of bounds. Must be between 0.0 and 1.0."
            )

        # 2. Validate confidence
        try:
            self.confidence = float(self.confidence)
        except (ValueError, TypeError):
            raise LLMSchemaValidationError(f"Invalid confidence: {self.confidence}. Must be a float.")

        if not (0.0 <= self.confidence <= 1.0):
            raise LLMSchemaValidationError(
                f"confidence {self.confidence} is out of bounds. Must be between 0.0 and 1.0."
            )

        # 3. Validate summary
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise LLMSchemaValidationError("summary must be a non-empty string.")
        self.summary = self.summary.strip()

        # 4. Validate list fields
        list_fields = [
            ("key_points", self.key_points),
            ("topics", self.topics),
            ("technologies", self.technologies),
            ("companies", self.companies),
            ("people", self.people),
            ("models", self.models),
        ]
        for name, val in list_fields:
            if not isinstance(val, list):
                raise LLMSchemaValidationError(f"Field '{name}' must be a list of strings, got {type(val).__name__}.")
            for item in val:
                if not isinstance(item, str):
                    raise LLMSchemaValidationError(f"Item in '{name}' must be a string, got {type(item).__name__}.")

        # 5. Validate event_type (normalize to lowercase, map unknown if not recognized)
        clean_event = str(self.event_type).strip().lower() if self.event_type else "unknown"
        if clean_event not in VALID_EVENT_TYPES:
            clean_event = "unknown"
        self.event_type = clean_event

        # 6. Validate novelty
        clean_novelty = str(self.novelty).strip().lower() if self.novelty else "unknown"
        if clean_novelty not in VALID_NOVELTY_LEVELS:
            clean_novelty = "unknown"
        self.novelty = clean_novelty

        # 7. Aggregate entities composite if not populated
        if not self.entities:
            self.entities = {
                "companies": self.companies,
                "models": self.models,
                "people": self.people,
                "technologies": self.technologies,
            }

    @classmethod
    def from_dict(cls, data: dict, prompt_version: str = ANALYSIS_PROMPT_VERSION, raw_response: Optional[str] = None) -> "AnalysisResult":
        """
        Construct and validate an AnalysisResult from a dictionary (e.g. parsed LLM JSON output).
        Raises LLMSchemaValidationError if required fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise LLMSchemaValidationError(f"Expected dictionary for analysis result, got {type(data).__name__}.")

        required_keys = {"relevance_score", "summary", "key_points", "topics", "technologies", "companies", "people", "models", "event_type", "novelty", "confidence"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise LLMSchemaValidationError(f"Analysis result missing required keys: {sorted(missing_keys)}")

        entities = data.get("entities")
        if not isinstance(entities, dict):
            entities = {
                "companies": data.get("companies", []),
                "models": data.get("models", []),
                "people": data.get("people", []),
                "technologies": data.get("technologies", []),
            }

        return cls(
            relevance_score=data["relevance_score"],
            summary=data["summary"],
            key_points=data.get("key_points", []),
            topics=data.get("topics", []),
            technologies=data.get("technologies", []),
            companies=data.get("companies", []),
            people=data.get("people", []),
            models=data.get("models", []),
            event_type=data.get("event_type", "unknown"),
            novelty=data.get("novelty", "unknown"),
            confidence=data.get("confidence", 1.0),
            entities=entities,
            prompt_version=prompt_version,
            raw_response=raw_response,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard JSON-serializable dictionary."""
        return {
            "relevance_score": self.relevance_score,
            "summary": self.summary,
            "key_points": self.key_points,
            "topics": self.topics,
            "technologies": self.technologies,
            "companies": self.companies,
            "people": self.people,
            "models": self.models,
            "event_type": self.event_type,
            "novelty": self.novelty,
            "confidence": self.confidence,
            "entities": self.entities,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
        }
