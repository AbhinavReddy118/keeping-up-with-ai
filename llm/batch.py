"""
Batch LLM Analysis & Persistence Layer (Task 2.9.3).

Provides deterministic article selection, bounded retries for transient errors,
atomic persistence to the `article_analysis` SQLite table, and crash/resume safety.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Any
from config import LLM_BATCH_SIZE, LLM_MAX_RETRIES
from database import get_connection, init_db
from llm.models import AnalysisResult
from llm.prompts import ANALYSIS_PROMPT_VERSION
from llm.provider import LLMProvider, get_llm_provider
from llm.analyzer import prepare_article_input
from llm.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMMalformedResponseError,
    LLMSchemaValidationError,
    LLMConfigurationError,
)

logger = logging.getLogger(__name__)

# Error classifications for bounded retry logic
TRANSIENT_EXCEPTIONS = (
    LLMTimeoutError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
)

NON_TRANSIENT_EXCEPTIONS = (
    LLMMalformedResponseError,
    LLMSchemaValidationError,
    LLMConfigurationError,
    ValueError,
)


def is_transient_error(exc: Exception) -> bool:
    """Return True if the exception represents a transient failure eligible for retry."""
    return isinstance(exc, TRANSIENT_EXCEPTIONS)


@dataclass
class BatchReport:
    """Summary of batch analysis execution."""
    total_attempted: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    retries_attempted: int = 0
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_attempted": self.total_attempted,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "retries_attempted": self.retries_attempted,
            "details_count": len(self.details),
        }


# ==============================================================================
# REPOSITORY / PERSISTENCE FUNCTIONS
# ==============================================================================

def get_unprocessed_articles(
    limit: int = 10,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """
    Select eligible articles for analysis respecting pipeline state and recency.
    Filters for stages ('NEW', 'ENRICHED', 'ANALYSIS_PENDING') without completed analysis.
    Deterministically ordered by published_at DESC, id DESC (nulls/blanks last).

    Args:
        limit: Maximum number of articles to select.
        conn: Optional active database connection.

    Returns:
        List of article dictionary records.
    """
    query = """
        SELECT a.id, a.title, a.link, a.summary, a.source, a.published_at, a.raw_content, a.content
        FROM articles a
        JOIN article_pipeline_state p ON a.id = p.article_id
        LEFT JOIN article_analysis an ON a.id = an.article_id
        WHERE p.stage IN ('NEW', 'ENRICHED', 'ANALYSIS_PENDING')
          AND (an.id IS NULL OR an.status != 'completed')
        ORDER BY
            CASE WHEN a.published_at IS NULL OR trim(a.published_at) = ''
                 THEN 1 ELSE 0 END,
            a.published_at DESC,
            a.id DESC
        LIMIT ?
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        if should_close:
            conn.close()


def save_analysis_success(
    article_id: int,
    result: AnalysisResult,
    provider: str,
    model: str,
    attempts: int = 1,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """
    Persist a successful AnalysisResult into `article_analysis`.
    Uses SQLite upsert (`ON CONFLICT(article_id) DO UPDATE`) to guarantee idempotency.

    Args:
        article_id: Primary key of the analyzed article.
        result: Validated AnalysisResult instance.
        provider: Provider identifier (e.g. 'mock', 'openai').
        model: Model identifier (e.g. 'gpt-4o-mini').
        attempts: Number of attempts made in this processing run.
        conn: Optional active database connection.
    """
    query = """
        INSERT INTO article_analysis (
            article_id, relevance_score, summary, key_points, topics, technologies,
            companies, people, models, event_type, novelty, confidence, entities,
            prompt_version, provider, model, raw_response, status, error_type, error_message,
            attempts, updated_at
        ) VALUES (
            :article_id, :relevance_score, :summary, :key_points, :topics, :technologies,
            :companies, :people, :models, :event_type, :novelty, :confidence, :entities,
            :prompt_version, :provider, :model, :raw_response, 'completed', NULL, NULL,
            :attempts, CURRENT_TIMESTAMP
        )
        ON CONFLICT(article_id) DO UPDATE SET
            relevance_score = excluded.relevance_score,
            summary = excluded.summary,
            key_points = excluded.key_points,
            topics = excluded.topics,
            technologies = excluded.technologies,
            companies = excluded.companies,
            people = excluded.people,
            models = excluded.models,
            event_type = excluded.event_type,
            novelty = excluded.novelty,
            confidence = excluded.confidence,
            entities = excluded.entities,
            prompt_version = excluded.prompt_version,
            provider = excluded.provider,
            model = excluded.model,
            raw_response = excluded.raw_response,
            status = 'completed',
            error_type = NULL,
            error_message = NULL,
            attempts = article_analysis.attempts + excluded.attempts,
            updated_at = CURRENT_TIMESTAMP
    """
    payload = {
        "article_id": article_id,
        "relevance_score": result.relevance_score,
        "summary": result.summary,
        "key_points": json.dumps(result.key_points),
        "topics": json.dumps(result.topics),
        "technologies": json.dumps(result.technologies),
        "companies": json.dumps(result.companies),
        "people": json.dumps(result.people),
        "models": json.dumps(result.models),
        "event_type": result.event_type,
        "novelty": result.novelty,
        "confidence": result.confidence,
        "entities": json.dumps(result.entities),
        "prompt_version": result.prompt_version,
        "provider": provider,
        "model": model,
        "raw_response": result.raw_response,
        "attempts": max(1, attempts),
    }

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, payload)
        conn.commit()
    finally:
        if should_close:
            conn.close()


def save_analysis_failure(
    article_id: int,
    error: Exception,
    provider: str,
    model: str,
    attempts: int = 1,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """
    Persist a failure record into `article_analysis` with status 'failed' and error metadata.

    Args:
        article_id: Primary key of the article.
        error: Exception encountered during analysis.
        provider: Provider identifier.
        model: Model identifier.
        attempts: Number of attempts made during this run.
        conn: Optional active database connection.
    """
    query = """
        INSERT INTO article_analysis (
            article_id, prompt_version, provider, model, status, error_type, error_message,
            attempts, updated_at
        ) VALUES (
            :article_id, :prompt_version, :provider, :model, 'failed', :error_type, :error_message,
            :attempts, CURRENT_TIMESTAMP
        )
        ON CONFLICT(article_id) DO UPDATE SET
            status = 'failed',
            error_type = excluded.error_type,
            error_message = excluded.error_message,
            attempts = article_analysis.attempts + excluded.attempts,
            updated_at = CURRENT_TIMESTAMP
    """
    payload = {
        "article_id": article_id,
        "prompt_version": ANALYSIS_PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "attempts": max(1, attempts),
    }

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, payload)
        conn.commit()
    finally:
        if should_close:
            conn.close()


def get_analysis_by_article_id(
    article_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[dict]:
    """
    Retrieve an analysis record by article_id and deserialize JSON list/dict fields.

    Args:
        article_id: Primary key of the article.
        conn: Optional active database connection.

    Returns:
        Dictionary representation of the analysis record or None if not found.
    """
    query = "SELECT * FROM article_analysis WHERE article_id = ?"
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (article_id,))
        row = cursor.fetchone()
        if not row:
            return None

        data = dict(row)
        # Deserialize JSON fields if present
        json_fields = [
            "key_points",
            "topics",
            "technologies",
            "companies",
            "people",
            "models",
            "entities",
        ]
        for f in json_fields:
            raw = data.get(f)
            if raw and isinstance(raw, str):
                try:
                    data[f] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        return data
    finally:
        if should_close:
            conn.close()


def get_analysis_stats(conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Get aggregate counts of analysis records.

    Returns:
        dict with total_analyzed, completed, failed.
    """
    query = """
        SELECT 
            COUNT(*) as total_analyzed,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM article_analysis
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        return {
            "total_analyzed": row["total_analyzed"] if row else 0,
            "completed": row["completed"] if row and row["completed"] else 0,
            "failed": row["failed"] if row and row["failed"] else 0,
        }
    finally:
        if should_close:
            conn.close()


# ==============================================================================
# BATCH PROCESSOR ORCHESTRATOR
# ==============================================================================

class BatchProcessor:
    """
    Orchestrates batch article analysis with bounded retries and atomic persistence.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        db_path: Optional[str] = None,
    ):
        init_db(db_path)
        self.provider = provider if provider is not None else get_llm_provider()
        self.batch_size = batch_size if batch_size is not None else LLM_BATCH_SIZE
        self.max_retries = max_retries if max_retries is not None else LLM_MAX_RETRIES
        self.db_path = db_path

    def process_single_article(
        self,
        article: dict,
        conn: Optional[sqlite3.Connection] = None,
    ) -> dict:
        """
        Process exactly one article with bounded retries for transient errors.
        Never allows an exception to escape; returns execution summary for the article.
        """
        raw_id = article.get("id")
        article_id = int(raw_id) if raw_id is not None else 0
        provider_name = self.provider.name
        model_name = getattr(self.provider, "model", "mock-model")

        attempts = 0
        retries_used = 0
        last_error: Optional[Exception] = None

        while attempts <= self.max_retries:
            attempts += 1
            try:
                # 1. Prepare article input
                request = prepare_article_input(article)

                # 2. Call provider
                result = self.provider.analyze(request)

                # 3. Save success
                save_analysis_success(
                    article_id=article_id,
                    result=result,
                    provider=provider_name,
                    model=model_name,
                    attempts=attempts,
                    conn=conn,
                )
                logger.info(f"Article {article_id} successfully analyzed and persisted.")
                return {
                    "article_id": article_id,
                    "status": "completed",
                    "attempts": attempts,
                    "retries": retries_used,
                    "relevance_score": result.relevance_score,
                    "error": None,
                }

            except Exception as e:
                last_error = e
                # Check retry eligibility
                if is_transient_error(e) and attempts <= self.max_retries:
                    retries_used += 1
                    logger.warning(
                        f"Transient error on article {article_id} (attempt {attempts}/{self.max_retries + 1}): {e}. Retrying..."
                    )
                    continue
                else:
                    logger.error(
                        f"Article {article_id} failed on attempt {attempts} ({e.__class__.__name__}): {e}"
                    )
                    break

        # Persist failure record
        actual_error = last_error if last_error is not None else RuntimeError("Unknown analysis failure")
        save_analysis_failure(
            article_id=article_id,
            error=actual_error,
            provider=provider_name,
            model=model_name,
            attempts=attempts,
            conn=conn,
        )

        return {
            "article_id": article_id,
            "status": "failed",
            "attempts": attempts,
            "retries": retries_used,
            "relevance_score": None,
            "error": f"{last_error.__class__.__name__}: {last_error}",
        }

    def run(self, limit: Optional[int] = None) -> BatchReport:
        """
        Execute batch analysis on unprocessed articles.

        Args:
            limit: Optional override for batch size.

    Returns:
            BatchReport containing aggregate execution metrics.
        """
        effective_limit = limit if limit is not None else self.batch_size
        report = BatchReport()

        with get_connection(self.db_path) as conn:
            articles = get_unprocessed_articles(limit=effective_limit, conn=conn)

            if not articles:
                logger.info("No unprocessed articles found in database.")
                return report

            for article in articles:
                report.total_attempted += 1
                detail = self.process_single_article(article, conn=conn)
                report.details.append(detail)

                aid = article.get("id")
                if aid is not None:
                    if detail["status"] == "completed":
                        report.completed += 1
                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = 'ANALYZED',
                                analysis_attempts = analysis_attempts + ?,
                                last_error = NULL,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (detail.get("attempts", 1), aid),
                        )
                    else:
                        report.failed += 1
                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = CASE WHEN analysis_attempts + ? >= 3 THEN 'FAILED' ELSE 'ANALYSIS_PENDING' END,
                                analysis_attempts = analysis_attempts + ?,
                                last_error = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (
                                detail.get("attempts", 1),
                                detail.get("attempts", 1),
                                str(detail.get("error", "Unknown error")),
                                aid,
                            ),
                        )
                    conn.commit()
                else:
                    if detail["status"] == "completed":
                        report.completed += 1
                    else:
                        report.failed += 1

                report.retries_attempted += detail["retries"]

        return report


def run_batch_analysis(
    limit: Optional[int] = None,
    max_retries: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
    db_path: Optional[str] = None,
) -> BatchReport:
    """
    Convenience function to run a batch analysis pass.
    """
    processor = BatchProcessor(
        provider=provider,
        batch_size=limit,
        max_retries=max_retries,
        db_path=db_path,
    )
    return processor.run(limit=limit)
