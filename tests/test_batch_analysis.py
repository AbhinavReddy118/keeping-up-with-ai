"""
Offline tests for Task 2.9.3: Batch LLM Analysis + Persistence.
Uses temporary SQLite databases and MockLLMProvider.
Guarantees zero network calls and full article table immutability.
"""

import json
import sqlite3
import pytest
from database import init_db, get_connection
from storage import save_article
from llm import (
    AnalysisResult,
    MockLLMProvider,
    BatchProcessor,
    run_batch_analysis,
    get_unprocessed_articles,
    save_analysis_success,
    save_analysis_failure,
    get_analysis_by_article_id,
    get_analysis_stats,
    LLMTimeoutError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMMalformedResponseError,
    LLMSchemaValidationError,
    LLMConfigurationError,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create and initialize an isolated SQLite test database."""
    db_file = tmp_path / "test_digest.db"
    init_db(str(db_file))
    return str(db_file)


@pytest.fixture
def sample_articles(temp_db):
    """Seed the temporary database with 5 normalized articles."""
    articles = [
        {
            "title": f"AI Innovation Article #{i}",
            "link": f"https://example.com/ai-news/{i}",
            "source": "AI Tech Daily",
            "published_at": f"2026-09-0{i}T12:00:00Z",
            "summary": f"Summary text for article number {i} discussing deep neural networks.",
            "raw_content": json.dumps({"raw_index": i}),
        }
        for i in range(1, 6)
    ]
    with get_connection(temp_db) as conn:
        for a in articles:
            save_article(a, conn=conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO article_pipeline_state (article_id, stage, created_at, updated_at)
            SELECT id, 'NEW', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP FROM articles
            """
        )
        conn.commit()

    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, link, summary, source, published_at, raw_content FROM articles ORDER BY id ASC")
        return [dict(r) for r in cursor.fetchall()]


@pytest.fixture
def valid_analysis_result():
    return AnalysisResult(
        relevance_score=0.92,
        summary="A major breakthrough in neural network optimization.",
        key_points=["Point A", "Point B"],
        topics=["Artificial Intelligence", "Deep Learning"],
        technologies=["Transformers", "FlashAttention"],
        companies=["Acme AI"],
        people=["Jane Doe"],
        models=["Model-X"],
        event_type="model_release",
        novelty="breakthrough",
        confidence=0.95,
        entities={"companies": ["Acme AI"], "models": ["Model-X"]},
        prompt_version="1.0",
        raw_response='{"simulated": "response"}',
    )


def test_new_article_is_selected_for_analysis(temp_db, sample_articles):
    """1. Unanalyzed articles are selected deterministically by recency descending."""
    unprocessed = get_unprocessed_articles(limit=3, conn=get_connection(temp_db))
    assert len(unprocessed) == 3
    # Ordered by published_at DESC, id DESC: art5 (09-05), art4 (09-04), art3 (09-03)
    assert unprocessed[0]["id"] == sample_articles[4]["id"]
    assert unprocessed[1]["id"] == sample_articles[3]["id"]
    assert unprocessed[2]["id"] == sample_articles[2]["id"]


def test_completed_article_is_skipped(temp_db, sample_articles, valid_analysis_result):
    """2. Completed article is skipped during subsequent selection."""
    top_article_id = sample_articles[4]["id"]
    save_analysis_success(
        article_id=top_article_id,
        result=valid_analysis_result,
        provider="mock",
        model="mock-model",
        conn=get_connection(temp_db),
    )

    unprocessed = get_unprocessed_articles(limit=5, conn=get_connection(temp_db))
    unprocessed_ids = [a["id"] for a in unprocessed]
    assert top_article_id not in unprocessed_ids
    assert len(unprocessed) == 4


def test_successful_analysis_is_persisted(temp_db, sample_articles, valid_analysis_result):
    """3. Successful analysis is persisted with status 'completed' and audited metadata."""
    article_id = sample_articles[0]["id"]
    save_analysis_success(
        article_id=article_id,
        result=valid_analysis_result,
        provider="mock",
        model="mock-model",
        attempts=1,
        conn=get_connection(temp_db),
    )

    record = get_analysis_by_article_id(article_id, conn=get_connection(temp_db))
    assert record is not None
    assert record["article_id"] == article_id
    assert record["status"] == "completed"
    assert record["relevance_score"] == 0.92
    assert record["provider"] == "mock"
    assert record["model"] == "mock-model"
    assert record["attempts"] == 1
    assert record["error_type"] is None
    assert record["error_message"] is None


def test_json_fields_round_trip_correctly(temp_db, sample_articles, valid_analysis_result):
    """4. JSON list and dictionary fields serialize and deserialize correctly."""
    article_id = sample_articles[1]["id"]
    save_analysis_success(
        article_id=article_id,
        result=valid_analysis_result,
        provider="mock",
        model="mock-model",
        conn=get_connection(temp_db),
    )

    record = get_analysis_by_article_id(article_id, conn=get_connection(temp_db))
    assert record is not None
    assert isinstance(record["key_points"], list)
    assert record["key_points"] == ["Point A", "Point B"]
    assert isinstance(record["topics"], list)
    assert record["topics"] == ["Artificial Intelligence", "Deep Learning"]
    assert isinstance(record["entities"], dict)
    assert record["entities"]["companies"] == ["Acme AI"]


def test_duplicate_analysis_cannot_be_created(temp_db, sample_articles, valid_analysis_result):
    """5. UNIQUE constraint prevents duplicate rows; upsert updates existing record."""
    article_id = sample_articles[0]["id"]
    conn = get_connection(temp_db)

    # First save
    save_analysis_success(article_id, valid_analysis_result, "mock", "mock-model", conn=conn)

    # Second save (e.g. updated score)
    updated_result = AnalysisResult(
        relevance_score=0.99,
        summary="Updated summary",
        key_points=["Point Z"],
        topics=["AI"],
        technologies=["Transformers"],
        companies=[],
        people=[],
        models=[],
        event_type="industry_news",
        novelty="incremental",
        confidence=0.9,
    )
    save_analysis_success(article_id, updated_result, "mock", "mock-model", conn=conn)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM article_analysis WHERE article_id = ?", (article_id,))
    assert cursor.fetchone()["cnt"] == 1

    record = get_analysis_by_article_id(article_id, conn=conn)
    assert record is not None
    assert record["relevance_score"] == 0.99
    assert record["summary"] == "Updated summary"


def test_batch_size_is_respected(temp_db, sample_articles):
    """6. Batch size limits the number of processed articles."""
    processor = BatchProcessor(
        provider=MockLLMProvider(),
        batch_size=2,
        max_retries=1,
        db_path=temp_db,
    )
    report = processor.run(limit=2)
    assert report.total_attempted == 2
    assert report.completed == 2
    assert report.failed == 0

    stats = get_analysis_stats(get_connection(temp_db))
    assert stats["completed"] == 2


def test_multiple_articles_are_processed(temp_db, sample_articles):
    """7. Multiple articles in a batch are processed and marked completed."""
    processor = BatchProcessor(
        provider=MockLLMProvider(),
        batch_size=5,
        max_retries=1,
        db_path=temp_db,
    )
    report = processor.run(limit=5)
    assert report.total_attempted == 5
    assert report.completed == 5
    assert report.failed == 0

    stats = get_analysis_stats(get_connection(temp_db))
    assert stats["completed"] == 5
    assert stats["total_analyzed"] == 5


def test_one_article_failure_does_not_stop_batch(temp_db, sample_articles):
    """8. A failing article records failure and allows subsequent articles to complete."""
    # Under recency ordering, batch of 3 selects art5, art4, art3. Fail on art4 (index 3).
    failing_id = sample_articles[3]["id"]

    class FlakyProvider(MockLLMProvider):
        def analyze(self, request):
            if request.article_id == failing_id:
                raise LLMMalformedResponseError("Simulated bad JSON for article")
            return super().analyze(request)

    processor = BatchProcessor(
        provider=FlakyProvider(),
        batch_size=3,
        max_retries=1,
        db_path=temp_db,
    )
    report = processor.run(limit=3)
    assert report.total_attempted == 3
    assert report.completed == 2
    assert report.failed == 1

    # Verify failing article is marked failed
    bad_rec = get_analysis_by_article_id(failing_id, conn=get_connection(temp_db))
    assert bad_rec is not None
    assert bad_rec["status"] == "failed"
    assert bad_rec["error_type"] == "LLMMalformedResponseError"

    # Verify remaining articles in batch are marked completed
    good_rec5 = get_analysis_by_article_id(sample_articles[4]["id"], conn=get_connection(temp_db))
    good_rec3 = get_analysis_by_article_id(sample_articles[2]["id"], conn=get_connection(temp_db))
    assert good_rec5 is not None
    assert good_rec3 is not None
    assert good_rec5["status"] == "completed"
    assert good_rec3["status"] == "completed"


def test_transient_failures_retry(temp_db, sample_articles):
    """9. Transient errors (e.g. timeout) trigger bounded retries."""
    call_counts = {}
    top_article_id = sample_articles[4]["id"]

    class TransientRetryProvider(MockLLMProvider):
        def analyze(self, request):
            aid = request.article_id
            call_counts[aid] = call_counts.get(aid, 0) + 1
            if call_counts[aid] == 1:
                raise LLMTimeoutError("First attempt timed out")
            return super().analyze(request)

    processor = BatchProcessor(
        provider=TransientRetryProvider(),
        batch_size=1,
        max_retries=2,
        db_path=temp_db,
    )
    report = processor.run(limit=1)
    assert report.completed == 1
    assert report.retries_attempted == 1
    assert call_counts[top_article_id] == 2


def test_non_transient_failures_do_not_retry(temp_db, sample_articles):
    """10. Non-transient errors (malformed/schema) do not retry and fail immediately."""
    call_counts = {}
    top_article_id = sample_articles[4]["id"]

    class MalformedProvider(MockLLMProvider):
        def analyze(self, request):
            aid = request.article_id
            call_counts[aid] = call_counts.get(aid, 0) + 1
            raise LLMSchemaValidationError("Missing required schema field")

    processor = BatchProcessor(
        provider=MalformedProvider(),
        batch_size=1,
        max_retries=3,
        db_path=temp_db,
    )
    report = processor.run(limit=1)
    assert report.failed == 1
    assert report.retries_attempted == 0
    # Called exactly once: no retries attempted
    assert call_counts[top_article_id] == 1


def test_failed_analysis_can_be_retried_in_later_run(temp_db, sample_articles):
    """11. A previously failed article is selected again and can succeed in a later run."""
    top_article_id = sample_articles[4]["id"]

    # Step 1: fail it
    fail_provider = MockLLMProvider(simulate_error="unavailable")
    processor_fail = BatchProcessor(provider=fail_provider, batch_size=1, max_retries=0, db_path=temp_db)
    report1 = processor_fail.run(limit=1)
    assert report1.failed == 1

    # Verify status is failed
    rec = get_analysis_by_article_id(top_article_id, conn=get_connection(temp_db))
    assert rec is not None
    assert rec["status"] == "failed"

    # Step 2: run with working provider -> should re-select and complete
    work_provider = MockLLMProvider()
    processor_work = BatchProcessor(provider=work_provider, batch_size=1, max_retries=0, db_path=temp_db)
    report2 = processor_work.run(limit=1)
    assert report2.completed == 1

    # Verify status is now completed
    rec_after = get_analysis_by_article_id(top_article_id, conn=get_connection(temp_db))
    assert rec_after is not None
    assert rec_after["status"] == "completed"
    assert rec_after["relevance_score"] is not None


def test_interrupted_partial_processing_can_resume(temp_db, sample_articles):
    """12. Interrupted/partial processing resumes smoothly without duplicate work."""
    # Process first 2 articles
    p1 = BatchProcessor(provider=MockLLMProvider(), batch_size=2, max_retries=1, db_path=temp_db)
    r1 = p1.run(limit=2)
    assert r1.completed == 2

    # Resume with batch size of 5: only 3 remaining unanalyzed articles should be selected
    p2 = BatchProcessor(provider=MockLLMProvider(), batch_size=5, max_retries=1, db_path=temp_db)
    r2 = p2.run(limit=5)
    assert r2.total_attempted == 3
    assert r2.completed == 3

    # Third run: all 5 articles are completed -> 0 articles attempted
    p3 = BatchProcessor(provider=MockLLMProvider(), batch_size=5, max_retries=1, db_path=temp_db)
    r3 = p3.run(limit=5)
    assert r3.total_attempted == 0
    assert r3.completed == 0


def test_articles_table_remains_unchanged(temp_db, sample_articles):
    """13. The articles table records and count remain strictly unchanged."""
    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_before = cursor.fetchone()["cnt"]
        cursor.execute("SELECT * FROM articles ORDER BY id ASC")
        rows_before = [dict(r) for r in cursor.fetchall()]

    # Run batch
    run_batch_analysis(limit=5, provider=MockLLMProvider(), db_path=temp_db)

    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_after = cursor.fetchone()["cnt"]
        cursor.execute("SELECT * FROM articles ORDER BY id ASC")
        rows_after = [dict(r) for r in cursor.fetchall()]

    assert count_before == count_after == 5
    assert rows_before == rows_after


def test_mock_mode_performs_zero_network_calls(temp_db, sample_articles, monkeypatch):
    """14 & 15. Offline mock mode performs zero network calls during batch processing."""
    import socket

    def guard(*args, **kwargs):
        raise AssertionError("Network socket call attempted during offline batch processing!")

    monkeypatch.setattr(socket, "socket", guard)

    report = run_batch_analysis(limit=3, provider=MockLLMProvider(), db_path=temp_db)
    assert report.completed == 3
    assert report.failed == 0


def test_historical_articles_without_pipeline_state_are_excluded(temp_db):
    """Historical articles with no article_pipeline_state record are excluded from batch query."""
    with get_connection(temp_db) as conn:
        save_article(
            {
                "title": "Historical Unstaged Article 101",
                "link": "https://example.com/hist/101",
                "source": "Legacy Source",
                "published_at": "2026-08-01T10:00:00Z",
                "summary": "Old article before pipeline states existed.",
            },
            conn=conn,
        )
        save_article(
            {
                "title": "Active Eligible Article 102",
                "link": "https://example.com/active/102",
                "source": "Active Source",
                "published_at": "2026-10-01T10:00:00Z",
                "summary": "Eligible article in NEW stage.",
            },
            conn=conn,
        )
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = 'https://example.com/active/102'")
        aid102 = c.fetchone()["id"]
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')", (aid102,))
        conn.commit()

        unprocessed = get_unprocessed_articles(limit=10, conn=conn)

    assert len(unprocessed) == 1
    assert unprocessed[0]["id"] == aid102
    assert unprocessed[0]["title"] == "Active Eligible Article 102"


def test_stored_and_terminal_failed_articles_excluded(temp_db):
    """STORED and terminal FAILED articles are strictly excluded from standalone analysis query."""
    with get_connection(temp_db) as conn:
        stages_to_test = {
            1: "STORED",
            2: "FAILED",
            3: "ANALYZED",
            4: "NOTIFIED",
            5: "NEW",
            6: "ENRICHED",
            7: "ANALYSIS_PENDING",
        }
        for aid, st in stages_to_test.items():
            save_article(
                {
                    "title": f"Article {aid} ({st})",
                    "link": f"https://example.com/stage/{aid}",
                    "source": "Test Source",
                    "published_at": f"2026-10-0{aid}T10:00:00Z",
                    "summary": f"Article with stage {st}",
                },
                conn=conn,
            )
            conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, ?)", (aid, st))
        conn.commit()

        unprocessed = get_unprocessed_articles(limit=20, conn=conn)
        unprocessed_ids = {a["id"] for a in unprocessed}

    # Only NEW (5), ENRICHED (6), and ANALYSIS_PENDING (7) should be selected
    assert unprocessed_ids == {5, 6, 7}
    # STORED (1), FAILED (2), ANALYZED (3), NOTIFIED (4) must be strictly excluded
    assert 1 not in unprocessed_ids
    assert 2 not in unprocessed_ids
    assert 3 not in unprocessed_ids
    assert 4 not in unprocessed_ids


def test_successfully_analyzed_articles_not_selected_again(temp_db, valid_analysis_result):
    """Articles already successfully analyzed are not selected even if stage was not updated."""
    with get_connection(temp_db) as conn:
        save_article(
            {
                "title": "Already Analyzed Article",
                "link": "https://example.com/already-analyzed",
                "source": "AI Daily",
                "published_at": "2026-10-05T10:00:00Z",
                "summary": "Completed analysis article.",
            },
            conn=conn,
        )
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = 'https://example.com/already-analyzed'")
        aid = c.fetchone()["id"]
        # In NEW stage, but has completed analysis
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')", (aid,))
        save_analysis_success(aid, valid_analysis_result, "mock", "mock-model", conn=conn)
        conn.commit()

        unprocessed = get_unprocessed_articles(limit=10, conn=conn)

    assert len(unprocessed) == 0


def test_retry_behavior_for_transient_failure_in_pipeline_state(temp_db):
    """Articles in ANALYSIS_PENDING with a failed analysis record remain eligible for retry."""
    with get_connection(temp_db) as conn:
        save_article(
            {
                "title": "Retry Candidate Article",
                "link": "https://example.com/retry-cand",
                "source": "AI Daily",
                "published_at": "2026-10-06T10:00:00Z",
                "summary": "Article that suffered transient failure.",
            },
            conn=conn,
        )
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = 'https://example.com/retry-cand'")
        aid = c.fetchone()["id"]
        # ANALYSIS_PENDING stage with 1 prior attempt and failed analysis record
        conn.execute(
            "INSERT INTO article_pipeline_state (article_id, stage, analysis_attempts) VALUES (?, 'ANALYSIS_PENDING', 1)",
            (aid,),
        )
        save_analysis_failure(aid, TimeoutError("Network dropped"), "mock", "mock-model", attempts=1, conn=conn)
        conn.commit()

        unprocessed = get_unprocessed_articles(limit=10, conn=conn)

    assert len(unprocessed) == 1
    assert unprocessed[0]["id"] == aid


def test_arxiv_selection_gating_remains_intact(temp_db):
    """ArXiv articles default to STORED and are not selected until promoted to NEW."""
    with get_connection(temp_db) as conn:
        # Article 1: arXiv article in STORED
        save_article(
            {
                "title": "Gated ArXiv Paper",
                "link": "https://arxiv.org/abs/2610.99999",
                "source": "arXiv cs.AI",
                "published_at": "2026-10-07T10:00:00Z",
                "summary": "Abstract: An unpromoted paper.",
            },
            conn=conn,
        )
        # Article 2: arXiv article promoted to NEW
        save_article(
            {
                "title": "Promoted ArXiv Paper",
                "link": "https://arxiv.org/abs/2610.88888",
                "source": "arXiv cs.AI",
                "published_at": "2026-10-07T09:00:00Z",
                "summary": "Abstract: A promoted high-value paper.",
            },
            conn=conn,
        )
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = 'https://arxiv.org/abs/2610.99999'")
        gated_id = c.fetchone()["id"]
        c.execute("SELECT id FROM articles WHERE link = 'https://arxiv.org/abs/2610.88888'")
        promoted_id = c.fetchone()["id"]

        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'STORED')", (gated_id,))
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'NEW')", (promoted_id,))
        conn.commit()

        unprocessed = get_unprocessed_articles(limit=10, conn=conn)
        unprocessed_ids = [a["id"] for a in unprocessed]

    assert promoted_id in unprocessed_ids
    assert gated_id not in unprocessed_ids
    assert len(unprocessed_ids) == 1
