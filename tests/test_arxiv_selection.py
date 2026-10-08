"""
Unit and integration tests for the ArXiv Analysis Selection Layer.

Covers all 16 specified requirements:
1. New arXiv article -> STORED
2. New non-arXiv article -> NEW
3. All announce types remain eligible (new, cross, replace, replace-cross)
4. Scoring is deterministic
5. Scoring categories work (Foundation, Reasoning, Systems, Safety, Generative)
6. Acronym detection works (LLM, VLM, MoE, RAG, RLHF)
7. Generic RL/agent terms don't dominate (safeguard cap)
8. Ranking is deterministic (score DESC, published_at DESC, article_id ASC)
9. Daily budget is enforced
10. Budget persists across multiple monitor cycles
11. Same article isn't promoted twice
12. Selected articles become NEW
13. Unselected articles remain STORED
14. ANALYZED articles remain untouched by reclassification
15. NOTIFIED articles remain untouched by reclassification
16. Existing non-arXiv processing remains unchanged
17. Diagnostic reporting mechanism output verification
"""

import sqlite3
from unittest.mock import patch
import pytest

from arxiv_selector import (
    is_arxiv_article,
    extract_announce_type,
    extract_abstract,
    score_arxiv_article,
    promote_top_arxiv_candidates,
    reclassify_unprocessed_arxiv_articles,
    get_daily_promoted_count,
    get_arxiv_selection_report,
    format_arxiv_selection_report,
)
from database import init_db, get_connection
from monitor import ContinuousMonitor
from llm.provider import MockLLMProvider


@pytest.fixture(autouse=True)
def default_mock_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")


@pytest.fixture
def temp_db(tmp_path):
    """Create an isolated test database."""
    db_file = tmp_path / "test_selection.db"
    init_db(str(db_file))
    return str(db_file)


@pytest.fixture
def mock_provider():
    """Mock LLM provider for testing analysis pipeline."""
    return MockLLMProvider()


# ------------------------------------------------------------------------------
# 1. New arXiv article -> STORED
# ------------------------------------------------------------------------------
def test_new_arxiv_article_stored(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)
    mock_arxiv_entries = [
        {
            "guid": "arxiv-test-1",
            "title": "A New Large Language Model Framework",
            "link": "https://arxiv.org/abs/2610.99991",
            "summary": "Announce Type: new Abstract: We present a new language model for reasoning.",
            "published_at": "2026-10-08T00:00:00Z",
        }
    ]

    arxiv_source = {
        "source_name": "arXiv CS.AI Papers",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "poll_interval_seconds": 86400,
    }

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_arxiv_entries, None, None, False)):
        inserted = monitor.poll_single_source(arxiv_source)

    assert inserted == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = ?", (mock_arxiv_entries[0]["link"],))
        art = c.fetchone()
        assert art is not None

        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art["id"],))
        p = c.fetchone()
        assert p is not None
        assert p["stage"] == "STORED"


# ------------------------------------------------------------------------------
# 2. New non-arXiv article -> NEW
# ------------------------------------------------------------------------------
def test_new_non_arxiv_article_new(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db)
    mock_openai_entries = [
        {
            "guid": "openai-test-1",
            "title": "OpenAI Introduces Next Generation Model",
            "link": "https://openai.com/news/next-generation",
            "summary": "OpenAI announces the next generation model.",
            "published_at": "2026-10-08T00:00:00Z",
        }
    ]

    openai_source = {
        "source_name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
        "poll_interval_seconds": 1200,
    }

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_openai_entries, None, None, False)):
        inserted = monitor.poll_single_source(openai_source)

    assert inserted == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM articles WHERE link = ?", (mock_openai_entries[0]["link"],))
        art = c.fetchone()
        assert art is not None

        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = ?", (art["id"],))
        p = c.fetchone()
        assert p is not None
        assert p["stage"] == "NEW"


# ------------------------------------------------------------------------------
# 3. All announce types remain eligible
# ------------------------------------------------------------------------------
def test_all_announce_types_eligible(temp_db):
    announce_types = ["new", "cross", "replace", "replace-cross"]
    with get_connection(temp_db) as conn:
        for i, ann in enumerate(announce_types, start=1):
            conn.execute(
                """
                INSERT INTO articles (id, title, link, summary, source, published_at)
                VALUES (?, ?, ?, ?, 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
                """,
                (
                    i,
                    f"Paper {i} on Large Language Model Reasoning",
                    f"https://arxiv.org/abs/2610.0000{i}",
                    f"arXiv:2610.0000{i} Announce Type: {ann} Abstract: Large language model with reasoning and agentic planning.",
                ),
            )
            conn.execute(
                "INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'STORED')",
                (i,),
            )
        conn.commit()

    # Promote with budget=4: all 4 must be promoted regardless of announce type
    with get_connection(temp_db) as conn:
        promoted = promote_top_arxiv_candidates(daily_budget=4, conn=conn, target_date_utc="2026-10-08")
    assert set(promoted) == {1, 2, 3, 4}


# ------------------------------------------------------------------------------
# 4. Scoring is deterministic
# ------------------------------------------------------------------------------
def test_scoring_is_deterministic():
    title = "Scaling Latent Reasoning in Large Language Model Agents via Speculative Decoding"
    summary = "Announce Type: new Abstract: We introduce speculative decoding for reasoning in large language model agents."
    score_1, details_1 = score_arxiv_article(title, summary)
    for _ in range(50):
        score_2, details_2 = score_arxiv_article(title, summary)
        assert score_1 == score_2
        assert details_1 == details_2


# ------------------------------------------------------------------------------
# 5. Scoring categories work
# ------------------------------------------------------------------------------
def test_scoring_categories_work():
    # Foundation title match (+3.0) + Safety title match (+3.0) + 2-category synergy (+1.5) + new bonus (+1.0) = 8.5
    title = "Multimodal Alignment in Transformers"
    summary = "Announce Type: new Abstract: Novel techniques for vision models."
    score, details = score_arxiv_article(title, summary, announce_type="new")
    assert "foundation_model" in details["matched_categories"]
    assert "safety_security" in details["matched_categories"]
    assert details["synergy_bonus"] == 1.5
    assert score >= 8.0


# ------------------------------------------------------------------------------
# 6. Acronym detection works
# ------------------------------------------------------------------------------
def test_acronym_detection_works():
    # LLM and LLMs
    sc1, det1 = score_arxiv_article("Benchmarking LLMs on Reasoning", "Abstract: Evaluation of reasoning.")
    assert "LLM" in det1["title_matches"].get("foundation_model", [])

    # VLM and VLMs
    sc2, det2 = score_arxiv_article("A Study on VLMs", "Abstract: Vision models.")
    assert "VLM" in det2["title_matches"].get("foundation_model", [])

    # MoE
    sc3, det3 = score_arxiv_article("Efficient Routing in MoE Models", "Abstract: Mixture of experts architecture.")
    assert "MoE" in det3["title_matches"].get("foundation_model", [])

    # RAG
    sc4, det4 = score_arxiv_article("Evaluating RAG Systems for Question Answering", "Abstract: Retrieval augmented methods.")
    assert "RAG" in det4["title_matches"].get("foundation_model", [])

    # RLHF
    sc5, det5 = score_arxiv_article("Direct Alignment Beyond RLHF", "Abstract: Reinforcement learning from human feedback.")
    assert "RLHF" in det5["title_matches"].get("reasoning_agents", [])


# ------------------------------------------------------------------------------
# 7. Generic RL/agent terms don't dominate
# ------------------------------------------------------------------------------
def test_generic_rl_agent_terms_do_not_dominate():
    # Classical engineering RL paper with multiple generic matches (traffic on-ramp)
    title = "Multi-Agent Reinforcement Learning for Freeway Traffic Control"
    abstract = "Abstract: We use deep reinforcement learning and agent coordination for freeway traffic."
    score, details = score_arxiv_article(title, abstract, announce_type="new")
    # Generic cap must be applied, preventing score from exceeding 3.5 (2.5 base + 1.0 announce)
    assert details["generic_cap_applied"] is True
    assert score <= 3.5

    # Conversely, a paper with specific frontier terms gets high score without cap
    f_title = "Large Language Model Agents with Speculative Decoding and Alignment"
    f_abstract = "Abstract: LLM agents using RLHF and speculative decoding."
    f_score, f_details = score_arxiv_article(f_title, f_abstract, announce_type="new")
    assert f_details["generic_cap_applied"] is False
    assert f_score >= 10.0


# ------------------------------------------------------------------------------
# 8. Ranking is deterministic
# ------------------------------------------------------------------------------
def test_ranking_is_deterministic(temp_db):
    with get_connection(temp_db) as conn:
        # Paper 1: high score (10.0)
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (1, 'Large Language Model Reasoning Agent with Speculative Decoding', 'https://arxiv.org/abs/1', 'Announce Type: new Abstract: LLM reasoning agent.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
            """
        )
        # Paper 2: moderate score, newer date
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (2, 'Vision-Language Alignment', 'https://arxiv.org/abs/2', 'Announce Type: new Abstract: Multimodal models.', 'arXiv CS.AI Papers', '2026-10-08T12:00:00Z')
            """
        )
        # Paper 3: moderate score, older date
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (3, 'Vision-Language Alignment', 'https://arxiv.org/abs/3', 'Announce Type: new Abstract: Multimodal models.', 'arXiv CS.AI Papers', '2026-10-07T12:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (1, 'STORED'), (2, 'STORED'), (3, 'STORED')")
        conn.commit()

    promoted = promote_top_arxiv_candidates(daily_budget=3, conn=get_connection(temp_db), target_date_utc="2026-10-08")
    # Paper 1 must be first (highest score), Paper 2 second (same score as 3, but newer date), Paper 3 third
    assert promoted == [1, 2, 3]


# ------------------------------------------------------------------------------
# 9. Daily budget is enforced
# ------------------------------------------------------------------------------
def test_daily_budget_enforced(temp_db):
    with get_connection(temp_db) as conn:
        for i in range(1, 21):
            conn.execute(
                """
                INSERT INTO articles (id, title, link, summary, source, published_at)
                VALUES (?, ?, ?, 'Announce Type: new Abstract: Deep learning study.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
                """,
                (i, f"Paper {i} on Deep Learning", f"https://arxiv.org/abs/{i}"),
            )
            conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'STORED')", (i,))
        conn.commit()

    # Budget = 5
    promoted = promote_top_arxiv_candidates(daily_budget=5, conn=get_connection(temp_db), target_date_utc="2026-10-08")
    assert len(promoted) == 5


# ------------------------------------------------------------------------------
# 10. Budget persists across multiple monitor cycles
# ------------------------------------------------------------------------------
def test_budget_persists_across_multiple_monitor_cycles(temp_db):
    monitor = ContinuousMonitor(db_path=temp_db, arxiv_daily_budget=5)

    with get_connection(temp_db) as conn:
        for i in range(1, 15):
            conn.execute(
                """
                INSERT INTO articles (id, title, link, summary, source, published_at)
                VALUES (?, ?, ?, 'Announce Type: new Abstract: LLM research.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
                """,
                (i, f"Paper {i}", f"https://arxiv.org/abs/{i}"),
            )
            conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (?, 'STORED')", (i,))
        conn.commit()

    # Cycle 1: Promotes 5
    res1 = monitor.step(force_poll=False)
    assert res1["promoted"] == 5

    # Cycle 2: Same day, budget exhausted -> 0 promoted
    res2 = monitor.step(force_poll=False)
    assert res2["promoted"] == 0

    # Total promoted today in DB log is exactly 5
    with get_connection(temp_db) as conn:
        count = get_daily_promoted_count(conn=conn)
        assert count == 5


# ------------------------------------------------------------------------------
# 11. Same article isn't promoted twice
# ------------------------------------------------------------------------------
def test_same_article_is_not_promoted_twice(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (100, 'Important Frontier Model', 'https://arxiv.org/abs/100', 'Announce Type: new Abstract: LLM reasoning.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (100, 'STORED')")
        conn.commit()

    promoted_1 = promote_top_arxiv_candidates(daily_budget=10, conn=get_connection(temp_db), target_date_utc="2026-10-08")
    assert promoted_1 == [100]

    # Run again next day (target_date_utc="2026-10-09")
    promoted_2 = promote_top_arxiv_candidates(daily_budget=10, conn=get_connection(temp_db), target_date_utc="2026-10-09")
    # Article 100 is no longer STORED (it is NEW or later stage) -> not promoted twice
    assert 100 not in promoted_2


# ------------------------------------------------------------------------------
# 12. Selected articles become NEW
# ------------------------------------------------------------------------------
def test_selected_articles_become_new(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (50, 'Major Breakthrough in Multimodal AI', 'https://arxiv.org/abs/50', 'Announce Type: new Abstract: Multimodal vision-language model.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (50, 'STORED')")
        conn.commit()

    promote_top_arxiv_candidates(daily_budget=5, conn=get_connection(temp_db), target_date_utc="2026-10-08")

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 50")
        assert c.fetchone()["stage"] == "NEW"


# ------------------------------------------------------------------------------
# 13. Unselected articles remain STORED
# ------------------------------------------------------------------------------
def test_unselected_articles_remain_stored(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (1, 'Top Paper on Large Language Model', 'https://arxiv.org/abs/1', 'Announce Type: new Abstract: LLM reasoning.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z'),
                   (2, 'Second Paper on General Optimization', 'https://arxiv.org/abs/2', 'Announce Type: new Abstract: Optimization.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (1, 'STORED'), (2, 'STORED')")
        conn.commit()

    # Budget 1: Only paper 1 is promoted, paper 2 remains STORED
    promote_top_arxiv_candidates(daily_budget=1, conn=get_connection(temp_db), target_date_utc="2026-10-08")

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 1")
        assert c.fetchone()["stage"] == "NEW"

        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 2")
        assert c.fetchone()["stage"] == "STORED"


# ------------------------------------------------------------------------------
# 14. ANALYZED articles remain untouched
# ------------------------------------------------------------------------------
def test_analyzed_articles_remain_untouched(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (10, 'Previously Analyzed arXiv Paper', 'https://arxiv.org/abs/10', 'Announce Type: new Abstract: Analyzed.', 'arXiv CS.AI Papers', '2026-10-05T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (10, 'ANALYZED')")
        conn.commit()

    # Run backlog reclassification
    with get_connection(temp_db) as conn:
        reclassified = reclassify_unprocessed_arxiv_articles(conn=conn)
        assert reclassified == 0

        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 10")
        assert c.fetchone()["stage"] == "ANALYZED"


# ------------------------------------------------------------------------------
# 15. NOTIFIED articles remain untouched
# ------------------------------------------------------------------------------
def test_notified_articles_remain_untouched(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at)
            VALUES (20, 'Previously Notified arXiv Paper', 'https://arxiv.org/abs/20', 'Announce Type: new Abstract: Notified.', 'arXiv CS.AI Papers', '2026-10-05T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (20, 'NOTIFIED')")
        conn.commit()

    with get_connection(temp_db) as conn:
        reclassified = reclassify_unprocessed_arxiv_articles(conn=conn)
        assert reclassified == 0

        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 20")
        assert c.fetchone()["stage"] == "NOTIFIED"


# ------------------------------------------------------------------------------
# 16. Existing non-arXiv processing remains unchanged
# ------------------------------------------------------------------------------
def test_existing_non_arxiv_processing_remains_unchanged(temp_db, mock_provider):
    monitor = ContinuousMonitor(db_path=temp_db, provider=mock_provider)

    mock_entries = [
        {
            "guid": "deepmind-1",
            "title": "Google DeepMind Releases Gemini Update",
            "link": "https://deepmind.google/blog/gemini-update",
            "summary": "DeepMind releases an update for Gemini with enhanced reasoning capabilities.",
            "published_at": "2026-10-08T00:00:00Z",
        }
    ]

    deepmind_source = {
        "source_name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "poll_interval_seconds": 1200,
    }

    with patch("monitor.fetch_rss_feed_conditional", return_value=(mock_entries, None, None, False)):
        inserted = monitor.poll_single_source(deepmind_source)

    assert inserted == 1

    # Ingested as NEW
    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 1")
        assert c.fetchone()["stage"] == "NEW"

    # Process pending articles through enrichment + mock Qwen analysis
    analyzed = monitor.process_pending_articles()
    assert analyzed == 1

    with get_connection(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT stage FROM article_pipeline_state WHERE article_id = 1")
        assert c.fetchone()["stage"] == "ANALYZED"

        c.execute("SELECT status FROM article_analysis WHERE article_id = 1")
        assert c.fetchone()["status"] == "completed"


# ------------------------------------------------------------------------------
# 17. Diagnostic reporting mechanism output verification
# ------------------------------------------------------------------------------
def test_diagnostic_report_mechanism(temp_db):
    with get_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (id, title, link, summary, source, published_at, created_at)
            VALUES (1, 'Frontier Vision-Language Model', 'https://arxiv.org/abs/1', 'Announce Type: new Abstract: VLM architecture.', 'arXiv CS.AI Papers', '2026-10-08T00:00:00Z', '2026-10-08 04:00:00')
            """
        )
        conn.execute("INSERT INTO article_pipeline_state (article_id, stage) VALUES (1, 'STORED')")
        conn.commit()

    report_data = get_arxiv_selection_report(daily_budget=50, conn=get_connection(temp_db), target_date_utc="2026-10-08")
    assert report_data["total_arxiv_articles"] == 1
    assert report_data["daily_budget"] == 50
    assert report_data["stored_count"] == 1
    assert report_data["today_remaining_budget"] == 50
    assert len(report_data["top_20_candidates"]) == 1

    text = format_arxiv_selection_report(report_data)
    assert "ARXIV ANALYSIS SELECTION DIAGNOSTIC" in text
    assert "Frontier Vision-Language Model" in text


# ------------------------------------------------------------------------------
# 18. Synergy bonus levels (+1.5 for >=2, +2.5 for >=3)
# ------------------------------------------------------------------------------
def test_synergy_bonus_levels():
    # 1 category: only foundation_model -> synergy = 0.0
    _, det1 = score_arxiv_article("Transformer Architecture", "Abstract: Study of transformers.", announce_type="new")
    assert len(det1["matched_categories"]) == 1
    assert det1["synergy_bonus"] == 0.0

    # 2 categories: foundation_model + systems_efficiency -> synergy = 1.5
    _, det2 = score_arxiv_article("Transformer with KV Cache", "Abstract: Study of kv caches.", announce_type="new")
    assert len(det2["matched_categories"]) == 2
    assert det2["synergy_bonus"] == 1.5

    # 3 categories: foundation_model + systems_efficiency + safety_security -> synergy = 2.5
    _, det3 = score_arxiv_article("Transformer with KV Cache and Unlearning", "Abstract: Study.", announce_type="new")
    assert len(det3["matched_categories"]) == 3
    assert det3["synergy_bonus"] == 2.5


# ------------------------------------------------------------------------------
# 19. New frontier signals detection
# ------------------------------------------------------------------------------
def test_new_frontier_signals_detection():
    # KV cache in systems_efficiency
    _, d1 = score_arxiv_article("Optimizing KV Cache for Long Contexts", "Abstract: KV-cache eviction.")
    assert "KV cache" in d1["title_matches"].get("systems_efficiency", [])
    assert "long-context" in d1["title_matches"].get("foundation_model", [])

    # FlashAttention in systems_efficiency
    _, d2 = score_arxiv_article("Scaling with FlashAttention-2", "Abstract: Accelerated kernel.")
    assert "FlashAttention" in d2["title_matches"].get("systems_efficiency", [])

    # Context window in foundation_model
    _, d3 = score_arxiv_article("Extending Context Windows to 1M Tokens", "Abstract: Context window scaling.")
    assert "context window" in d3["title_matches"].get("foundation_model", [])

    # Mechanistic interpretability & representation engineering & activation patching
    _, d4 = score_arxiv_article("Mechanistic Interpretability of Transformers", "Abstract: Using representation engineering and activation patching.")
    assert "mechanistic interpretability" in d4["title_matches"].get("foundation_model", [])
    assert "representation engineering" in d4["abstract_matches"].get("foundation_model", [])
    assert "activation patching" in d4["abstract_matches"].get("foundation_model", [])

    # Probing in foundation_model
    _, d5 = score_arxiv_article("Probing Hidden Representations in LLMs", "Abstract: Linear probing analysis.")
    assert "probing" in d5["title_matches"].get("foundation_model", [])

    # Synthetic data & data curation in foundation_model
    _, d6 = score_arxiv_article("Synthetic Data for Foundation Model Pretraining", "Abstract: Automated data curation pipeline.")
    assert "synthetic data" in d6["title_matches"].get("foundation_model", [])
    assert "data curation" in d6["abstract_matches"].get("foundation_model", [])

    # DPO & Direct Preference Optimization in reasoning_agents
    _, d7 = score_arxiv_article("Beyond DPO: Direct Preference Optimization Frontiers", "Abstract: Aligning reasoning models.")
    assert "DPO" in d7["title_matches"].get("reasoning_agents", [])
    assert "Direct Preference Optimization" in d7["title_matches"].get("reasoning_agents", [])

    # ORPO & SimPO in reasoning_agents
    _, d8 = score_arxiv_article("Empirical Comparison of ORPO and SimPO", "Abstract: Direct preference methods.")
    assert "ORPO" in d8["title_matches"].get("reasoning_agents", [])
    assert "SimPO" in d8["title_matches"].get("reasoning_agents", [])

    # Unlearning in safety_security
    _, d9 = score_arxiv_article("Machine Unlearning in Large Language Models", "Abstract: Safe forgetting.")
    assert "unlearning" in d9["title_matches"].get("safety_security", [])


# ------------------------------------------------------------------------------
# 20. Regex precision and word boundaries
# ------------------------------------------------------------------------------
def test_regex_precision_and_boundaries():
    # Case sensitivity for DPO / ORPO / SimPO
    _, d_low = score_arxiv_article("The dpo of orpo was simple", "Abstract: Generic text with lowercase dpo.")
    assert "DPO" not in d_low["title_matches"].get("reasoning_agents", [])
    assert "ORPO" not in d_low["title_matches"].get("reasoning_agents", [])
    assert "SimPO" not in d_low["title_matches"].get("reasoning_agents", [])

    # KV cache requires cache
    _, d_kv = score_arxiv_article("A 50 kV transmission line power monitor", "Abstract: High voltage kv measurements.")
    assert "KV cache" not in d_kv["title_matches"].get("systems_efficiency", [])

    # Substring safety for RAG (e.g. fragment or courage)
    _, d_rag = score_arxiv_article("Fragmented Courage in Drama", "Abstract: Courageous fragments.")
    assert "RAG" not in d_rag["title_matches"].get("foundation_model", [])

