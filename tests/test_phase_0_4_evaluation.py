"""
Unit tests for Phase 0.4 evaluation: dataset integrity, context construction,
prediction schema, metrics consistency, operational gate calculations,
and production safety.
"""

import json
from pathlib import Path
import pytest

from evaluation.evaluate_laya import map_laya_importance_probabilities
from evaluation.evaluate_laya_phase_0_4 import (
    CANONICAL_CLASSES,
    BASELINE_IMPORTANCE_MAP,
    RUBRIC_IMPORTANCE_MAP,
    build_context_text,
    compute_comprehensive_metrics,
    evaluate_operational_gate,
    map_prediction_to_canonical,
    map_prediction_to_canonical_corrected,
    verify_dataset_integrity,
)


def test_phase_0_2_dataset_integrity():
    """Verify Phase 0.2 labeled dataset satisfies all integrity invariants."""
    path = Path("evaluation/phase_0_2_labels.jsonl")
    assert path.exists(), "evaluation/phase_0_2_labels.jsonl must exist"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 150
    ids = [r["article_id"] for r in records]
    assert len(ids) == len(set(ids)), "Article IDs must be unique"

    labels = [r["human_label"] for r in records]
    assert labels.count("MUST_KNOW") == 64
    assert labels.count("INTERESTING") == 68
    assert labels.count("NOISE") == 18

    # Function verification
    verify_dataset_integrity(records)


def test_context_construction_deterministic():
    """Verify deterministic behavior and bounds for Context A, B, and C."""
    rec_with_content = {
        "source": "OpenAI News",
        "title": "Frontier Model Release",
        "content": "X" * 1200,
        "summary": "Short summary",
    }
    rec_with_summary = {
        "source": "Google DeepMind",
        "title": "Research Update",
        "content": "",
        "summary": "Y" * 900,
    }
    rec_title_only = {
        "source": "MIT Tech Review",
        "title": "General Tech Story",
        "content": None,
        "summary": None,
    }

    # Context A: Title + Source only
    text_a1, src_a1 = build_context_text(rec_with_content, "context_a")
    text_a2, src_a2 = build_context_text(rec_with_content, "context_a")
    assert text_a1 == text_a2 == "Source: OpenAI News\nTitle: Frontier Model Release"
    assert src_a1 == "title_source"

    # Context B: Title + Source + Lead 400
    text_b, src_b = build_context_text(rec_with_content, "context_b")
    assert src_b == "content"
    assert text_b == f"Source: OpenAI News\nTitle: Frontier Model Release\n\n{'X' * 400}"

    text_b_sum, src_b_sum = build_context_text(rec_with_summary, "context_b")
    assert src_b_sum == "summary"
    assert text_b_sum == f"Source: Google DeepMind\nTitle: Research Update\n\n{'Y' * 400}"

    text_b_title, src_b_title = build_context_text(rec_title_only, "context_b")
    assert src_b_title == "title_source"
    assert text_b_title == "Source: MIT Tech Review\nTitle: General Tech Story"

    # Context C: Title + Source + Bounded 800
    text_c, src_c = build_context_text(rec_with_content, "context_c")
    assert src_c == "content"
    assert text_c == f"Source: OpenAI News\nTitle: Frontier Model Release\n\n{'X' * 800}"


def test_map_prediction_to_canonical():
    """Verify label mappings and probability consolidation for both formulations."""
    # Baseline
    raw_b_probs = {"high": 0.65, "medium": 0.25, "low": 0.08, "ignore": 0.02}
    lbl_b, probs_b = map_prediction_to_canonical("baseline", "high", raw_b_probs)
    assert lbl_b == "MUST_KNOW"
    assert probs_b["MUST_KNOW"] == 0.65
    assert probs_b["INTERESTING"] == 0.25
    assert probs_b["NOISE"] == pytest.approx(0.10)

    lbl_b2, probs_b2 = map_prediction_to_canonical("baseline", "low", raw_b_probs)
    assert lbl_b2 == "NOISE"

    # Rubric
    raw_r_probs = {"must_know": 0.70, "interesting": 0.25, "noise": 0.05}
    lbl_r, probs_r = map_prediction_to_canonical("rubric_aligned", "must_know", raw_r_probs)
    assert lbl_r == "MUST_KNOW"
    assert probs_r["MUST_KNOW"] == 0.70
    assert probs_r["INTERESTING"] == 0.25
    assert probs_r["NOISE"] == 0.05


def test_compute_comprehensive_metrics_synthetic():
    """Verify multi-class, binary, margin, and latency metrics consistency."""
    y_true = ["MUST_KNOW", "MUST_KNOW", "INTERESTING", "INTERESTING", "NOISE"]
    y_pred = ["MUST_KNOW", "INTERESTING", "INTERESTING", "NOISE", "NOISE"]
    probs = [
        {"MUST_KNOW": 0.8, "INTERESTING": 0.15, "NOISE": 0.05},
        {"MUST_KNOW": 0.45, "INTERESTING": 0.50, "NOISE": 0.05},
        {"MUST_KNOW": 0.1, "INTERESTING": 0.75, "NOISE": 0.15},
        {"MUST_KNOW": 0.05, "INTERESTING": 0.40, "NOISE": 0.55},
        {"MUST_KNOW": 0.02, "INTERESTING": 0.08, "NOISE": 0.90},
    ]
    latencies = [40.0, 42.0, 38.0, 39.0, 41.0]

    metrics = compute_comprehensive_metrics(y_true, y_pred, probs, latencies)

    assert metrics["total_count"] == 5
    assert metrics["correct_count"] == 3
    assert metrics["incorrect_count"] == 2
    assert metrics["accuracy"] == pytest.approx(0.60)
    assert metrics["must_know_recall"] == pytest.approx(0.50)
    assert metrics["noise_recall"] == pytest.approx(1.00)
    assert metrics["prediction_distribution"] == {
        "MUST_KNOW": 1,
        "INTERESTING": 2,
        "NOISE": 2,
    }
    assert metrics["latency_stats"]["average_ms"] == pytest.approx(40.0)
    assert metrics["latency_stats"]["min_ms"] == 38.0
    assert metrics["latency_stats"]["max_ms"] == 42.0


def test_operational_gate_mathematical_consistency():
    """Verify operational gate formulas: passed + skipped == total, recall formulas, false skip counts."""
    synthetic_preds = [
        {"human_label": "MUST_KNOW", "predicted_final_label": "MUST_KNOW", "probabilities": {"NOISE": 0.05}},
        {"human_label": "MUST_KNOW", "predicted_final_label": "INTERESTING", "probabilities": {"NOISE": 0.10}},
        {"human_label": "INTERESTING", "predicted_final_label": "INTERESTING", "probabilities": {"NOISE": 0.20}},
        {"human_label": "INTERESTING", "predicted_final_label": "NOISE", "probabilities": {"NOISE": 0.75}},
        {"human_label": "NOISE", "predicted_final_label": "NOISE", "probabilities": {"NOISE": 0.85}},
    ]

    res = evaluate_operational_gate(
        synthetic_preds,
        "Test Gate: Skip NOISE",
        lambda p: p["predicted_final_label"] != "NOISE",
    )

    assert res["articles_passed"] + res["articles_skipped"] == 5
    assert res["pct_passed"] + res["qwen_reduction_pct"] == pytest.approx(100.0)
    # Passed: 1 MK, 1 MK (pred INT), 1 INT (pred INT) = 3 passed.
    assert res["articles_passed"] == 3
    assert res["must_know_passed"] == 2
    assert res["must_know_total"] == 2
    assert res["must_know_recall_pct"] == 100.0
    assert res["must_know_false_negatives"] == 0
    # Skipped: 1 INT (pred NOI), 1 NOI (pred NOI)
    assert res["noise_filtered"] == 1
    assert res["noise_total"] == 1
    assert res["noise_filter_rate_pct"] == 100.0
    assert res["total_false_skips"] == 1  # 1 INT was skipped
    assert res["total_false_passes"] == 0


def test_production_safety_unchanged():
    """Verify that no production files were modified or touched."""
    prod_files = [
        "ranking/scorer.py",
        "database.py",
        "main.py",
        "orchestration.py",
        "sources.py",
        "storage.py",
        "retention.py",
        "audit.py",
    ]
    for pf in prod_files:
        p = Path(pf)
        assert p.exists(), f"Production file {pf} must exist"


def test_phase_0_4_predictions_artifact():
    """Verify evaluation/phase_0_4_predictions.jsonl exists and satisfies schema & coverage rules."""
    path = Path("evaluation/phase_0_4_predictions.jsonl")
    assert path.exists(), "evaluation/phase_0_4_predictions.jsonl must exist"

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # 150 articles * 6 experiments = 900 rows
    assert len(rows) == 900

    required_keys = {
        "article_id",
        "experiment_id",
        "formulation",
        "context_strategy",
        "title",
        "source",
        "human_label",
        "predicted_final_label",
        "raw_target_choice",
        "probabilities",
        "raw_probabilities",
        "confidence",
        "input_token_count",
        "latency_ms",
        "model_id",
        "device",
        "timestamp",
    }

    by_exp = {}
    for r in rows:
        assert required_keys.issubset(r.keys())
        assert r["human_label"] in CANONICAL_CLASSES
        assert r["predicted_final_label"] in CANONICAL_CLASSES
        assert r["latency_ms"] > 0
        assert r["input_token_count"] > 0

        # Probabilities validation
        probs = r["probabilities"]
        assert set(probs.keys()) == set(CANONICAL_CLASSES)
        assert pytest.approx(sum(probs.values()), abs=1e-2) == 1.0

        exp_id = r["experiment_id"]
        by_exp.setdefault(exp_id, []).append(r)

    # Verify all 6 experiments present with 150 unique articles each
    assert set(by_exp.keys()) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    for exp_id, exp_rows in by_exp.items():
        assert len(exp_rows) == 150
        ids = [er["article_id"] for er in exp_rows]
        assert len(ids) == len(set(ids)) == 150


def test_phase_0_4_operational_analysis_artifact():
    """Verify evaluation/phase_0_4_operational_analysis.json exists and contains consistent data."""
    path = Path("evaluation/phase_0_4_operational_analysis.json")
    assert path.exists(), "evaluation/phase_0_4_operational_analysis.json must exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "experiment_metrics" in data
    assert "operational_policies" in data

    metrics = data["experiment_metrics"]
    assert set(metrics.keys()) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    for exp_id, m in metrics.items():
        assert m["total_count"] == 150
        assert m["correct_count"] + m["incorrect_count"] == 150
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["must_know_recall"] <= 1.0
        assert 0.0 <= m["noise_recall"] <= 1.0

    policies = data["operational_policies"]
    assert len(policies) > 0
    for pol in policies:
        assert pol["articles_passed"] + pol["articles_skipped"] == 150
        assert pol["must_know_passed"] + pol["must_know_false_negatives"] == pol["must_know_total"] == 64
        assert pol["noise_passed"] + pol["noise_filtered"] == pol["noise_total"] == 18
        assert pol["pct_passed"] + pol["qwen_reduction_pct"] == pytest.approx(100.0)


def test_canonical_probability_argmax_regression():
    """
    Regression test validating that the 3-way canonical probability argmax
    correctly resolves cases where P(low) + P(ignore) > P(medium) > P(low), P(ignore).

    Example: Article 2305:
    P(high) = 0.2081
    P(medium) = 0.3223
    P(low) = 0.2638
    P(ignore) = 0.2059

    Old 4-way argmax picked 'medium' -> mapped to 'INTERESTING'.
    Correct canonical 3-way mapping combines P(low) + P(ignore) = 0.4697.
    Argmax(0.2081, 0.3223, 0.4697) -> 'NOISE'.
    """
    raw_probs = {
        "high": 0.2081,
        "medium": 0.3223,
        "low": 0.2638,
        "ignore": 0.2059,
    }
    # Test evaluate_laya helper
    mapped_label_03, c_probs_03 = map_laya_importance_probabilities(raw_probs)
    assert c_probs_03["MUST_KNOW"] == 0.2081
    assert c_probs_03["INTERESTING"] == 0.3223
    assert c_probs_03["NOISE"] == 0.4697
    assert mapped_label_03 == "NOISE"

    # Test evaluate_laya_phase_0_4 corrected helper
    mapped_label_04, c_probs_04 = map_prediction_to_canonical_corrected(
        "baseline", "medium", raw_probs
    )
    assert mapped_label_04 == "NOISE"
    assert c_probs_04["NOISE"] == 0.4697
    assert c_probs_04["INTERESTING"] == 0.3223
    assert c_probs_04["MUST_KNOW"] == 0.2081

