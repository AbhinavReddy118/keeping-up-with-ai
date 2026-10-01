"""
Unit tests for Phase 0.3 Laya evaluation logic, dataset integrity, and metric computations.
"""

import json
from pathlib import Path
import pytest

from evaluation.evaluate_laya import (
    CANONICAL_CLASSES,
    IMPORTANCE_TO_HUMAN_LABEL,
    build_laya_input,
    compute_metrics,
    map_laya_importance,
)


def test_labeled_dataset_count_and_distribution():
    """Verify that phase_0_2_labels.jsonl has exactly 150 rows with the expected distribution."""
    path = Path("evaluation/phase_0_2_labels.jsonl")
    assert path.exists(), "phase_0_2_labels.jsonl must exist"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 150, f"Expected 150 articles, got {len(records)}"

    # Verify no duplicate IDs
    ids = [r["article_id"] for r in records]
    assert len(ids) == len(set(ids)), "Article IDs must be unique"

    # Verify distribution
    labels = [r["human_label"] for r in records]
    assert labels.count("MUST_KNOW") == 64
    assert labels.count("INTERESTING") == 68
    assert labels.count("NOISE") == 18

    # Verify all labels belong to CANONICAL_CLASSES
    for lbl in labels:
        assert lbl in CANONICAL_CLASSES


def test_map_laya_importance():
    """Verify that all raw Laya importance choices map correctly to human classes."""
    assert map_laya_importance("high") == "MUST_KNOW"
    assert map_laya_importance("medium") == "INTERESTING"
    assert map_laya_importance("low") == "NOISE"
    assert map_laya_importance("ignore") == "NOISE"

    # Case-insensitivity and whitespace handling
    assert map_laya_importance(" HIGH ") == "MUST_KNOW"
    assert map_laya_importance("Medium") == "INTERESTING"

    # Invalid choice raises ValueError
    with pytest.raises(ValueError, match="Unrecognized Laya importance"):
        map_laya_importance("unsupported_choice")


def test_build_laya_input_strategies():
    """Verify that input construction follows the preferred hierarchy without exceeding context."""
    # 1. Content present -> content_lead
    rec_content = {
        "source": "OpenAI News",
        "title": "Sample Title",
        "content": "A" * 600,
        "summary": "Short summary",
    }
    text, strat = build_laya_input(rec_content)
    assert strat == "content_lead"
    assert text.startswith("Source: OpenAI News\nTitle: Sample Title\n\n")
    # Verify content lead bounded to ~400 chars
    assert len(text.split("\n\n")[1]) == 400

    # 2. Content absent, summary present -> summary
    rec_summary = {
        "source": "Hugging Face Blog",
        "title": "HF Title",
        "content": None,
        "summary": "Summary text here",
    }
    text, strat = build_laya_input(rec_summary)
    assert strat == "summary"
    assert "Summary text here" in text

    # 3. Content and summary absent -> title_source
    rec_title_only = {
        "source": "Google DeepMind",
        "title": "Only Title",
        "content": "",
        "summary": "",
    }
    text, strat = build_laya_input(rec_title_only)
    assert strat == "title_source"
    assert text == "Source: Google DeepMind\nTitle: Only Title"


def test_compute_metrics_synthetic():
    """Verify standard metric calculations on a controlled synthetic confusion example."""
    # 3 classes: MUST_KNOW, INTERESTING, NOISE
    # True:  MK, MK, INT, INT, NOI
    # Pred:  MK, INT, INT, NOI, NOI
    y_true = ["MUST_KNOW", "MUST_KNOW", "INTERESTING", "INTERESTING", "NOISE"]
    y_pred = ["MUST_KNOW", "INTERESTING", "INTERESTING", "NOISE", "NOISE"]

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["total_count"] == 5
    assert metrics["correct_count"] == 3
    assert metrics["incorrect_count"] == 2
    assert metrics["accuracy"] == pytest.approx(0.60)

    # MUST_KNOW: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=2*(1*0.5)/(1+0.5)=0.6667
    assert metrics["per_class"]["MUST_KNOW"]["precision"] == pytest.approx(1.0)
    assert metrics["per_class"]["MUST_KNOW"]["recall"] == pytest.approx(0.5)
    assert metrics["must_know_recall"] == pytest.approx(0.5)

    # Confusion matrix ordering: [MUST_KNOW, INTERESTING, NOISE]
    cm = metrics["confusion_matrix"]
    # Row 0 (MK): [1, 1, 0]
    # Row 1 (INT): [0, 1, 1]
    # Row 2 (NOI): [0, 0, 1]
    assert cm == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_predictions_jsonl_schema():
    """Verify that phase_0_3_predictions.jsonl exists and contains valid rows."""
    path = Path("evaluation/phase_0_3_predictions.jsonl")
    assert path.exists(), "Predictions file must be generated"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 150

    expected_keys = {
        "article_id",
        "title",
        "source",
        "human_label",
        "laya_label",
        "laya_importance",
        "laya_relevance",
        "laya_event",
        "laya_novelty",
        "probabilities",
        "input_strategy",
        "input_tokens",
        "latency_ms",
        "device",
        "checkpoint",
    }

    for r in records:
        assert expected_keys.issubset(r.keys())
        assert r["human_label"] in CANONICAL_CLASSES
        assert r["laya_label"] in CANONICAL_CLASSES
        assert r["input_strategy"] in {"content_lead", "summary", "title_source"}
        assert r["latency_ms"] > 0
        assert isinstance(r["probabilities"], dict)
