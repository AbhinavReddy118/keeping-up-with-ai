"""
Focused unit tests for Phase 0.2 evaluation sampling and labeling workflow.
"""

import json
import tempfile
from pathlib import Path
import pytest

from evaluation.sampling import (
    DEFAULT_SAMPLING_SEED,
    sample_evaluation_dataset,
    write_evaluation_jsonl,
)
from evaluation.labeler import (
    apply_label,
    get_labeling_progress,
    load_dataset,
    normalize_label,
    save_dataset,
    VALID_LABELS,
)


def test_sampling_determinism():
    """Verify that sampling with the same seed produces identical article lists."""
    ds1 = sample_evaluation_dataset(seed=DEFAULT_SAMPLING_SEED)
    ds2 = sample_evaluation_dataset(seed=DEFAULT_SAMPLING_SEED)

    assert len(ds1) == len(ds2) == 150
    ids1 = [r["article_id"] for r in ds1]
    ids2 = [r["article_id"] for r in ds2]
    assert ids1 == ids2


def test_sampling_schema_integrity():
    """Verify that all sampled records conform to the unpopulated human labeling schema."""
    dataset = sample_evaluation_dataset(seed=DEFAULT_SAMPLING_SEED)
    assert len(dataset) == 150

    sources = set()
    for record in dataset:
        assert isinstance(record["article_id"], int)
        assert isinstance(record["title"], str) and len(record["title"]) > 0
        assert isinstance(record["source"], str)
        assert "link" in record
        assert "stratum" in record
        assert record["human_label"] is None
        assert record["label_reason"] is None
        assert record["labeled_at"] is None
        assert record["rubric_version"] == "1.0"
        sources.add(record["source"])

    # All 7 database sources must be represented
    assert len(sources) == 7
    expected_sources = {
        "OpenAI News",
        "Hugging Face Blog",
        "Google DeepMind Blog",
        "arXiv CS.AI Papers",
        "Hacker News AI",
        "MIT Tech Review AI",
        "Microsoft Research",
    }
    assert sources == expected_sources


def test_dataset_jsonl_roundtrip():
    """Verify saving and loading dataset preserves all records and fields."""
    sample_records = [
        {
            "article_id": 101,
            "title": "Sample Article One",
            "source": "OpenAI News",
            "published_at": "2026-09-01T00:00:00Z",
            "link": "https://example.com/1",
            "summary": "Sample summary text",
            "content": None,
            "enrichment_status": "pending",
            "stratum": "OpenAI News (Recent)",
            "human_label": None,
            "label_reason": None,
            "labeled_at": None,
            "rubric_version": "1.0",
        },
        {
            "article_id": 102,
            "title": "Sample Article Two",
            "source": "Google DeepMind Blog",
            "published_at": "2026-09-02T00:00:00Z",
            "link": "https://example.com/2",
            "summary": None,
            "content": "Full content text here",
            "enrichment_status": "enriched",
            "stratum": "Google DeepMind Blog (2026)",
            "human_label": None,
            "label_reason": None,
            "labeled_at": None,
            "rubric_version": "1.0",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_labels.jsonl"
        write_evaluation_jsonl(sample_records, str(test_file))

        loaded = load_dataset(str(test_file))
        assert len(loaded) == 2
        assert loaded[0]["article_id"] == 101
        assert loaded[1]["article_id"] == 102
        assert loaded[0]["title"] == "Sample Article One"

        # Test save_dataset update
        loaded[0]["human_label"] = "MUST_KNOW"
        loaded[0]["label_reason"] = "Frontier release"
        save_dataset(str(test_file), loaded)

        reloaded = load_dataset(str(test_file))
        assert reloaded[0]["human_label"] == "MUST_KNOW"
        assert reloaded[0]["label_reason"] == "Frontier release"


def test_normalize_label():
    """Verify label shortcuts and normalization."""
    assert normalize_label("1") == "MUST_KNOW"
    assert normalize_label("2") == "INTERESTING"
    assert normalize_label("3") == "NOISE"
    assert normalize_label("m") == "MUST_KNOW"
    assert normalize_label("i") == "INTERESTING"
    assert normalize_label("n") == "NOISE"
    assert normalize_label("must_know") == "MUST_KNOW"
    assert normalize_label("INTERESTING") == "INTERESTING"
    assert normalize_label("noise") == "NOISE"
    assert normalize_label("invalid") is None


def test_apply_label_success():
    """Verify applying a valid label updates the record with reason and timestamp."""
    rec = {
        "article_id": 200,
        "human_label": None,
        "label_reason": None,
        "labeled_at": None,
        "rubric_version": "1.0",
    }

    updated = apply_label(rec, label="1", reason="New frontier model launch")
    assert updated["human_label"] == "MUST_KNOW"
    assert updated["label_reason"] == "New frontier model launch"
    assert updated["labeled_at"] is not None
    assert updated["rubric_version"] == "1.0"


def test_apply_label_invalid_label_raises():
    """Verify invalid labels raise ValueError."""
    rec = {"article_id": 200, "human_label": None}
    with pytest.raises(ValueError, match="Invalid label"):
        apply_label(rec, label="invalid_choice", reason="Some valid reason")


def test_apply_label_empty_reason_raises():
    """Verify empty or whitespace-only reason raises ValueError."""
    rec = {"article_id": 200, "human_label": None}
    with pytest.raises(ValueError, match="non-empty human reasoning"):
        apply_label(rec, label="MUST_KNOW", reason="   ")


def test_get_labeling_progress():
    """Verify progress tracking calculation."""
    dataset = [
        {"article_id": 1, "human_label": "MUST_KNOW"},
        {"article_id": 2, "human_label": "INTERESTING"},
        {"article_id": 3, "human_label": "NOISE"},
        {"article_id": 4, "human_label": None},
        {"article_id": 5, "human_label": None},
    ]

    progress = get_labeling_progress(dataset)
    assert progress["total"] == 5
    assert progress["labeled"] == 3
    assert progress["unlabeled"] == 2
    assert progress["percent_complete"] == 60.0
    assert progress["must_know"] == 1
    assert progress["interesting"] == 1
    assert progress["noise"] == 1
