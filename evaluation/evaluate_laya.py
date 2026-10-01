"""
Phase 0.3: Laya vs Human Gold-Standard Evaluation Runner.

Loads the 150 human-labeled articles from evaluation/phase_0_2_labels.jsonl,
runs local Laya model inference using the bounded context strategy,
maps Laya's importance output to the human categories,
computes standard classification metrics, and generates prediction records and report.
"""

import json
import logging
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["MUST_KNOW", "INTERESTING", "NOISE"]

IMPORTANCE_TO_HUMAN_LABEL = {
    "high": "MUST_KNOW",
    "medium": "INTERESTING",
    "low": "NOISE",
    "ignore": "NOISE",
}

LAYA_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "ai_relevance": {
        "type": "choice",
        "instructions": "Determine whether this article has useful relationship to AI models, research, tools, engineering, companies, or AI industry developments.",
        "criteria": {
            "not_relevant": "The article has little or no useful relationship to AI models, research, tools, engineering, companies, or AI industry developments.",
            "weakly_relevant": "The article has some AI connection but limited value to an AI-focused intelligence system.",
            "moderately_relevant": "The article contains meaningful AI-related information useful for understanding the field.",
            "highly_relevant": "The article directly concerns important AI models, research, tools, engineering developments, major companies, or significant AI industry events.",
        },
    },
    "event_type": {
        "type": "choice",
        "instructions": "Classify the primary event type of this article.",
        "criteria": {
            "model_release": "New AI model release or weights announcement",
            "research": "AI research paper, methodology, or academic finding",
            "tool_release": "New software tool, library, framework, or developer platform",
            "company_news": "Corporate announcements, business deals, funding, or executive changes",
            "industry_news": "Broader AI ecosystem developments, policy, or market trends",
            "opinion_editorial": "Commentary, think piece, perspective, or analysis",
            "other": "Any other topic or uncategorized event",
        },
    },
    "novelty": {
        "type": "choice",
        "instructions": "Assess the level of novelty or breakthrough significance.",
        "criteria": {
            "breakthrough": "Pioneering breakthrough or major paradigm shift",
            "significant": "Substantial improvement, new capability, or notable milestone",
            "incremental": "Minor update, routine iteration, or maintenance release",
            "unknown": "Novelty cannot be determined or is not applicable",
        },
    },
    "importance": {
        "type": "choice",
        "instructions": "Assess the importance in terms of usefulness to an AI engineering/research news intelligence system.",
        "criteria": {
            "ignore": "Noise, completely irrelevant, or no technical value",
            "low": "Marginal interest, routine news, or niche topic",
            "medium": "Solid value, relevant to practitioners and researchers",
            "high": "Crucial update, must-read for AI engineers and researchers",
        },
    },
}


def build_laya_input(record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Construct bounded input state text from article record.

    Priority:
    1. Title + Source + First ~400 characters of Content (content_lead)
    2. Title + Source + Available Summary (summary)
    3. Title + Source (title_source)

    Returns:
        (input_state_text, input_strategy_name)
    """
    source = record.get("source", "Unknown Source")
    title = record.get("title", "Untitled")
    content = record.get("content") or ""
    summary = record.get("summary") or ""

    if content.strip():
        text = f"Source: {source}\nTitle: {title}\n\n{content[:400].strip()}"
        return text, "content_lead"
    elif summary.strip():
        text = f"Source: {source}\nTitle: {title}\n\n{summary[:400].strip()}"
        return text, "summary"
    else:
        text = f"Source: {source}\nTitle: {title}"
        return text, "title_source"


def map_laya_importance(importance_choice: str) -> str:
    """
    Map Laya raw importance choice ('high', 'medium', 'low', 'ignore')
    to canonical human ground-truth labels ('MUST_KNOW', 'INTERESTING', 'NOISE').
    """
    clean = importance_choice.strip().lower()
    if clean not in IMPORTANCE_TO_HUMAN_LABEL:
        raise ValueError(f"Unrecognized Laya importance choice: '{importance_choice}'")
    return IMPORTANCE_TO_HUMAN_LABEL[clean]


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute standard multi-class classification metrics using scikit-learn.
    """
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty dataset.")

    acc = accuracy_score(y_true, y_pred)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    incorrect = total - correct

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average="weighted", zero_division=0
    )

    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    per_class = {}
    for idx, cname in enumerate(CANONICAL_CLASSES):
        per_class[cname] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(support_class[idx]),
        }

    must_know_recall = per_class["MUST_KNOW"]["recall"]

    return {
        "total_count": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy": float(acc),
        "macro": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1": float(f1_macro),
        },
        "weighted": {
            "precision": float(p_weighted),
            "recall": float(r_weighted),
            "f1": float(f1_weighted),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "must_know_recall": float(must_know_recall),
    }


def run_evaluation(
    labels_file: str = "evaluation/phase_0_2_labels.jsonl",
    model_id: str = "convaiinnovations/laya",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Run full Phase 0.3 evaluation of Laya against human gold-standard labels.

    Returns:
        (predictions, metrics, metadata)
    """
    import laya

    path = Path(labels_file)
    if not path.exists():
        raise FileNotFoundError(f"Labeled dataset not found at {labels_file}")

    labeled_records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                labeled_records.append(json.loads(line_str))

    if len(labeled_records) != 150:
        logger.warning(f"Expected 150 records, got {len(labeled_records)}")

    # Model loading with timer
    t0_load = time.perf_counter()
    agent = laya.load(model_id)
    load_time_sec = time.perf_counter() - t0_load

    predictions: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    t0_eval = time.perf_counter()

    for rec in labeled_records:
        art_id = rec["article_id"]
        title = rec.get("title", "")
        source = rec.get("source", "")
        human_label = rec.get("human_label")

        if human_label not in CANONICAL_CLASSES:
            raise ValueError(f"Invalid or missing human label '{human_label}' for article ID {art_id}")

        state_text, input_strategy = build_laya_input(rec)

        t_start = time.perf_counter()
        res = agent.predict(state_text, LAYA_QUESTIONS)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(lat_ms)

        answers = res.get("answers", {})
        usage = res.get("usage", {})

        ans_rel = answers.get("ai_relevance", {})
        ans_evt = answers.get("event_type", {})
        ans_nov = answers.get("novelty", {})
        ans_imp = answers.get("importance", {})

        raw_importance = ans_imp.get("choice", "low")
        mapped_label = map_laya_importance(raw_importance)

        pred_entry = {
            "article_id": art_id,
            "title": title,
            "source": source,
            "human_label": human_label,
            "laya_label": mapped_label,
            "laya_importance": raw_importance,
            "laya_relevance": ans_rel.get("choice"),
            "laya_event": ans_evt.get("choice"),
            "laya_novelty": ans_nov.get("choice"),
            "probabilities": ans_imp.get("probabilities", {}),
            "confidence": ans_imp.get("confidence"),
            "answer_confidence": ans_imp.get("answer_confidence"),
            "input_strategy": input_strategy,
            "input_tokens": usage.get("input_tokens", 0),
            "latency_ms": round(lat_ms, 2),
            "device": str(getattr(agent, "device", "unknown")),
            "checkpoint": model_id,
        }

        predictions.append(pred_entry)
        y_true.append(human_label)
        y_pred.append(mapped_label)

    total_eval_sec = time.perf_counter() - t0_eval

    # Compute classification metrics
    metrics = compute_metrics(y_true, y_pred)

    # Process resource measurement
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    max_rss_mb = rusage.ru_maxrss / (1024.0 * 1024.0) if os.uname().sysname == "Darwin" else rusage.ru_maxrss / 1024.0

    metadata = {
        "model_id": model_id,
        "device": str(getattr(agent, "device", "unknown")),
        "load_time_sec": round(load_time_sec, 3),
        "total_eval_time_sec": round(total_eval_sec, 2),
        "average_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "median_latency_ms": round(float(np.median(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "min_latency_ms": round(float(np.min(latencies_ms)), 2),
        "max_latency_ms": round(float(np.max(latencies_ms)), 2),
        "max_rss_mb": round(max_rss_mb, 1),
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_strategies_count": {
            "content_lead": sum(1 for p in predictions if p["input_strategy"] == "content_lead"),
            "summary": sum(1 for p in predictions if p["input_strategy"] == "summary"),
            "title_source": sum(1 for p in predictions if p["input_strategy"] == "title_source"),
        },
    }

    return predictions, metrics, metadata


def write_predictions_jsonl(predictions: List[Dict[str, Any]], filepath: str) -> None:
    """Save predictions to JSON Lines file."""
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/phase_0_2_labels.jsonl"
    out_preds = sys.argv[2] if len(sys.argv) > 2 else "evaluation/phase_0_3_predictions.jsonl"

    print(f"Running Phase 0.3 evaluation against {in_path}...")
    preds, metrs, meta = run_evaluation(in_path)
    write_predictions_jsonl(preds, out_preds)
    print(f"Predictions written to {out_preds}.")
    print(f"Overall Accuracy: {metrs['accuracy'] * 100:.2f}% ({metrs['correct_count']}/{metrs['total_count']})")
    print(f"MUST_KNOW Recall: {metrs['must_know_recall'] * 100:.2f}%")
    print(f"Avg Latency: {meta['average_latency_ms']} ms | Total Time: {meta['total_eval_time_sec']} s")
