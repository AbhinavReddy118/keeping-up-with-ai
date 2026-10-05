"""
Phase 0.4: Laya Decision-Formulation, Context Ablation, Operational Gate, and Failure Analysis.

This module evaluates Laya under controlled context strategies (A, B, C)
and decision formulations (Baseline vs. Rubric-Aligned Structured),
records detailed predictions, computes classification and operational gate metrics,
and saves artifacts for Phase 0.4 analysis.

READ-ONLY evaluation against evaluation/phase_0_2_labels.jsonl.
DO NOT modify production code or existing human labels.
"""

import json
import logging
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

if TYPE_CHECKING:
    import laya

logger = logging.getLogger(__name__)

# Zero-division fallback parameter for sklearn metrics
ZERO_DIVISION: Any = 0

CANONICAL_CLASSES = ["MUST_KNOW", "INTERESTING", "NOISE"]

# Baseline Phase 0.3 Questions & Mapping
BASELINE_IMPORTANCE_MAP = {
    "high": "MUST_KNOW",
    "medium": "INTERESTING",
    "low": "NOISE",
    "ignore": "NOISE",
}

BASELINE_QUESTIONS: Dict[str, Dict[str, Any]] = {
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

# Rubric-Aligned Structured Formulation (Formulation 2)
# Decomposes the human decision procedure from docs/importance_rubric.md
RUBRIC_IMPORTANCE_MAP = {
    "must_know": "MUST_KNOW",
    "interesting": "INTERESTING",
    "noise": "NOISE",
}

RUBRIC_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "ai_centrality": {
        "type": "choice",
        "instructions": "Assess whether artificial intelligence is the primary core subject of this article or merely a tangential mention.",
        "criteria": {
            "core_focus": "Directly concerns AI models, architectures, training, data, algorithms, hardware, or AI systems engineering.",
            "secondary_focus": "Focuses on general software, tech, or business with significant AI components or applications.",
            "tangential_or_unrelated": "Only mentions AI in passing, general technology news with no AI technical development, or unrelated.",
        },
    },
    "new_development": {
        "type": "choice",
        "instructions": "Determine whether this article reports a concrete, verifiable new development or artifact versus general commentary.",
        "criteria": {
            "concrete_artifact_or_event": "Announces a specific newly released model, code repository, paper, benchmark, security disclosure, or measurable production deployment.",
            "ongoing_evolution": "Describes ongoing engineering work, incremental updates, corporate progress, or industry trend overview.",
            "speculative_or_editorial": "Presents opinion, philosophical debate, hypothetical speculation, general advice, or promotional marketing fluff with no technical deliverables.",
        },
    },
    "event_type": {
        "type": "choice",
        "instructions": "Identify the primary nature of the event or content described in the article.",
        "criteria": {
            "model_release": "Release or major update of a foundation model, frontier architecture, or open model weights.",
            "developer_tool_or_infra": "Release or update of developer tooling, libraries, frameworks, compilers, or compute infrastructure.",
            "research_paper": "Academic research paper, theoretical breakthrough, empirical benchmark study, or algorithmic analysis.",
            "security_or_vulnerability": "Vulnerability disclosure, exploit, safety incident, or systemic security risk.",
            "industry_or_business": "Corporate partnerships, funding rounds, talent movements, legal or regulatory policy developments.",
            "case_study_or_tutorial": "Technical deployment post-mortem, implementation case study, or in-depth technical tutorial.",
            "opinion_or_commentary": "General think piece, interview, editorial, listicle, or market commentary.",
        },
    },
    "blast_radius": {
        "type": "choice",
        "instructions": "Assess the potential impact scope or blast radius of this development across the AI industry.",
        "criteria": {
            "industry_wide_disruption": "Directly alters standard engineering workflows, introduces new state-of-the-art frontier capabilities, or forces teams to re-evaluate roadmaps.",
            "domain_specific_utility": "Offers solid utility, empirical progress, or valuable improvements within a specific subfield, niche domain, or library ecosystem.",
            "minimal_or_localized": "Routine iteration, localized maintenance, minor bugfix, narrow case study, or minimal external relevance.",
        },
    },
    "practitioner_awareness_value": {
        "type": "choice",
        "instructions": "Determine the practical awareness value of this article for an active AI engineer or researcher.",
        "criteria": {
            "essential_attention": "Materially important knowledge; missing this leaves a practitioner technologically disadvantaged or operationally unaware.",
            "informative_reference": "Technically sound and valuable context to file away for reference, but does not demand immediate action or workflow change.",
            "negligible_value": "Contains zero actionable technical insight, trivial maintenance details, or low-information content that can be safely discarded.",
        },
    },
    "final_importance": {
        "type": "choice",
        "instructions": "Classify the overall editorial importance of this article according to practitioner briefing standards.",
        "criteria": {
            "must_know": "Crucial frontier release, major paradigm shift, industry-altering tool, or critical incident requiring immediate practitioner attention.",
            "interesting": "Substantive technical development, valid research paper, useful library, or instructive case study worth passive awareness.",
            "noise": "Generic speculation, marketing fluff, routine patch, uninformative mention, or content with no actionable technical substance.",
        },
    },
}


def verify_dataset_integrity(records: List[Dict[str, Any]]) -> None:
    """
    Verify Phase 0.2 dataset integrity rules:
    - exactly 150 rows
    - exactly 150 unique article IDs
    - no missing human labels
    - distribution: MUST_KNOW=64, INTERESTING=68, NOISE=18
    """
    if len(records) != 150:
        raise ValueError(f"Dataset integrity check failed: Expected 150 rows, got {len(records)}")

    ids = [r.get("article_id") for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset integrity check failed: Duplicate article IDs found.")

    labels = [r.get("human_label") for r in records]
    for lbl in labels:
        if lbl not in CANONICAL_CLASSES:
            raise ValueError(f"Dataset integrity check failed: Invalid human label '{lbl}'.")

    c_mk = labels.count("MUST_KNOW")
    c_int = labels.count("INTERESTING")
    c_noi = labels.count("NOISE")

    if (c_mk, c_int, c_noi) != (64, 68, 18):
        raise ValueError(
            f"Dataset integrity check failed: Expected (64, 68, 18), got ({c_mk}, {c_int}, {c_noi})"
        )


def build_context_text(record: Dict[str, Any], strategy: str) -> Tuple[str, str]:
    """
    Construct bounded input state text from article record.

    Strategies:
    - context_a: title + source
    - context_b: title + source + first 400 characters of available article text
    - context_c: title + source + first bounded excerpt ~800 characters

    Precedence for article text:
    1. Prefer enriched content (record.get("content"))
    2. Otherwise use summary (record.get("summary"))
    3. Otherwise use title + source only

    Returns:
        (state_text, text_source_type)
    """
    source = (record.get("source") or "Unknown Source").strip()
    title = (record.get("title") or "Untitled").strip()
    content = (record.get("content") or "").strip()
    summary = (record.get("summary") or "").strip()

    if strategy == "context_a":
        return f"Source: {source}\nTitle: {title}", "title_source"

    body = ""
    source_type = "title_source"
    if content:
        body = content
        source_type = "content"
    elif summary:
        body = summary
        source_type = "summary"

    if not body:
        return f"Source: {source}\nTitle: {title}", "title_source"

    if strategy == "context_b":
        excerpt = body[:400].strip()
        return f"Source: {source}\nTitle: {title}\n\n{excerpt}", source_type
    elif strategy == "context_c":
        excerpt = body[:800].strip()
        return f"Source: {source}\nTitle: {title}\n\n{excerpt}", source_type
    else:
        raise ValueError(f"Unknown context strategy: '{strategy}'")


def map_prediction_to_canonical(
    formulation: str,
    raw_choice: str,
    raw_probabilities: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Map raw model choices and probabilities to canonical classes:
    MUST_KNOW, INTERESTING, NOISE.
    """
    clean_choice = raw_choice.strip().lower()

    if formulation == "baseline":
        if clean_choice not in BASELINE_IMPORTANCE_MAP:
            raise ValueError(f"Unrecognized baseline importance choice: '{raw_choice}'")
        mapped_label = BASELINE_IMPORTANCE_MAP[clean_choice]

        # Consolidate 4-way probabilities into 3 canonical classes
        p_mk = float(raw_probabilities.get("high", 0.0))
        p_int = float(raw_probabilities.get("medium", 0.0))
        p_noi = float(raw_probabilities.get("low", 0.0)) + float(raw_probabilities.get("ignore", 0.0))

        canonical_probs = {
            "MUST_KNOW": round(p_mk, 4),
            "INTERESTING": round(p_int, 4),
            "NOISE": round(p_noi, 4),
        }
        return mapped_label, canonical_probs

    elif formulation == "rubric_aligned":
        if clean_choice not in RUBRIC_IMPORTANCE_MAP:
            raise ValueError(f"Unrecognized rubric importance choice: '{raw_choice}'")
        mapped_label = RUBRIC_IMPORTANCE_MAP[clean_choice]

        canonical_probs = {
            "MUST_KNOW": round(float(raw_probabilities.get("must_know", 0.0)), 4),
            "INTERESTING": round(float(raw_probabilities.get("interesting", 0.0)), 4),
            "NOISE": round(float(raw_probabilities.get("noise", 0.0)), 4),
        }
        return mapped_label, canonical_probs

    else:
        raise ValueError(f"Unknown formulation: '{formulation}'")


def map_prediction_to_canonical_corrected(
    formulation: str,
    raw_choice: str,
    raw_probabilities: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Corrected canonical probability mapping using 3-way probability argmax:
    MUST_KNOW   = P(high)
    INTERESTING = P(medium)
    NOISE       = P(low) + P(ignore)
    predicted_label = argmax(canonical_probs)

    For rubric_aligned, the raw output already has 3 classes:
    must_know, interesting, noise, which map directly to MUST_KNOW, INTERESTING, NOISE.
    """
    if formulation == "baseline":
        p_mk = round(float(raw_probabilities.get("high", 0.0)), 4)
        p_int = round(float(raw_probabilities.get("medium", 0.0)), 4)
        p_noi = round(
            float(raw_probabilities.get("low", 0.0)) + float(raw_probabilities.get("ignore", 0.0)),
            4,
        )
        canonical_probs = {
            "MUST_KNOW": p_mk,
            "INTERESTING": p_int,
            "NOISE": p_noi,
        }
        mapped_label = max(canonical_probs.items(), key=lambda x: x[1])[0]
        return mapped_label, canonical_probs

    elif formulation == "rubric_aligned":
        clean_choice = raw_choice.strip().lower()
        if clean_choice not in RUBRIC_IMPORTANCE_MAP:
            raise ValueError(f"Unrecognized rubric importance choice: '{raw_choice}'")
        mapped_label = RUBRIC_IMPORTANCE_MAP[clean_choice]
        canonical_probs = {
            "MUST_KNOW": round(float(raw_probabilities.get("must_know", 0.0)), 4),
            "INTERESTING": round(float(raw_probabilities.get("interesting", 0.0)), 4),
            "NOISE": round(float(raw_probabilities.get("noise", 0.0)), 4),
        }
        return mapped_label, canonical_probs
    else:
        raise ValueError(f"Unknown formulation: '{formulation}'")


def compute_comprehensive_metrics(
    y_true: List[str],
    y_pred: List[str],
    probabilities_list: List[Dict[str, float]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """
    Compute multi-class metrics, per-class metrics, probability margin metrics,
    and latency statistics.
    """
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty dataset.")

    acc = accuracy_score(y_true, y_pred)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    incorrect = total - correct

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=ZERO_DIVISION
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average="weighted", zero_division=ZERO_DIVISION
    )
    res_class = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=ZERO_DIVISION
    )
    p_class: np.ndarray = np.asarray(res_class[0])
    r_class: np.ndarray = np.asarray(res_class[1])
    f1_class: np.ndarray = np.asarray(res_class[2])
    support_class: np.ndarray = np.asarray(res_class[3])

    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    per_class = {}
    for idx, cname in enumerate(CANONICAL_CLASSES):
        per_class[cname] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(support_class[idx]),
        }

    # Prediction distribution
    pred_counts = {c: y_pred.count(c) for c in CANONICAL_CLASSES}

    # Margins and confidence
    margins = []
    confidences = []
    low_margin_count_15 = 0
    low_margin_count_10 = 0

    for probs in probabilities_list:
        p_vals = sorted(probs.values(), reverse=True)
        top1 = p_vals[0] if len(p_vals) > 0 else 0.0
        top2 = p_vals[1] if len(p_vals) > 1 else 0.0
        margin = top1 - top2
        margins.append(margin)
        confidences.append(top1)
        if margin < 0.15:
            low_margin_count_15 += 1
        if margin < 0.10:
            low_margin_count_10 += 1

    # Binary metrics:
    # 1. MUST_KNOW vs NOT_MUST_KNOW
    y_true_mk = [1 if y == "MUST_KNOW" else 0 for y in y_true]
    y_pred_mk = [1 if y == "MUST_KNOW" else 0 for y in y_pred]
    mk_binary_p = precision_score(y_true_mk, y_pred_mk, zero_division=ZERO_DIVISION)
    mk_binary_r = recall_score(y_true_mk, y_pred_mk, zero_division=ZERO_DIVISION)
    mk_binary_f1 = f1_score(y_true_mk, y_pred_mk, zero_division=ZERO_DIVISION)

    # 2. NOISE vs NOT_NOISE
    y_true_noi = [1 if y == "NOISE" else 0 for y in y_true]
    y_pred_noi = [1 if y == "NOISE" else 0 for y in y_pred]
    noi_binary_p = precision_score(y_true_noi, y_pred_noi, zero_division=ZERO_DIVISION)
    noi_binary_r = recall_score(y_true_noi, y_pred_noi, zero_division=ZERO_DIVISION)
    noi_binary_f1 = f1_score(y_true_noi, y_pred_noi, zero_division=ZERO_DIVISION)

    return {
        "total_count": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy": round(float(acc), 4),
        "macro": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1": round(float(f1_macro), 4),
        },
        "weighted": {
            "precision": round(float(p_weighted), 4),
            "recall": round(float(r_weighted), 4),
            "f1": round(float(f1_weighted), 4),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": pred_counts,
        "must_know_recall": round(per_class["MUST_KNOW"]["recall"], 4),
        "noise_recall": round(per_class["NOISE"]["recall"], 4),
        "interesting_recall": round(per_class["INTERESTING"]["recall"], 4),
        "latency_stats": {
            "average_ms": round(float(np.mean(latencies_ms)), 2),
            "median_ms": round(float(np.median(latencies_ms)), 2),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "min_ms": round(float(np.min(latencies_ms)), 2),
            "max_ms": round(float(np.max(latencies_ms)), 2),
        },
        "margin_stats": {
            "average_confidence": round(float(np.mean(confidences)), 4),
            "average_margin": round(float(np.mean(margins)), 4),
            "low_margin_count_15": low_margin_count_15,
            "low_margin_count_10": low_margin_count_10,
        },
        "binary_evaluations": {
            "must_know_vs_rest": {
                "precision": round(float(mk_binary_p), 4),
                "recall": round(float(mk_binary_r), 4),
                "f1": round(float(mk_binary_f1), 4),
            },
            "noise_vs_rest": {
                "precision": round(float(noi_binary_p), 4),
                "recall": round(float(noi_binary_r), 4),
                "f1": round(float(noi_binary_f1), 4),
            },
        },
    }


def evaluate_operational_gate(
    predictions: List[Dict[str, Any]],
    policy_name: str,
    pass_condition_fn: Callable[[Dict[str, Any]], bool],
) -> Dict[str, Any]:
    """
    Evaluate an operational gating policy against the 150 gold-standard articles.

    pass_condition_fn(p: Dict[str, Any]) -> bool:
      Returns True if article should PASS to Qwen, False if SKIP.
    """
    total = len(predictions)
    passed_records = [p for p in predictions if pass_condition_fn(p)]
    skipped_records = [p for p in predictions if not pass_condition_fn(p)]

    n_pass = len(passed_records)
    pct_pass = (n_pass / total) * 100.0 if total > 0 else 0.0
    qwen_reduction = 100.0 - pct_pass

    # Ground truth distributions
    mk_total = sum(1 for p in predictions if p["human_label"] == "MUST_KNOW")
    int_total = sum(1 for p in predictions if p["human_label"] == "INTERESTING")
    noi_total = sum(1 for p in predictions if p["human_label"] == "NOISE")

    # Passed distributions
    mk_passed = sum(1 for p in passed_records if p["human_label"] == "MUST_KNOW")
    int_passed = sum(1 for p in passed_records if p["human_label"] == "INTERESTING")
    noi_passed = sum(1 for p in passed_records if p["human_label"] == "NOISE")

    # Skipped distributions
    mk_skipped = sum(1 for p in skipped_records if p["human_label"] == "MUST_KNOW")
    int_skipped = sum(1 for p in skipped_records if p["human_label"] == "INTERESTING")
    noi_skipped = sum(1 for p in skipped_records if p["human_label"] == "NOISE")

    mk_recall = (mk_passed / mk_total) * 100.0 if mk_total > 0 else 0.0
    noise_filter_rate = (noi_skipped / noi_total) * 100.0 if noi_total > 0 else 0.0

    # Errors:
    # False skip: valuable article (MUST_KNOW or INTERESTING) wrongly skipped
    false_skips_mk = mk_skipped
    false_skips_int = int_skipped
    total_false_skips = false_skips_mk + false_skips_int

    # False pass: noise article wrongly passed to Qwen
    total_false_passes = noi_passed

    return {
        "policy": policy_name,
        "articles_passed": n_pass,
        "articles_skipped": total - n_pass,
        "pct_passed": round(pct_pass, 2),
        "qwen_reduction_pct": round(qwen_reduction, 2),
        "must_know_passed": mk_passed,
        "must_know_total": mk_total,
        "must_know_recall_pct": round(mk_recall, 2),
        "must_know_false_negatives": mk_skipped,
        "noise_passed": noi_passed,
        "noise_total": noi_total,
        "noise_filtered": noi_skipped,
        "noise_filter_rate_pct": round(noise_filter_rate, 2),
        "interesting_passed": int_passed,
        "interesting_total": int_total,
        "interesting_skipped": int_skipped,
        "total_false_skips": total_false_skips,
        "total_false_passes": total_false_passes,
    }


def run_experiment_matrix(
    labels_file: str = "evaluation/phase_0_2_labels.jsonl",
    model_id: str = "convaiinnovations/laya",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Execute complete Phase 0.4 Experiment Matrix:
    - E1: Baseline Formulation + Context A
    - E2: Baseline Formulation + Context B
    - E3: Baseline Formulation + Context C
    - E4: Rubric Formulation + Context A
    - E5: Rubric Formulation + Context B
    - E6: Rubric Formulation + Context C

    Returns:
        (all_prediction_records, experiment_metrics, operational_analysis)
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

    verify_dataset_integrity(labeled_records)

    logger.info("Loading Laya model: %s", model_id)
    t0_load = time.perf_counter()
    agent = laya.load(model_id)
    load_time_sec = time.perf_counter() - t0_load

    experiments_spec = [
        {
            "experiment_id": "E1",
            "formulation": "baseline",
            "context_strategy": "context_a",
            "description": "Baseline formulation + Context A (title + source)",
            "questions": BASELINE_QUESTIONS,
            "target_key": "importance",
        },
        {
            "experiment_id": "E2",
            "formulation": "baseline",
            "context_strategy": "context_b",
            "description": "Baseline formulation + Context B (title + source + lead 400 chars)",
            "questions": BASELINE_QUESTIONS,
            "target_key": "importance",
        },
        {
            "experiment_id": "E3",
            "formulation": "baseline",
            "context_strategy": "context_c",
            "description": "Baseline formulation + Context C (title + source + bounded 800 chars)",
            "questions": BASELINE_QUESTIONS,
            "target_key": "importance",
        },
        {
            "experiment_id": "E4",
            "formulation": "rubric_aligned",
            "context_strategy": "context_a",
            "description": "Rubric formulation + Context A (title + source)",
            "questions": RUBRIC_QUESTIONS,
            "target_key": "final_importance",
        },
        {
            "experiment_id": "E5",
            "formulation": "rubric_aligned",
            "context_strategy": "context_b",
            "description": "Rubric formulation + Context B (title + source + lead 400 chars)",
            "questions": RUBRIC_QUESTIONS,
            "target_key": "final_importance",
        },
        {
            "experiment_id": "E6",
            "formulation": "rubric_aligned",
            "context_strategy": "context_c",
            "description": "Rubric formulation + Context C (title + source + bounded 800 chars)",
            "questions": RUBRIC_QUESTIONS,
            "target_key": "final_importance",
        },
    ]

    all_predictions: List[Dict[str, Any]] = []
    experiment_metrics: Dict[str, Any] = {}
    experiment_preds_by_id: Dict[str, List[Dict[str, Any]]] = {}

    for exp in experiments_spec:
        exp_id = exp["experiment_id"]
        formulation = exp["formulation"]
        ctx_strat = exp["context_strategy"]
        q_schema = exp["questions"]
        target_k = exp["target_key"]

        logger.info("Executing experiment %s: %s", exp_id, exp["description"])

        exp_preds: List[Dict[str, Any]] = []
        y_true: List[str] = []
        y_pred: List[str] = []
        probs_list: List[Dict[str, float]] = []
        latencies: List[float] = []

        t0_exp = time.perf_counter()

        for rec in labeled_records:
            art_id = rec["article_id"]
            title = rec.get("title", "")
            source = rec.get("source", "")
            human_label = rec["human_label"]

            state_text, text_src_type = build_context_text(rec, ctx_strat)

            t_start = time.perf_counter()
            res = agent.predict(state_text, q_schema)
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(lat_ms)

            answers = res.get("answers", {})
            usage = res.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)

            # Extract target answer
            target_ans = answers.get(target_k, {})
            raw_choice = target_ans.get("choice", "")
            raw_probs = target_ans.get("probabilities", {})
            conf = target_ans.get("confidence", 0.0)

            # Map to canonical
            mapped_label, canonical_probs = map_prediction_to_canonical(
                formulation, raw_choice, raw_probs
            )

            # Collect intermediate decisions
            intermediates = {}
            for q_name, q_val in answers.items():
                if q_name != target_k:
                    intermediates[q_name] = {
                        "choice": q_val.get("choice"),
                        "probabilities": q_val.get("probabilities"),
                        "confidence": q_val.get("confidence"),
                    }

            pred_row = {
                "article_id": art_id,
                "experiment_id": exp_id,
                "formulation": formulation,
                "context_strategy": ctx_strat,
                "title": title,
                "source": source,
                "human_label": human_label,
                "predicted_final_label": mapped_label,
                "raw_target_choice": raw_choice,
                "probabilities": canonical_probs,
                "raw_probabilities": raw_probs,
                "confidence": round(float(conf), 4) if conf is not None else None,
                "input_token_count": input_tokens,
                "context_text_source_type": text_src_type,
                "context_text_length": len(state_text),
                "latency_ms": round(lat_ms, 2),
                "intermediate_decisions": intermediates,
                "model_id": model_id,
                "device": str(getattr(agent, "device", "unknown")),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            exp_preds.append(pred_row)
            all_predictions.append(pred_row)
            y_true.append(human_label)
            y_pred.append(mapped_label)
            probs_list.append(canonical_probs)

        exp_time = time.perf_counter() - t0_exp
        metrics = compute_comprehensive_metrics(y_true, y_pred, probs_list, latencies)
        metrics["runtime_seconds"] = round(exp_time, 2)
        metrics["description"] = exp["description"]
        metrics["formulation"] = formulation
        metrics["context_strategy"] = ctx_strat
        metrics["model_id"] = model_id
        metrics["max_len"] = agent.cfg.get("max_len")
        metrics["head_max_len"] = agent.cfg.get("head_max_len")

        experiment_metrics[exp_id] = metrics
        experiment_preds_by_id[exp_id] = exp_preds

    # Compute Operational Gate Analysis across meaningful policies
    # We evaluate policies on E2 (Baseline Lead 400), E3 (Baseline Lead 800),
    # E5 (Rubric Lead 400), and E6 (Rubric Lead 800).
    operational_policies = []

    for eval_exp_id in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        preds = experiment_preds_by_id[eval_exp_id]

        # Policy 1: Standard ArgMax Gating (Pass if predicted in {MUST_KNOW, INTERESTING})
        operational_policies.append(
            evaluate_operational_gate(
                preds,
                f"{eval_exp_id}: ArgMax Gate (Skip NOISE)",
                lambda p: p["predicted_final_label"] != "NOISE",
            )
        )

        # Policy 2: Conservative Must-Know Only Gate (Pass ONLY if predicted == MUST_KNOW)
        operational_policies.append(
            evaluate_operational_gate(
                preds,
                f"{eval_exp_id}: Strict Must-Know Only",
                lambda p: p["predicted_final_label"] == "MUST_KNOW",
            )
        )

        # Policy 3: Noise Probability Filtering (Pass if P(NOISE) < 0.40)
        operational_policies.append(
            evaluate_operational_gate(
                preds,
                f"{eval_exp_id}: Prob Gate P(NOISE) < 0.40",
                lambda p: p["probabilities"].get("NOISE", 0.0) < 0.40,
            )
        )

        # Policy 4: Noise Probability Filtering (Pass if P(NOISE) < 0.25)
        operational_policies.append(
            evaluate_operational_gate(
                preds,
                f"{eval_exp_id}: Prob Gate P(NOISE) < 0.25",
                lambda p: p["probabilities"].get("NOISE", 0.0) < 0.25,
            )
        )

        # Policy 5: Guarded Must-Know Sensitivity (Pass if P(MUST_KNOW) >= 0.20 or predicted != NOISE)
        operational_policies.append(
            evaluate_operational_gate(
                preds,
                f"{eval_exp_id}: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)",
                lambda p: p["probabilities"].get("MUST_KNOW", 0.0) >= 0.20
                or p["predicted_final_label"] != "NOISE",
            )
        )

    # Reference Baseline: Pass All (No Gating)
    operational_policies.append(
        evaluate_operational_gate(
            experiment_preds_by_id["E2"],
            "Reference Baseline: Pass All to Qwen (No Filter)",
            lambda p: True,
        )
    )

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = (
        rusage.ru_maxrss / (1024.0 * 1024.0)
        if os.uname().sysname == "Darwin"
        else rusage.ru_maxrss / 1024.0
    )

    operational_analysis = {
        "metadata": {
            "model_id": model_id,
            "device": str(getattr(agent, "device", "unknown")),
            "load_time_sec": round(load_time_sec, 3),
            "max_rss_mb": round(max_rss_mb, 1),
            "total_predictions": len(all_predictions),
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "experiment_metrics": experiment_metrics,
        "operational_policies": operational_policies,
    }

    return all_predictions, experiment_metrics, operational_analysis


def write_predictions_jsonl(predictions: List[Dict[str, Any]], filepath: str) -> None:
    """Save all predictions to JSON Lines file."""
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def write_operational_analysis_json(data: Dict[str, Any], filepath: str) -> None:
    """Save operational analysis data to JSON file."""
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    labels_in = sys.argv[1] if len(sys.argv) > 1 else "evaluation/phase_0_2_labels.jsonl"
    out_preds = sys.argv[2] if len(sys.argv) > 2 else "evaluation/phase_0_4_predictions.jsonl"
    out_ops = sys.argv[3] if len(sys.argv) > 3 else "evaluation/phase_0_4_operational_analysis.json"

    print("=================================================================")
    print("PHASE 0.4: LAYA DECISION & CONTEXT ABLATION EXPERIMENT RUNNER")
    print("=================================================================")
    preds, metrics, ops = run_experiment_matrix(labels_in)

    write_predictions_jsonl(preds, out_preds)
    print(f"-> Successfully wrote {len(preds)} prediction rows to {out_preds}")

    write_operational_analysis_json(ops, out_ops)
    print(f"-> Successfully wrote operational analysis to {out_ops}")

    print("\n----------------- EXPERIMENT RESULTS SUMMARY -----------------")
    print(f"{'Exp':<4} | {'Formulation':<15} | {'Context':<10} | {'Acc':<7} | {'MK Rec':<8} | {'NOISE Rec':<9} | {'Pred MK/INT/NOI'}")
    print("-" * 80)
    for exp_id, m in metrics.items():
        dist = m['prediction_distribution']
        dist_str = f"{dist.get('MUST_KNOW', 0)}/{dist.get('INTERESTING', 0)}/{dist.get('NOISE', 0)}"
        print(
            f"{exp_id:<4} | {m['formulation']:<15} | {m['context_strategy']:<10} | "
            f"{m['accuracy']*100:5.2f}% | {m['must_know_recall']*100:6.2f}% | {m['noise_recall']*100:7.2f}% | "
            f"{dist_str}"
        )
    print("=================================================================")
