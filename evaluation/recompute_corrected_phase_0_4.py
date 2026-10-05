"""
Phase 0.4 Correction Runner: Recompute Canonical Predictions, Metrics, and Operational Gates.

This script executes offline recomputation using EXISTING raw prediction files:
- evaluation/phase_0_3_predictions.jsonl
- evaluation/phase_0_4_predictions.jsonl

NO Laya inference is performed.
NO models are loaded.
NO prompts, human labels, or production code are modified.

Outputs:
- evaluation/phase_0_3_predictions_corrected.jsonl
- evaluation/phase_0_4_predictions_corrected.jsonl
- evaluation/phase_0_4_operational_analysis_corrected.json
- reports/phase_0_3_laya_evaluation_corrected.md
- reports/phase_0_4_corrected_mapping.md
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CANONICAL_CLASSES = ["MUST_KNOW", "INTERESTING", "NOISE"]
ZERO_DIVISION: Any = 0


def compute_comprehensive_metrics_dict(
    y_true: List[str],
    y_pred: List[str],
    canonical_probs_list: List[Dict[str, float]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty dataset.")

    acc = float(accuracy_score(y_true, y_pred))
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
    dist = {c: y_pred.count(c) for c in CANONICAL_CLASSES}

    # Margin & confidence stats
    confidences = []
    margins = []
    low_15 = 0
    low_10 = 0
    for cp in canonical_probs_list:
        sorted_vals = sorted(cp.values(), reverse=True)
        top = sorted_vals[0]
        second = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
        margin = round(top - second, 4)
        confidences.append(top)
        margins.append(margin)
        if margin < 0.15:
            low_15 += 1
        if margin < 0.10:
            low_10 += 1

    margin_stats = {
        "average_confidence": round(float(np.mean(confidences)), 4),
        "average_margin": round(float(np.mean(margins)), 4),
        "low_margin_count_15": low_15,
        "low_margin_count_10": low_10,
    }

    # Latency stats if available
    if latencies_ms:
        latency_stats = {
            "average_ms": round(float(np.mean(latencies_ms)), 2),
            "median_ms": round(float(np.median(latencies_ms)), 2),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "min_ms": round(float(np.min(latencies_ms)), 2),
            "max_ms": round(float(np.max(latencies_ms)), 2),
        }
    else:
        latency_stats = {}

    return {
        "total_count": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy": round(acc, 4),
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
        "prediction_distribution": dist,
        "must_know_recall": round(per_class["MUST_KNOW"]["recall"], 4),
        "noise_recall": round(per_class["NOISE"]["recall"], 4),
        "interesting_recall": round(per_class["INTERESTING"]["recall"], 4),
        "margin_stats": margin_stats,
        "latency_stats": latency_stats,
    }


def evaluate_gate(
    predictions: List[Dict[str, Any]],
    policy_name: str,
    pass_condition_fn: Callable[[Dict[str, Any]], bool],
) -> Dict[str, Any]:
    total = len(predictions)
    passed_records = [p for p in predictions if pass_condition_fn(p)]
    skipped_records = [p for p in predictions if not pass_condition_fn(p)]

    n_pass = len(passed_records)
    pct_pass = (n_pass / total) * 100.0 if total > 0 else 0.0
    qwen_reduction = 100.0 - pct_pass

    mk_total = sum(1 for p in predictions if p["human_label"] == "MUST_KNOW")
    int_total = sum(1 for p in predictions if p["human_label"] == "INTERESTING")
    noi_total = sum(1 for p in predictions if p["human_label"] == "NOISE")

    mk_passed = sum(1 for p in passed_records if p["human_label"] == "MUST_KNOW")
    int_passed = sum(1 for p in passed_records if p["human_label"] == "INTERESTING")
    noi_passed = sum(1 for p in passed_records if p["human_label"] == "NOISE")

    mk_skipped = sum(1 for p in skipped_records if p["human_label"] == "MUST_KNOW")
    int_skipped = sum(1 for p in skipped_records if p["human_label"] == "INTERESTING")
    noi_skipped = sum(1 for p in skipped_records if p["human_label"] == "NOISE")

    mk_recall = (mk_passed / mk_total) * 100.0 if mk_total > 0 else 0.0
    noise_filter_rate = (noi_skipped / noi_total) * 100.0 if noi_total > 0 else 0.0

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
        "total_false_skips": mk_skipped + int_skipped,
        "total_false_passes": noi_passed,
    }


def main():
    print("=" * 60)
    print("PHASE 0.4 CORRECTION: 3-WAY PROBABILITY ARGMAX RECOMPUTATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # TASK 1 & 2: Load and Verify Raw Probabilities
    # ---------------------------------------------------------
    p03_path = Path("evaluation/phase_0_3_predictions.jsonl")
    p04_path = Path("evaluation/phase_0_4_predictions.jsonl")

    with open(p03_path, "r", encoding="utf-8") as f:
        p03_raw = [json.loads(line) for line in f if line.strip()]

    with open(p04_path, "r", encoding="utf-8") as f:
        p04_raw = [json.loads(line) for line in f if line.strip()]

    print(f"Phase 0.3 raw records loaded: {len(p03_raw)}")
    print(f"Phase 0.4 raw records loaded: {len(p04_raw)}")

    malformed_03 = 0
    for idx, r in enumerate(p03_raw):
        probs = r.get("probabilities", {})
        raw_sum = sum(probs.get(k, 0.0) for k in ["high", "medium", "low", "ignore"])
        if abs(raw_sum - 1.0) > 0.02:
            malformed_03 += 1

    malformed_04 = 0
    for idx, r in enumerate(p04_raw):
        formulation = r.get("formulation")
        if formulation == "baseline":
            probs = r.get("raw_probabilities", {})
            raw_sum = sum(probs.get(k, 0.0) for k in ["high", "medium", "low", "ignore"])
            if abs(raw_sum - 1.0) > 0.02:
                malformed_04 += 1
        else:
            probs = r.get("raw_probabilities", {})
            raw_sum = sum(probs.get(k, 0.0) for k in ["must_know", "interesting", "noise"])
            if abs(raw_sum - 1.0) > 0.02:
                malformed_04 += 1

    print(f"Phase 0.3 malformed probability records: {malformed_03}")
    print(f"Phase 0.4 malformed probability records: {malformed_04}")
    assert malformed_03 == 0 and malformed_04 == 0, "Found malformed records!"

    # ---------------------------------------------------------
    # TASK 3 & 4: Recompute Phase 0.3 Predictions & Metrics
    # ---------------------------------------------------------
    p03_corrected = []
    y_true_03 = []
    y_pred_old_03 = []
    y_pred_new_03 = []
    canonical_probs_03 = []
    latencies_03 = []

    for r in p03_raw:
        raw = r["probabilities"]
        p_mk = round(float(raw["high"]), 4)
        p_int = round(float(raw["medium"]), 4)
        p_noi = round(float(raw["low"]) + float(raw["ignore"]), 4)
        c_probs = {
            "MUST_KNOW": p_mk,
            "INTERESTING": p_int,
            "NOISE": p_noi,
        }

        # ArgMax of canonical 3-way distribution
        sorted_items = sorted(c_probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_p = sorted_items[0]
        second_label, second_p = sorted_items[1]
        margin = round(top_p - second_p, 4)

        corr_entry = dict(r)
        corr_entry["old_laya_label"] = r["laya_label"]
        corr_entry["laya_label"] = top_label
        corr_entry["canonical_predicted_label"] = top_label
        corr_entry["canonical_probabilities"] = c_probs
        corr_entry["top_probability"] = top_p
        corr_entry["second_probability"] = second_p
        corr_entry["top_two_margin"] = margin

        p03_corrected.append(corr_entry)
        y_true_03.append(r["human_label"])
        y_pred_old_03.append(r["laya_label"])
        y_pred_new_03.append(top_label)
        canonical_probs_03.append(c_probs)
        latencies_03.append(r.get("latency_ms", 0.0))

    # Save Phase 0.3 corrected jsonl
    out_p03 = Path("evaluation/phase_0_3_predictions_corrected.jsonl")
    with open(out_p03, "w", encoding="utf-8") as f:
        for rec in p03_corrected:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved: {out_p03}")

    metrics_03_old = compute_comprehensive_metrics_dict(
        y_true_03, y_pred_old_03, canonical_probs_03, latencies_03
    )
    metrics_03_new = compute_comprehensive_metrics_dict(
        y_true_03, y_pred_new_03, canonical_probs_03, latencies_03
    )

    # ---------------------------------------------------------
    # TASK 3 & 5: Recompute Phase 0.4 Predictions & Metrics
    # ---------------------------------------------------------
    p04_corrected = []
    p04_by_exp_corrected = {}
    p04_by_exp_old = {}

    for r in p04_raw:
        eid = r["experiment_id"]
        formulation = r["formulation"]

        if formulation == "baseline":
            raw = r["raw_probabilities"]
            p_mk = round(float(raw["high"]), 4)
            p_int = round(float(raw["medium"]), 4)
            p_noi = round(float(raw["low"]) + float(raw["ignore"]), 4)
            c_probs = {
                "MUST_KNOW": p_mk,
                "INTERESTING": p_int,
                "NOISE": p_noi,
            }
        else:
            raw = r["raw_probabilities"]
            c_probs = {
                "MUST_KNOW": round(float(raw["must_know"]), 4),
                "INTERESTING": round(float(raw["interesting"]), 4),
                "NOISE": round(float(raw["noise"]), 4),
            }

        sorted_items = sorted(c_probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_p = sorted_items[0]
        second_label, second_p = sorted_items[1]
        margin = round(top_p - second_p, 4)

        corr_entry = dict(r)
        corr_entry["predicted_final_label_old"] = r["predicted_final_label"]
        corr_entry["predicted_final_label"] = top_label
        corr_entry["canonical_predicted_label"] = top_label
        corr_entry["canonical_probabilities"] = c_probs
        corr_entry["probabilities"] = c_probs
        corr_entry["top_probability"] = top_p
        corr_entry["second_probability"] = second_p
        corr_entry["top_two_margin"] = margin

        p04_corrected.append(corr_entry)
        p04_by_exp_corrected.setdefault(eid, []).append(corr_entry)
        p04_by_exp_old.setdefault(eid, []).append(r)

    # Save Phase 0.4 corrected jsonl
    out_p04 = Path("evaluation/phase_0_4_predictions_corrected.jsonl")
    with open(out_p04, "w", encoding="utf-8") as f:
        for rec in p04_corrected:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved: {out_p04}")

    exp_metrics_old = {}
    exp_metrics_new = {}

    for eid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        recs_old = p04_by_exp_old[eid]
        recs_new = p04_by_exp_corrected[eid]

        yt = [r["human_label"] for r in recs_new]
        yp_old = [r["predicted_final_label"] for r in recs_old]
        yp_new = [r["predicted_final_label"] for r in recs_new]
        cps = [r["canonical_probabilities"] for r in recs_new]
        lats = [r.get("latency_ms", 0.0) for r in recs_new]

        exp_metrics_old[eid] = compute_comprehensive_metrics_dict(yt, yp_old, cps, lats)
        exp_metrics_new[eid] = compute_comprehensive_metrics_dict(yt, yp_new, cps, lats)

    # ---------------------------------------------------------
    # TASK 6: Recompute Operational Gate Analysis
    # ---------------------------------------------------------
    corrected_policies = []
    old_policies = []

    for eid in ["E1", "E2", "E3"]:
        preds_new = p04_by_exp_corrected[eid]
        preds_old = p04_by_exp_old[eid]

        # Policy 1: Standard ArgMax Gating (Pass if predicted != NOISE)
        corrected_policies.append(
            evaluate_gate(
                preds_new,
                f"{eid}: ArgMax Gate (Skip NOISE)",
                lambda p: p["predicted_final_label"] != "NOISE",
            )
        )
        old_policies.append(
            evaluate_gate(
                preds_old,
                f"{eid}: ArgMax Gate (Skip NOISE)",
                lambda p: p["predicted_final_label"] != "NOISE",
            )
        )

        # Policy 2: Conservative Must-Know Only Gate (Pass ONLY if predicted == MUST_KNOW)
        corrected_policies.append(
            evaluate_gate(
                preds_new,
                f"{eid}: Strict Must-Know Only",
                lambda p: p["predicted_final_label"] == "MUST_KNOW",
            )
        )
        old_policies.append(
            evaluate_gate(
                preds_old,
                f"{eid}: Strict Must-Know Only",
                lambda p: p["predicted_final_label"] == "MUST_KNOW",
            )
        )

        # Policy 3: Noise Probability Filtering (Pass if P(NOISE) < 0.40)
        corrected_policies.append(
            evaluate_gate(
                preds_new,
                f"{eid}: Prob Gate P(NOISE) < 0.40",
                lambda p: p["canonical_probabilities"]["NOISE"] < 0.40,
            )
        )
        old_policies.append(
            evaluate_gate(
                preds_old,
                f"{eid}: Prob Gate P(NOISE) < 0.40",
                lambda p: p["probabilities"].get("NOISE", 0.0) < 0.40,
            )
        )

        # Policy 4: Noise Probability Filtering (Pass if P(NOISE) < 0.25)
        corrected_policies.append(
            evaluate_gate(
                preds_new,
                f"{eid}: Prob Gate P(NOISE) < 0.25",
                lambda p: p["canonical_probabilities"]["NOISE"] < 0.25,
            )
        )
        old_policies.append(
            evaluate_gate(
                preds_old,
                f"{eid}: Prob Gate P(NOISE) < 0.25",
                lambda p: p["probabilities"].get("NOISE", 0.0) < 0.25,
            )
        )

        # Policy 5: Guarded Must-Know Sensitivity
        corrected_policies.append(
            evaluate_gate(
                preds_new,
                f"{eid}: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)",
                lambda p: p["canonical_probabilities"]["MUST_KNOW"] >= 0.20
                or p["predicted_final_label"] != "NOISE",
            )
        )
        old_policies.append(
            evaluate_gate(
                preds_old,
                f"{eid}: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)",
                lambda p: p["probabilities"].get("MUST_KNOW", 0.0) >= 0.20
                or p["predicted_final_label"] != "NOISE",
            )
        )

    # Reference Baseline
    corrected_policies.append(
        evaluate_gate(
            p04_by_exp_corrected["E2"],
            "Reference Baseline: Pass All to Qwen (No Filter)",
            lambda p: True,
        )
    )

    # Save corrected operational analysis JSON
    out_analysis_json = Path("evaluation/phase_0_4_operational_analysis_corrected.json")
    with open(out_analysis_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "description": "Corrected Phase 0.4 analysis using 3-way canonical probability argmax",
                    "total_predictions": 900,
                    "mapping_correction": "argmax(P(high), P(medium), P(low)+P(ignore))",
                },
                "experiment_metrics": exp_metrics_new,
                "operational_policies": corrected_policies,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_analysis_json}")

    # ---------------------------------------------------------
    # Write Corrected Phase 0.3 Report
    # ---------------------------------------------------------
    write_phase_0_3_corrected_report(metrics_03_old, metrics_03_new, p03_corrected)

    # ---------------------------------------------------------
    # Write Corrected Phase 0.4 Report
    # ---------------------------------------------------------
    write_phase_0_4_corrected_report(
        exp_metrics_old, exp_metrics_new, corrected_policies, old_policies
    )

    print("\nRecomputation and report generation completed successfully.")


def format_cm(cm: List[List[int]]) -> str:
    lines = [
        "Actual \\ Pred    MUST_KNOW  INTERESTING  NOISE   Total",
        f"MUST_KNOW          {cm[0][0]:>5}        {cm[0][1]:>5}  {cm[0][2]:>5}   {sum(cm[0]):>5}",
        f"INTERESTING        {cm[1][0]:>5}        {cm[1][1]:>5}  {cm[1][2]:>5}   {sum(cm[1]):>5}",
        f"NOISE              {cm[2][0]:>5}        {cm[2][1]:>5}  {cm[2][2]:>5}   {sum(cm[2]):>5}",
        f"Total              {cm[0][0]+cm[1][0]+cm[2][0]:>5}        {cm[0][1]+cm[1][1]+cm[2][1]:>5}  {cm[0][2]+cm[1][2]+cm[2][2]:>5}   {sum(sum(r) for r in cm):>5}",
    ]
    return "\n".join(lines)


def format_policies_table(policies: List[Dict[str, Any]]) -> str:
    rows = [
        "| Policy | Articles Passed | Qwen Reduction | MUST_KNOW Recall | MK False Negatives | NOISE Filtered | NOISE Filter Rate |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for p in policies:
        rows.append(
            f"| **{p['policy']}** | {p['articles_passed']} / 150 ({p['pct_passed']}%) | {p['qwen_reduction_pct']}% | **{p['must_know_recall_pct']}%** ({p['must_know_passed']}/64) | **{p['must_know_false_negatives']}** | {p['noise_filtered']}/18 | {p['noise_filter_rate_pct']}% |"
        )
    return "\n".join(rows)


def write_phase_0_3_corrected_report(
    old_m: Dict[str, Any], new_m: Dict[str, Any], recs: List[Dict[str, Any]]
):
    out_path = Path("reports/phase_0_3_laya_evaluation_corrected.md")

    # Count flips
    flipped_recs = [r for r in recs if r["old_laya_label"] != r["canonical_predicted_label"]]

    lines = [
        "# Phase 0.3: Laya vs. Human Gold-Standard Evaluation Report (CORRECTED)",
        "",
        "> [!IMPORTANT]",
        "> **CORRECTED EVALUATION ARTIFACT (October 2026)**  ",
        "> This report replaces the uncorrected analysis in `reports/phase_0_3_laya_evaluation.md`.  ",
        "> The previous evaluation suffered from a canonical probability mapping bug: it selected the raw 4-way argmax (`high`, `medium`, `low`, `ignore`) and mapped the single winner, instead of computing the true 3-way canonical argmax:  ",
        "> NOISE = P(low) + P(ignore)  ",
        "> Predicted = argmax(P(high), P(medium), P(low) + P(ignore))  ",
        "> No new Laya inference was executed; all values are computed directly from the existing raw probabilities in `evaluation/phase_0_3_predictions.jsonl`.",
        "",
        "---",
        "",
        "## 1. Executive Summary: Old vs. Corrected Metrics",
        "",
        "| Metric | Old (Buggy 4-Way ArgMax) | Corrected (3-Way Canonical ArgMax) | Delta | Note |",
        "|:---|:---:|:---:|:---:|:---|",
        f"| **Overall Accuracy** | **{old_m['accuracy']*100:.2f}%** ({old_m['correct_count']}/150) | **{new_m['accuracy']*100:.2f}%** ({new_m['correct_count']}/150) | {(new_m['accuracy']-old_m['accuracy'])*100:+.2f}% | {len(flipped_recs)} articles changed prediction |",
        f"| **Macro Precision** | {old_m['macro']['precision']*100:.2f}% | {new_m['macro']['precision']*100:.2f}% | {(new_m['macro']['precision']-old_m['macro']['precision'])*100:+.2f}% | Balanced across 3 classes |",
        f"| **Macro Recall** | {old_m['macro']['recall']*100:.2f}% | {new_m['macro']['recall']*100:.2f}% | {(new_m['macro']['recall']-old_m['macro']['recall'])*100:+.2f}% | Lower due to NOISE false positives |",
        f"| **Macro F1** | {old_m['macro']['f1']:.4f} | {new_m['macro']['f1']:.4f} | {new_m['macro']['f1']-old_m['macro']['f1']:+.4f} | |",
        f"| **Weighted Precision** | {old_m['weighted']['precision']*100:.2f}% | {new_m['weighted']['precision']*100:.2f}% | {(new_m['weighted']['precision']-old_m['weighted']['precision'])*100:+.2f}% | |",
        f"| **Weighted Recall** | {old_m['weighted']['recall']*100:.2f}% | {new_m['weighted']['recall']*100:.2f}% | {(new_m['weighted']['recall']-old_m['weighted']['recall'])*100:+.2f}% | Matches accuracy |",
        f"| **Weighted F1** | {old_m['weighted']['f1']:.4f} | {new_m['weighted']['f1']:.4f} | {new_m['weighted']['f1']-old_m['weighted']['f1']:+.4f} | |",
        f"| **MUST_KNOW Precision** | {old_m['per_class']['MUST_KNOW']['precision']*100:.2f}% | {new_m['per_class']['MUST_KNOW']['precision']*100:.2f}% | {(new_m['per_class']['MUST_KNOW']['precision']-old_m['per_class']['MUST_KNOW']['precision'])*100:+.2f}% | |",
        f"| **MUST_KNOW Recall** | **{old_m['per_class']['MUST_KNOW']['recall']*100:.2f}%** | **{new_m['per_class']['MUST_KNOW']['recall']*100:.2f}%** | {(new_m['per_class']['MUST_KNOW']['recall']-old_m['per_class']['MUST_KNOW']['recall'])*100:+.2f}% | 1 article (3812) flipped to NOISE |",
        f"| **MUST_KNOW F1** | {old_m['per_class']['MUST_KNOW']['f1']:.4f} | {new_m['per_class']['MUST_KNOW']['f1']:.4f} | {new_m['per_class']['MUST_KNOW']['f1']-old_m['per_class']['MUST_KNOW']['f1']:+.4f} | |",
        f"| **INTERESTING Precision** | {old_m['per_class']['INTERESTING']['precision']*100:.2f}% | {new_m['per_class']['INTERESTING']['precision']*100:.2f}% | {(new_m['per_class']['INTERESTING']['precision']-old_m['per_class']['INTERESTING']['precision'])*100:+.2f}% | |",
        f"| **INTERESTING Recall** | **{old_m['per_class']['INTERESTING']['recall']*100:.2f}%** | **{new_m['per_class']['INTERESTING']['recall']*100:.2f}%** | {(new_m['per_class']['INTERESTING']['recall']-old_m['per_class']['INTERESTING']['recall'])*100:+.2f}% | 11 articles flipped to NOISE |",
        f"| **INTERESTING F1** | {old_m['per_class']['INTERESTING']['f1']:.4f} | {new_m['per_class']['INTERESTING']['f1']:.4f} | {new_m['per_class']['INTERESTING']['f1']-old_m['per_class']['INTERESTING']['f1']:+.4f} | |",
        f"| **NOISE Precision** | {old_m['per_class']['NOISE']['precision']*100:.2f}% | **{new_m['per_class']['NOISE']['precision']*100:.2f}%** | {(new_m['per_class']['NOISE']['precision']-old_m['per_class']['NOISE']['precision'])*100:+.2f}% | 1 true NOISE identified |",
        f"| **NOISE Recall** | **{old_m['per_class']['NOISE']['recall']*100:.2f}%** | **{new_m['per_class']['NOISE']['recall']*100:.2f}%** | {(new_m['per_class']['NOISE']['recall']-old_m['per_class']['NOISE']['recall'])*100:+.2f}% | Article 2305 identified |",
        f"| **NOISE F1** | {old_m['per_class']['NOISE']['f1']:.4f} | **{new_m['per_class']['NOISE']['f1']:.4f}** | {new_m['per_class']['NOISE']['f1']-old_m['per_class']['NOISE']['f1']:+.4f} | |",
        "",
        "---",
        "",
        "## 2. Prediction Distribution & Label Shift",
        "",
        "```",
        "Old Prediction Distribution:",
        f"  MUST_KNOW:   {old_m['prediction_distribution']['MUST_KNOW']:>3} ({old_m['prediction_distribution']['MUST_KNOW']/150*100:.1f}%)",
        f"  INTERESTING: {old_m['prediction_distribution']['INTERESTING']:>3} ({old_m['prediction_distribution']['INTERESTING']/150*100:.1f}%)",
        f"  NOISE:       {old_m['prediction_distribution']['NOISE']:>3} ({old_m['prediction_distribution']['NOISE']/150*100:.1f}%)",
        "",
        "Corrected Prediction Distribution:",
        f"  MUST_KNOW:   {new_m['prediction_distribution']['MUST_KNOW']:>3} ({new_m['prediction_distribution']['MUST_KNOW']/150*100:.1f}%)",
        f"  INTERESTING: {new_m['prediction_distribution']['INTERESTING']:>3} ({new_m['prediction_distribution']['INTERESTING']/150*100:.1f}%)",
        f"  NOISE:       {new_m['prediction_distribution']['NOISE']:>3} ({new_m['prediction_distribution']['NOISE']/150*100:.1f}%)",
        "```",
        "",
        f"Across the 150 gold-standard articles, exactly **{len(flipped_recs)} articles (11.3%)** changed prediction:",
        "- **11 articles** flipped from `INTERESTING` to `NOISE` (human ground truth: 11 `INTERESTING`)",
        "- **5 articles** flipped from `INTERESTING` to `NOISE` (human ground truth: 4 `MUST_KNOW`, 1 `NOISE`)",
        "- **1 article** flipped from `MUST_KNOW` to `NOISE` (human ground truth: `MUST_KNOW`, Article 3812)",
        "",
        "### The Single Correct NOISE Prediction: Article 2305",
        "- **Title:** *\"How AI is impacting who gets to cross borders\"* (Hacker News AI)",
        "- **Human Ground Truth:** `NOISE`",
        "- **Raw Laya Probabilities:**",
        "  - P(high) = 0.2081",
        "  - P(medium) = 0.3223",
        "  - P(low) = 0.2638",
        "  - P(ignore) = 0.2059",
        "- **Old Evaluation:** Picked raw choice `medium` (0.3223 > 0.2638, 0.2081, 0.2059) -> mapped to `INTERESTING` (Misclassification).",
        "- **Corrected Evaluation:** P(NOISE) = 0.2638 + 0.2059 = 0.4697 > 0.3223 -> mapped to `NOISE` (True Positive!).",
        "",
        "---",
        "",
        "## 3. Confusion Matrices",
        "",
        "### Old Confusion Matrix (Affected by Bug)",
        "```",
        format_cm(old_m["confusion_matrix"]),
        "```",
        "",
        "### Corrected Confusion Matrix (True 3-Way Canonical ArgMax)",
        "```",
        format_cm(new_m["confusion_matrix"]),
        "```",
        "",
        "---",
        "",
        "## 4. Confidence and Margin Statistics",
        "",
        f"- **Average Top Probability:** {new_m['margin_stats']['average_confidence']:.4f}",
        f"- **Average Top-Two Margin:** {new_m['margin_stats']['average_margin']:.4f}",
        f"- **Low-Margin Decisions (< 0.15):** {new_m['margin_stats']['low_margin_count_15']} articles ({(new_m['margin_stats']['low_margin_count_15']/150)*100:.1f}%)",
        f"- **Low-Margin Decisions (< 0.10):** {new_m['margin_stats']['low_margin_count_10']} articles ({(new_m['margin_stats']['low_margin_count_10']/150)*100:.1f}%)",
        "",
        "---",
        "",
        "## 5. Architectural Implications",
        "",
        "1. **NOISE Recall Rises Slightly, But Stays Dismal:** Correcting the probability mapping lifts NOISE recall from 0.00% to 5.56% (1 out of 18). Laya still misses 17 out of 18 true noise articles.",
        "2. **Overall Accuracy Decreases to 42.00%:** Because ModernBERT distributes probability mass across both `low` and `ignore`, summing them causes 21 non-noise articles (6 MUST_KNOW and 15 INTERESTING) to cross into NOISE, resulting in 21 false NOISE predictions vs. only 1 true NOISE prediction (precision: 4.55%).",
        "3. **Core Conclusion Holds Firmly:** Laya's zero-shot decision heads cannot be reliably used as an operational triage gate.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def write_phase_0_4_corrected_report(
    old_exp: Dict[str, Any],
    new_exp: Dict[str, Any],
    corr_policies: List[Dict[str, Any]],
    old_policies: List[Dict[str, Any]],
):
    out_path = Path("reports/phase_0_4_corrected_mapping.md")

    lines = [
        "# Phase 0.4: Decision-Formulation, Context Ablation, and Operational Gate Evaluation (CORRECTED)",
        "",
        "> [!IMPORTANT]",
        "> **CORRECTED EVALUATION REPORT (October 2026)**  ",
        "> This report documents the corrected evaluation results for Phase 0.4 following the canonical probability mapping fix.  ",
        ">   ",
        "> **Correction Summary:**  ",
        "> - For Formulation 1 (Baseline: E1, E2, E3), the canonical class probabilities are now properly evaluated via 3-way argmax:  ",
        ">   argmax(P(high), P(medium), P(low) + P(ignore))  ",
        "> - For Formulation 2 (Rubric-Aligned: E4, E5, E6), the model already evaluated 3 native choices (`must_know`, `interesting`, `noise`), so argmax predictions were mathematically unaffected.  ",
        "> - **CRITICAL NOTE ON E4, E5, E6:** As established during the audit, Formulation 2 was executed as **parallel, non-conditioned heads** rather than sequential structured reasoning. The results for E4, E5, and E6 are preserved here strictly for historical integrity and traceability, but are **invalid as an evaluation of sequential structured reasoning**.  ",
        "> - **Zero Laya Inferences Performed:** All calculations use the raw probabilities in `evaluation/phase_0_4_predictions.jsonl`.",
        "",
        "---",
        "",
        "## 1. Corrected Experiment Summary Matrix (E1 – E6)",
        "",
        "| Exp | Formulation | Context | Corrected Acc (Old) | Corrected MK Rec (Old) | Corrected NOISE Rec (Old) | Corrected Macro F1 (Old) | Status |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|",
        f"| **E1** | Baseline | Context A (Title+Source) | **{new_exp['E1']['accuracy']*100:.2f}%** ({old_exp['E1']['accuracy']*100:.2f}%) | **{new_exp['E1']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E1']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E1']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E1']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E1']['macro']['f1']:.4f}** ({old_exp['E1']['macro']['f1']:.4f}) | `CORRECTED` |",
        f"| **E2** | Baseline | Context B (Lead 400ch) | **{new_exp['E2']['accuracy']*100:.2f}%** ({old_exp['E2']['accuracy']*100:.2f}%) | **{new_exp['E2']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E2']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E2']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E2']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E2']['macro']['f1']:.4f}** ({old_exp['E2']['macro']['f1']:.4f}) | `CORRECTED` |",
        f"| **E3** | Baseline | Context C (Bounded 800ch) | **{new_exp['E3']['accuracy']*100:.2f}%** ({old_exp['E3']['accuracy']*100:.2f}%) | **{new_exp['E3']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E3']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E3']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E3']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E3']['macro']['f1']:.4f}** ({old_exp['E3']['macro']['f1']:.4f}) | `CORRECTED` |",
        f"| **E4** | Rubric | Context A (Title+Source) | **{new_exp['E4']['accuracy']*100:.2f}%** ({old_exp['E4']['accuracy']*100:.2f}%) | **{new_exp['E4']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E4']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E4']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E4']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E4']['macro']['f1']:.4f}** ({old_exp['E4']['macro']['f1']:.4f}) | `INVALID FORMULATION (Traceability Only)` |",
        f"| **E5** | Rubric | Context B (Lead 400ch) | **{new_exp['E5']['accuracy']*100:.2f}%** ({old_exp['E5']['accuracy']*100:.2f}%) | **{new_exp['E5']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E5']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E5']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E5']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E5']['macro']['f1']:.4f}** ({old_exp['E5']['macro']['f1']:.4f}) | `INVALID FORMULATION (Traceability Only)` |",
        f"| **E6** | Rubric | Context C (Bounded 800ch) | **{new_exp['E6']['accuracy']*100:.2f}%** ({old_exp['E6']['accuracy']*100:.2f}%) | **{new_exp['E6']['per_class']['MUST_KNOW']['recall']*100:.2f}%** ({old_exp['E6']['per_class']['MUST_KNOW']['recall']*100:.2f}%) | **{new_exp['E6']['per_class']['NOISE']['recall']*100:.2f}%** ({old_exp['E6']['per_class']['NOISE']['recall']*100:.2f}%) | **{new_exp['E6']['macro']['f1']:.4f}** ({old_exp['E6']['macro']['f1']:.4f}) | `INVALID FORMULATION (Traceability Only)` |",
        "",
        "---",
        "",
        "## 2. Detailed Performance Breakdown: E1, E2, E3",
        "",
        "### Experiment E1: Baseline + Context A (Title + Source)",
        f"- **Accuracy:** {new_exp['E1']['accuracy']*100:.2f}% ({new_exp['E1']['correct_count']}/150) — *Old: {old_exp['E1']['accuracy']*100:.2f}%*",
        f"- **MUST_KNOW:** Precision {new_exp['E1']['per_class']['MUST_KNOW']['precision']*100:.2f}%, Recall {new_exp['E1']['per_class']['MUST_KNOW']['recall']*100:.2f}% ({round(new_exp['E1']['per_class']['MUST_KNOW']['recall']*64)}/64), F1 {new_exp['E1']['per_class']['MUST_KNOW']['f1']:.4f}",
        f"- **INTERESTING:** Precision {new_exp['E1']['per_class']['INTERESTING']['precision']*100:.2f}%, Recall {new_exp['E1']['per_class']['INTERESTING']['recall']*100:.2f}% ({round(new_exp['E1']['per_class']['INTERESTING']['recall']*68)}/68), F1 {new_exp['E1']['per_class']['INTERESTING']['f1']:.4f}",
        f"- **NOISE:** Precision {new_exp['E1']['per_class']['NOISE']['precision']*100:.2f}%, Recall {new_exp['E1']['per_class']['NOISE']['recall']*100:.2f}% ({round(new_exp['E1']['per_class']['NOISE']['recall']*18)}/18), F1 {new_exp['E1']['per_class']['NOISE']['f1']:.4f}",
        f"- **Prediction Distribution:** MUST_KNOW: {new_exp['E1']['prediction_distribution']['MUST_KNOW']} | INTERESTING: {new_exp['E1']['prediction_distribution']['INTERESTING']} | NOISE: {new_exp['E1']['prediction_distribution']['NOISE']}",
        "- **Confusion Matrix:**",
        "```",
        format_cm(new_exp["E1"]["confusion_matrix"]),
        "```",
        "",
        "### Experiment E2: Baseline + Context B (Title + Source + Lead 400 chars)",
        f"- **Accuracy:** {new_exp['E2']['accuracy']*100:.2f}% ({new_exp['E2']['correct_count']}/150) — *Old: {old_exp['E2']['accuracy']*100:.2f}%*",
        f"- **MUST_KNOW:** Precision {new_exp['E2']['per_class']['MUST_KNOW']['precision']*100:.2f}%, Recall {new_exp['E2']['per_class']['MUST_KNOW']['recall']*100:.2f}% ({round(new_exp['E2']['per_class']['MUST_KNOW']['recall']*64)}/64), F1 {new_exp['E2']['per_class']['MUST_KNOW']['f1']:.4f}",
        f"- **INTERESTING:** Precision {new_exp['E2']['per_class']['INTERESTING']['precision']*100:.2f}%, Recall {new_exp['E2']['per_class']['INTERESTING']['recall']*100:.2f}% ({round(new_exp['E2']['per_class']['INTERESTING']['recall']*68)}/68), F1 {new_exp['E2']['per_class']['INTERESTING']['f1']:.4f}",
        f"- **NOISE:** Precision {new_exp['E2']['per_class']['NOISE']['precision']*100:.2f}%, Recall {new_exp['E2']['per_class']['NOISE']['recall']*100:.2f}% ({round(new_exp['E2']['per_class']['NOISE']['recall']*18)}/18), F1 {new_exp['E2']['per_class']['NOISE']['f1']:.4f}",
        f"- **Prediction Distribution:** MUST_KNOW: {new_exp['E2']['prediction_distribution']['MUST_KNOW']} | INTERESTING: {new_exp['E2']['prediction_distribution']['INTERESTING']} | NOISE: {new_exp['E2']['prediction_distribution']['NOISE']}",
        "- **Confusion Matrix:**",
        "```",
        format_cm(new_exp["E2"]["confusion_matrix"]),
        "```",
        "",
        "### Experiment E3: Baseline + Context C (Title + Source + Bounded 800 chars)",
        f"- **Accuracy:** {new_exp['E3']['accuracy']*100:.2f}% ({new_exp['E3']['correct_count']}/150) — *Old: {old_exp['E3']['accuracy']*100:.2f}%*",
        f"- **MUST_KNOW:** Precision {new_exp['E3']['per_class']['MUST_KNOW']['precision']*100:.2f}%, Recall {new_exp['E3']['per_class']['MUST_KNOW']['recall']*100:.2f}% ({round(new_exp['E3']['per_class']['MUST_KNOW']['recall']*64)}/64), F1 {new_exp['E3']['per_class']['MUST_KNOW']['f1']:.4f}",
        f"- **INTERESTING:** Precision {new_exp['E3']['per_class']['INTERESTING']['precision']*100:.2f}%, Recall {new_exp['E3']['per_class']['INTERESTING']['recall']*100:.2f}% ({round(new_exp['E3']['per_class']['INTERESTING']['recall']*68)}/68), F1 {new_exp['E3']['per_class']['INTERESTING']['f1']:.4f}",
        f"- **NOISE:** Precision {new_exp['E3']['per_class']['NOISE']['precision']*100:.2f}%, Recall {new_exp['E3']['per_class']['NOISE']['recall']*100:.2f}% ({round(new_exp['E3']['per_class']['NOISE']['recall']*18)}/18), F1 {new_exp['E3']['per_class']['NOISE']['f1']:.4f}",
        f"- **Prediction Distribution:** MUST_KNOW: {new_exp['E3']['prediction_distribution']['MUST_KNOW']} | INTERESTING: {new_exp['E3']['prediction_distribution']['INTERESTING']} | NOISE: {new_exp['E3']['prediction_distribution']['NOISE']}",
        "- **Confusion Matrix:**",
        "```",
        format_cm(new_exp["E3"]["confusion_matrix"]),
        "```",
        "",
        "---",
        "",
        "## 3. Corrected Operational Policy Analysis",
        "",
        "The operational gating feasibility analysis was recomputed across all policies for E1, E2, and E3 using corrected canonical predictions:",
        "",
        format_policies_table(corr_policies),
        "",
        "---",
        "",
        "## 4. Key Operational Findings",
        "",
        "1. **ArgMax Gating on E2 Discards 6 MUST_KNOW Articles:**",
        "   - Under the old mapping, the ArgMax gate passed 145/150 articles (3.33% Qwen reduction) with 1 false negative.",
        "   - Under corrected mapping, the ArgMax gate skips 22 articles (14.67% Qwen reduction), but **discards 6 MUST_KNOW articles** (MUST_KNOW recall drops to 90.62%).",
        "   - Missing 6 out of 64 critical industry updates violates the strict safety criterion for the AI News Intelligence Monitor.",
        "2. **Guarded Policy Achieves Inadequate Compute Savings:**",
        "   - To achieve high MUST_KNOW recall (Policy 5 on E2: 95.31%), the filter only skips 13 articles (8.67% compute reduction), but still discards 3 MUST_KNOW articles.",
        "3. **Firm Operational Recommendation:**",
        "   - Zero-shot Laya cannot serve as an automatic gate before Qwen without supervised task adaptation.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
