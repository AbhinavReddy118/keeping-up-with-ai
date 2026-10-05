# Phase 0.4: Decision-Formulation, Context Ablation, and Operational Gate Evaluation (CORRECTED)

> [!IMPORTANT]
> **CORRECTED EVALUATION REPORT (October 2026)**  
> This report documents the corrected evaluation results for Phase 0.4 following the canonical probability mapping fix.  
>   
> **Correction Summary:**  
> - For Formulation 1 (Baseline: E1, E2, E3), the canonical class probabilities are now properly evaluated via 3-way argmax:  
>   argmax(P(high), P(medium), P(low) + P(ignore))  
> - For Formulation 2 (Rubric-Aligned: E4, E5, E6), the model already evaluated 3 native choices (`must_know`, `interesting`, `noise`), so argmax predictions were mathematically unaffected.  
> - **CRITICAL NOTE ON E4, E5, E6:** As established during the audit, Formulation 2 was executed as **parallel, non-conditioned heads** rather than sequential structured reasoning. The results for E4, E5, and E6 are preserved here strictly for historical integrity and traceability, but are **invalid as an evaluation of sequential structured reasoning**.  
> - **Zero Laya Inferences Performed:** All calculations use the raw probabilities in `evaluation/phase_0_4_predictions.jsonl`.

---

## 1. Corrected Experiment Summary Matrix (E1 – E6)

| Exp | Formulation | Context | Corrected Acc (Old) | Corrected MK Rec (Old) | Corrected NOISE Rec (Old) | Corrected Macro F1 (Old) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **E1** | Baseline | Context A (Title+Source) | **34.67%** (44.00%) | **20.31%** (21.88%) | **33.33%** (5.56%) | **0.3097** (0.3148) | `CORRECTED` |
| **E2** | Baseline | Context B (Lead 400ch) | **42.00%** (49.33%) | **37.50%** (39.06%) | **5.56%** (0.00%) | **0.3321** (0.3479) | `CORRECTED` |
| **E3** | Baseline | Context C (Bounded 800ch) | **40.67%** (48.00%) | **39.06%** (40.62%) | **5.56%** (0.00%) | **0.3237** (0.3409) | `CORRECTED` |
| **E4** | Rubric | Context A (Title+Source) | **47.33%** (47.33%) | **7.81%** (7.81%) | **5.56%** (5.56%) | **0.2863** (0.2863) | `INVALID FORMULATION (Traceability Only)` |
| **E5** | Rubric | Context B (Lead 400ch) | **45.33%** (45.33%) | **10.94%** (10.94%) | **0.00%** (0.00%) | **0.2600** (0.2600) | `INVALID FORMULATION (Traceability Only)` |
| **E6** | Rubric | Context C (Bounded 800ch) | **45.33%** (45.33%) | **10.94%** (10.94%) | **0.00%** (0.00%) | **0.2600** (0.2600) | `INVALID FORMULATION (Traceability Only)` |

---

## 2. Detailed Performance Breakdown: E1, E2, E3

### Experiment E1: Baseline + Context A (Title + Source)
- **Accuracy:** 34.67% (52/150) — *Old: 44.00%*
- **MUST_KNOW:** Precision 52.00%, Recall 20.31% (13/64), F1 0.2921
- **INTERESTING:** Precision 39.76%, Recall 48.53% (33/68), F1 0.4371
- **NOISE:** Precision 14.29%, Recall 33.33% (6/18), F1 0.2000
- **Prediction Distribution:** MUST_KNOW: 25 | INTERESTING: 83 | NOISE: 42
- **Confusion Matrix:**
```
Actual \ Pred    MUST_KNOW  INTERESTING  NOISE   Total
MUST_KNOW             13           39     12      64
INTERESTING           11           33     24      68
NOISE                  1           11      6      18
Total                 25           83     42     150
```

### Experiment E2: Baseline + Context B (Title + Source + Lead 400 chars)
- **Accuracy:** 42.00% (63/150) — *Old: 49.33%*
- **MUST_KNOW:** Precision 57.14%, Recall 37.50% (24/64), F1 0.4528
- **INTERESTING:** Precision 44.19%, Recall 55.88% (38/68), F1 0.4935
- **NOISE:** Precision 4.55%, Recall 5.56% (1/18), F1 0.0500
- **Prediction Distribution:** MUST_KNOW: 42 | INTERESTING: 86 | NOISE: 22
- **Confusion Matrix:**
```
Actual \ Pred    MUST_KNOW  INTERESTING  NOISE   Total
MUST_KNOW             24           34      6      64
INTERESTING           15           38     15      68
NOISE                  3           14      1      18
Total                 42           86     22     150
```

### Experiment E3: Baseline + Context C (Title + Source + Bounded 800 chars)
- **Accuracy:** 40.67% (61/150) — *Old: 48.00%*
- **MUST_KNOW:** Precision 54.35%, Recall 39.06% (25/64), F1 0.4545
- **INTERESTING:** Precision 42.68%, Recall 51.47% (35/68), F1 0.4667
- **NOISE:** Precision 4.55%, Recall 5.56% (1/18), F1 0.0500
- **Prediction Distribution:** MUST_KNOW: 46 | INTERESTING: 82 | NOISE: 22
- **Confusion Matrix:**
```
Actual \ Pred    MUST_KNOW  INTERESTING  NOISE   Total
MUST_KNOW             25           33      6      64
INTERESTING           18           35     15      68
NOISE                  3           14      1      18
Total                 46           82     22     150
```

---

## 3. Corrected Operational Policy Analysis

The operational gating feasibility analysis was recomputed across all policies for E1, E2, and E3 using corrected canonical predictions:

| Policy | Articles Passed | Qwen Reduction | MUST_KNOW Recall | MK False Negatives | NOISE Filtered | NOISE Filter Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1: ArgMax Gate (Skip NOISE)** | 108 / 150 (72.0%) | 28.0% | **81.25%** (52/64) | **12** | 6/18 | 33.33% |
| **E1: Strict Must-Know Only** | 25 / 150 (16.67%) | 83.33% | **20.31%** (13/64) | **51** | 17/18 | 94.44% |
| **E1: Prob Gate P(NOISE) < 0.40** | 102 / 150 (68.0%) | 32.0% | **81.25%** (52/64) | **12** | 8/18 | 44.44% |
| **E1: Prob Gate P(NOISE) < 0.25** | 22 / 150 (14.67%) | 85.33% | **15.62%** (10/64) | **54** | 17/18 | 94.44% |
| **E1: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)** | 115 / 150 (76.67%) | 23.33% | **87.5%** (56/64) | **8** | 6/18 | 33.33% |
| **E2: ArgMax Gate (Skip NOISE)** | 128 / 150 (85.33%) | 14.67% | **90.62%** (58/64) | **6** | 1/18 | 5.56% |
| **E2: Strict Must-Know Only** | 42 / 150 (28.0%) | 72.0% | **37.5%** (24/64) | **40** | 15/18 | 83.33% |
| **E2: Prob Gate P(NOISE) < 0.40** | 130 / 150 (86.67%) | 13.33% | **92.19%** (59/64) | **5** | 2/18 | 11.11% |
| **E2: Prob Gate P(NOISE) < 0.25** | 76 / 150 (50.67%) | 49.33% | **59.38%** (38/64) | **26** | 12/18 | 66.67% |
| **E2: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)** | 137 / 150 (91.33%) | 8.67% | **95.31%** (61/64) | **3** | 0/18 | 0.0% |
| **E3: ArgMax Gate (Skip NOISE)** | 128 / 150 (85.33%) | 14.67% | **90.62%** (58/64) | **6** | 1/18 | 5.56% |
| **E3: Strict Must-Know Only** | 46 / 150 (30.67%) | 69.33% | **39.06%** (25/64) | **39** | 15/18 | 83.33% |
| **E3: Prob Gate P(NOISE) < 0.40** | 130 / 150 (86.67%) | 13.33% | **92.19%** (59/64) | **5** | 2/18 | 11.11% |
| **E3: Prob Gate P(NOISE) < 0.25** | 83 / 150 (55.33%) | 44.67% | **64.06%** (41/64) | **23** | 12/18 | 66.67% |
| **E3: High-Recall Guarded (P(MK)>=0.20 or pred!=NOISE)** | 137 / 150 (91.33%) | 8.67% | **95.31%** (61/64) | **3** | 0/18 | 0.0% |
| **Reference Baseline: Pass All to Qwen (No Filter)** | 150 / 150 (100.0%) | 0.0% | **100.0%** (64/64) | **0** | 0/18 | 0.0% |

---

## 4. Key Operational Findings

1. **ArgMax Gating on E2 Discards 6 MUST_KNOW Articles:**
   - Under the old mapping, the ArgMax gate passed 145/150 articles (3.33% Qwen reduction) with 1 false negative.
   - Under corrected mapping, the ArgMax gate skips 22 articles (14.67% Qwen reduction), but **discards 6 MUST_KNOW articles** (MUST_KNOW recall drops to 90.62%).
   - Missing 6 out of 64 critical industry updates violates the strict safety criterion for the AI News Intelligence Monitor.
2. **Guarded Policy Achieves Inadequate Compute Savings:**
   - To achieve high MUST_KNOW recall (Policy 5 on E2: 95.31%), the filter only skips 13 articles (8.67% compute reduction), but still discards 3 MUST_KNOW articles.
3. **Firm Operational Recommendation:**
   - Zero-shot Laya cannot serve as an automatic gate before Qwen without supervised task adaptation.
