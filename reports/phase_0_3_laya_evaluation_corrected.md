# Phase 0.3: Laya vs. Human Gold-Standard Evaluation Report (CORRECTED)

> [!IMPORTANT]
> **CORRECTED EVALUATION ARTIFACT (October 2026)**  
> This report replaces the uncorrected analysis in `reports/phase_0_3_laya_evaluation.md`.  
> The previous evaluation suffered from a canonical probability mapping bug: it selected the raw 4-way argmax (`high`, `medium`, `low`, `ignore`) and mapped the single winner, instead of computing the true 3-way canonical argmax:  
> NOISE = P(low) + P(ignore)  
> Predicted = argmax(P(high), P(medium), P(low) + P(ignore))  
> No new Laya inference was executed; all values are computed directly from the existing raw probabilities in `evaluation/phase_0_3_predictions.jsonl`.

---

## 1. Executive Summary: Old vs. Corrected Metrics

| Metric | Old (Buggy 4-Way ArgMax) | Corrected (3-Way Canonical ArgMax) | Delta | Note |
|:---|:---:|:---:|:---:|:---|
| **Overall Accuracy** | **49.33%** (74/150) | **42.00%** (63/150) | -7.33% | 17 articles changed prediction |
| **Macro Precision** | 35.39% | 35.29% | -0.10% | Balanced across 3 classes |
| **Macro Recall** | 37.04% | 32.98% | -4.06% | Lower due to NOISE false positives |
| **Macro F1** | 0.3479 | 0.3321 | -0.0158 | |
| **Weighted Precision** | 46.58% | 44.96% | -1.62% | |
| **Weighted Recall** | 49.33% | 42.00% | -7.33% | Matches accuracy |
| **Weighted F1** | 0.4607 | 0.4229 | -0.0378 | |
| **MUST_KNOW Precision** | 58.14% | 57.14% | -1.00% | |
| **MUST_KNOW Recall** | **39.06%** | **37.50%** | -1.56% | 1 article (3812) flipped to NOISE |
| **MUST_KNOW F1** | 0.4673 | 0.4528 | -0.0145 | |
| **INTERESTING Precision** | 48.04% | 44.19% | -3.85% | |
| **INTERESTING Recall** | **72.06%** | **55.88%** | -16.18% | 11 articles flipped to NOISE |
| **INTERESTING F1** | 0.5765 | 0.4935 | -0.0830 | |
| **NOISE Precision** | 0.00% | **4.55%** | +4.55% | 1 true NOISE identified |
| **NOISE Recall** | **0.00%** | **5.56%** | +5.56% | Article 2305 identified |
| **NOISE F1** | 0.0000 | **0.0500** | +0.0500 | |

---

## 2. Prediction Distribution & Label Shift

```
Old Prediction Distribution:
  MUST_KNOW:    43 (28.7%)
  INTERESTING: 102 (68.0%)
  NOISE:         5 (3.3%)

Corrected Prediction Distribution:
  MUST_KNOW:    42 (28.0%)
  INTERESTING:  86 (57.3%)
  NOISE:        22 (14.7%)
```

Across the 150 gold-standard articles, exactly **17 articles (11.3%)** changed prediction:
- **11 articles** flipped from `INTERESTING` to `NOISE` (human ground truth: 11 `INTERESTING`)
- **5 articles** flipped from `INTERESTING` to `NOISE` (human ground truth: 4 `MUST_KNOW`, 1 `NOISE`)
- **1 article** flipped from `MUST_KNOW` to `NOISE` (human ground truth: `MUST_KNOW`, Article 3812)

### The Single Correct NOISE Prediction: Article 2305
- **Title:** *"How AI is impacting who gets to cross borders"* (Hacker News AI)
- **Human Ground Truth:** `NOISE`
- **Raw Laya Probabilities:**
  - P(high) = 0.2081
  - P(medium) = 0.3223
  - P(low) = 0.2638
  - P(ignore) = 0.2059
- **Old Evaluation:** Picked raw choice `medium` (0.3223 > 0.2638, 0.2081, 0.2059) -> mapped to `INTERESTING` (Misclassification).
- **Corrected Evaluation:** P(NOISE) = 0.2638 + 0.2059 = 0.4697 > 0.3223 -> mapped to `NOISE` (True Positive!).

---

## 3. Confusion Matrices

### Old Confusion Matrix (Affected by Bug)
```
Actual \ Pred    MUST_KNOW  INTERESTING  NOISE   Total
MUST_KNOW             25           38      1      64
INTERESTING           15           49      4      68
NOISE                  3           15      0      18
Total                 43          102      5     150
```

### Corrected Confusion Matrix (True 3-Way Canonical ArgMax)
```
Actual \ Pred    MUST_KNOW  INTERESTING  NOISE   Total
MUST_KNOW             24           34      6      64
INTERESTING           15           38     15      68
NOISE                  3           14      1      18
Total                 42           86     22     150
```

---

## 4. Confidence and Margin Statistics

- **Average Top Probability:** 0.4747
- **Average Top-Two Margin:** 0.1518
- **Low-Margin Decisions (< 0.15):** 79 articles (52.7%)
- **Low-Margin Decisions (< 0.10):** 59 articles (39.3%)

---

## 5. Architectural Implications

1. **NOISE Recall Rises Slightly, But Stays Dismal:** Correcting the probability mapping lifts NOISE recall from 0.00% to 5.56% (1 out of 18). Laya still misses 17 out of 18 true noise articles.
2. **Overall Accuracy Decreases to 42.00%:** Because ModernBERT distributes probability mass across both `low` and `ignore`, summing them causes 21 non-noise articles (6 MUST_KNOW and 15 INTERESTING) to cross into NOISE, resulting in 21 false NOISE predictions vs. only 1 true NOISE prediction (precision: 4.55%).
3. **Core Conclusion Holds Firmly:** Laya's zero-shot decision heads cannot be reliably used as an operational triage gate.
