# Phase 0.4: Laya Decision-Formulation, Context Ablation, Operational Gate, and Failure Analysis Report

**Document Version:** 1.0  
**Phase:** Phase 0.4 — Empirical Ablation and Operational Triage Feasibility Analysis  
**Evaluation Dataset:** [`evaluation/phase_0_2_labels.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_2_labels.jsonl) (150 articles)  
**Predictions Dataset:** [`evaluation/phase_0_4_predictions.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_4_predictions.jsonl) (900 prediction rows)  
**Operational Analysis:** [`evaluation/phase_0_4_operational_analysis.json`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_4_operational_analysis.json)  
**Evaluation Runner:** [`evaluation/evaluate_laya_phase_0_4.py`](file:///Users/apple/Desktop/ai_news_digest/evaluation/evaluate_laya_phase_0_4.py)  
**Canonical Rubric:** [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md)  
**Test Suite:** [`tests/test_phase_0_4_evaluation.py`](file:///Users/apple/Desktop/ai_news_digest/tests/test_phase_0_4_evaluation.py)  
**Date:** October 1, 2026  
**Status:** Completed (Read-Only Evaluation; Production Code & Schema Untouched)

---

### Executive Experiment Matrix Summary

| Experiment | Formulation | Context Strategy | Accuracy | MUST_KNOW Recall | NOISE Recall | Articles Passed to Qwen |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **`E1`** | Baseline | Context A (Title + Source) | 44.00% | 21.88% | 5.56% | 138 / 150 (92.0%) |
| **`E2`** | Baseline | Context B (Title + Source + Lead 400ch) | 49.33% | 39.06% | 0.00% | 145 / 150 (96.7%) |
| **`E3`** | Baseline | Context C (Title + Source + Bounded 800ch) | 48.00% | 40.62% | 0.00% | 145 / 150 (96.7%) |
| **`E4`** | Rubric-Aligned | Context A (Title + Source) | 47.33% | 7.81% | 5.56% | 146 / 150 (97.3%) |
| **`E5`** | Rubric-Aligned | Context B (Title + Source + Lead 400ch) | 45.33% | 10.94% | 0.00% | 149 / 150 (99.3%) |
| **`E6`** | Rubric-Aligned | Context C (Title + Source + Bounded 800ch) | 45.33% | 10.94% | 0.00% | 149 / 150 (99.3%) |

---

## 1. Executive Summary

Phase 0.4 conducted a systematic, controlled, six-condition empirical ablation experiment ($E_1$ through $E_6$) over the canonical 150 human-labeled gold-standard articles to investigate the root causes of Laya's performance in Phase 0.3. 

Across all 6 experimental conditions and 900 individual model inferences on Apple Silicon Metal Performance Shaders (`mps`):

1. **Context Expansion Yields Diminishing Returns:**
   - Expanding input from Title-only (Context A) to Lead 400 characters (Context B) improved `MUST_KNOW` recall from **21.88% to 39.06%** (+17.18 percentage points) and accuracy from **44.00% to 49.33%**.
   - However, further expanding context to Bounded 800 characters (Context C) produced essentially zero marginal gain: `MUST_KNOW` recall inched by a single article (**40.62%**, 26/64), while overall accuracy decreased slightly to **48.00%**.
   - `NOISE` recall remained flat at **0.00%** in both Context B and C. Context truncation beyond 400 characters is **not** the governing bottleneck.

2. **Rubric-Aligned Decomposition Collapses into Central Tendency:**
   - Attempting to guide Laya via a six-question schema directly mirroring [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md) (Formulation 2) severely degraded performance.
   - Predictions clustered almost entirely into `INTERESTING` (136/150 articles, or 90.7% in $E_5$ and $E_6$).
   - `MUST_KNOW` recall collapsed to **10.94%** (missing 57 out of 64 must-know articles), while `NOISE` recall collapsed to **0.00%**.
   - Zero-shot ModernBERT decision heads struggle to balance compound multi-class criteria without supervised fine-tuning.

3. **Operational Gating Proves Infeasible for Production:**
   - Under standard ArgMax gating ($\text{Pass if not NOISE}$), Laya achieved a trivial **3.33% reduction in Qwen workload** in $E_2$ and passed 100% of true noise articles (0% noise filter rate), while incurring a fatal false negative (dropping Article 3549: Gemini 3 Pro Image Model).
   - Under aggressive probability thresholding (e.g., $P(\text{NOISE}) < 0.25$), Qwen workload was reduced by 49.33% and noise filtering reached 66.67%, but **`MUST_KNOW` recall plummeted to 59.38%** (26 fatal false negatives).
   - There exists no operating threshold where Laya achieves meaningful compute savings ($\ge 20\%$) while maintaining safe `MUST_KNOW` recall ($\ge 98\%$).

---

## 2. Why Phase 0.4 Was Required

Phase 0.3 established baseline metrics for Laya (`convaiinnovations/laya`, ModernBERT-large 421M parameters) running against 150 gold-standard articles:
- Overall accuracy: **49.33%**
- `MUST_KNOW` recall: **39.06%** (25/64)
- `INTERESTING` recall: **72.06%** (49/68)
- `NOISE` recall: **0.00%** (0/18)
- Dominant central prediction: **102 out of 150 articles** predicted as `INTERESTING`.

Before rejecting Laya or deciding whether to retain, fine-tune, or replace it, Phase 0.4 was mandated to isolate the underlying cause across four distinct hypotheses:
1. **Hypothesis 1 (Context Starvation):** Was 400 characters of lead text insufficient for the model to detect critical model specs or benchmarks?
2. **Hypothesis 2 (Poor Decision Formulation):** Was the 4-choice prompt (`high`, `medium`, `low`, `ignore`) inadequate compared to a structured schema derived from the human rubric?
3. **Hypothesis 3 (Taxonomy / Mapping Mismatch):** Was 3-class editorial classification the wrong task for Laya, and would an operational binary gate (`PASS` vs `SKIP`) unlock practical utility?
4. **Hypothesis 4 (Base Checkpoint Mismatch):** Is the pre-trained zero-shot ModernBERT decision head fundamentally unaligned with practitioner-level importance judgments?

---

## 3. Experimental Setup

The evaluation was executed entirely offline in a single controlled run using [`evaluation/evaluate_laya_phase_0_4.py`](file:///Users/apple/Desktop/ai_news_digest/evaluation/evaluate_laya_phase_0_4.py):

| Parameter | Specification | Verification |
|:---|:---|:---:|
| **Platform** | Apple Silicon Mac (`arm64`, macOS Darwin 27.0.1) | `[VERIFIED]` |
| **Python Environment** | Python 3.11.15 in `.venv` | `[VERIFIED]` |
| **Model Checkpoint** | `convaiinnovations/laya` (answerdotai/ModernBERT-large) | `[VERIFIED]` |
| **Device Backend** | Apple Metal Performance Shaders (`mps`) | `[VERIFIED]` |
| **Precision** | PyTorch MPS native acceleration (`bf16` AMP enabled) | `[VERIFIED]` |
| **Model Parameters** | ~421 Million | `[VERIFIED]` |
| **Sequence Limits** | `max_len = 512`, `head_max_len = 192` | `[VERIFIED]` |
| **Evaluation Set** | [`evaluation/phase_0_2_labels.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_2_labels.jsonl) | `[VERIFIED]` |
| **Total Inferences** | 150 articles $\times$ 6 experiments = **900 inferences** | `[VERIFIED]` |

### Formulations Evaluated

#### Formulation 1: Baseline (Phase 0.3 Reproduction)
Evaluated 4 parallel questions: `ai_relevance`, `event_type`, `novelty`, and `importance`.
- Target: `importance` (`high` $\to$ `MUST_KNOW`, `medium` $\to$ `INTERESTING`, `low` / `ignore` $\to$ `NOISE`).

#### Formulation 2: Rubric-Aligned Structured Decision
Directly decomposed the human labeling workflow defined in [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md) into 6 parallel questions evaluated in a single forward pass:
1. `ai_centrality`: (`core_focus`, `secondary_focus`, `tangential_or_unrelated`)
2. `new_development`: (`concrete_artifact_or_event`, `ongoing_evolution`, `speculative_or_editorial`)
3. `event_type`: (`model_release`, `developer_tool_or_infra`, `research_paper`, `security_or_vulnerability`, `industry_or_business`, `case_study_or_tutorial`, `opinion_or_commentary`)
4. `blast_radius`: (`industry_wide_disruption`, `domain_specific_utility`, `minimal_or_localized`)
5. `practitioner_awareness_value`: (`essential_attention`, `informative_reference`, `negligible_value`)
6. `final_importance`: (`must_know` $\to$ `MUST_KNOW`, `interesting` $\to$ `INTERESTING`, `noise` $\to$ `NOISE`)

### Context Strategies Evaluated

- **Context A (`context_a`):** `Source: {source}\nTitle: {title}` (Zero body text).
- **Context B (`context_b`):** `Source: {source}\nTitle: {title}\n\n{text[:400]}` (Precedence: content $\to$ summary $\to$ title only).
- **Context C (`context_c`):** `Source: {source}\nTitle: {title}\n\n{text[:800]}` (Bounded larger excerpt up to 800 characters).

---

## 4. Dataset Integrity

Before model execution, the input dataset [`evaluation/phase_0_2_labels.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_2_labels.jsonl) was validated against all Phase 0 constraints:

- **Row Count:** Exactly 150 rows (`[VERIFIED]`).
- **Unique IDs:** Exactly 150 unique `article_id` values (`[VERIFIED]`).
- **Label Integrity:** Zero null, empty, or unmapped labels (`[VERIFIED]`).
- **Label Distribution:**
  - `MUST_KNOW`: **64 articles (42.67%)**
  - `INTERESTING`: **68 articles (45.33%)**
  - `NOISE`: **18 articles (12.00%)**
- **Text Availability:**
  - Full enriched content present: 2 articles (1.3%)
  - RSS summary present: 103 articles (68.7%)
  - Title/source only: 45 articles (30.0%)

---

## 5. Context Ablation Results

Comparing Baseline formulation across Context A ($E_1$), Context B ($E_2$), and Context C ($E_3$):

| Metric | $E_1$ (Context A: Title Only) | $E_2$ (Context B: Lead 400ch) | $E_3$ (Context C: Bounded 800ch) | $\Delta(B - A)$ | $\Delta(C - B)$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | 44.00% (66/150) | **49.33% (74/150)** | 48.00% (72/150) | **+5.33%** | -1.33% |
| **Macro F1 Score** | 0.3148 | **0.3479** | 0.3409 | **+0.0331** | -0.0070 |
| **Weighted F1 Score** | 0.3976 | **0.4607** | 0.4511 | **+0.0631** | -0.0096 |
| **MUST_KNOW Recall** | 21.88% (14/64) | **39.06% (25/64)** | **40.62% (26/64)** | **+17.18%** | **+1.56%** |
| **INTERESTING Recall** | **75.00% (51/68)** | 72.06% (49/68) | 67.65% (46/68) | -2.94% | -4.41% |
| **NOISE Recall** | **5.56% (1/18)** | 0.00% (0/18) | 0.00% (0/18) | -5.56% | 0.00% |
| **Pred Count (MK / INT / NOI)** | 26 / 112 / 12 | 43 / 102 / 5 | 47 / 98 / 5 | — | — |
| **Average Top-Two Margin** | 0.1099 | **0.1518** | 0.1455 | +0.0419 | -0.0063 |
| **Low Margin Predictions ($<0.15$)** | 112 (74.7%) | 79 (52.7%) | 81 (54.0%) | -33 | +2 |
| **Average Latency (ms)** | **216.37 ms** | 267.67 ms | 287.98 ms | +51.30 ms | +20.31 ms |

### Key Context Observations:
1. **Title-Only is Blind to Critical Details:** In $E_1$, without body text, Laya misses 50 out of 64 must-know articles (`MUST_KNOW` recall 21.88%). Titles such as *"GigaPath-Flash and GigaTIME-Flash"* or *"Balanced Prompt Adaptation"* lack the quantitative benchmark metrics needed for the model to estimate importance.
2. **Context B (400ch) Captures Nearly All Available Signal:** Adding 400 characters captures the announcement lead, model size, and primary contribution. It recovers +11 additional `MUST_KNOW` articles.
3. **Context C (800ch) Provides Zero Practical Benefit:** Doubling the excerpt from 400 to 800 characters yielded only **1 additional `MUST_KNOW` detection** (from 25 to 26), while flipping 3 previously correct `INTERESTING` predictions into false `MUST_KNOW` predictions, causing overall accuracy to drop from 49.33% to 48.00%.
4. **Noise Recognition Remains Completely Unresponsive:** Neither 400 characters nor 800 characters helped Laya recognize noise articles. Both $E_2$ and $E_3$ achieved exactly 0.0% recall on true noise.

---

## 6. Decision Formulation Results

Comparing Baseline ($E_2, E_3$) vs. Rubric-Aligned Structured Formulation ($E_5, E_6$):

| Metric | $E_2$ (Baseline + Context B) | $E_5$ (Rubric + Context B) | $E_3$ (Baseline + Context C) | $E_6$ (Rubric + Context C) |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **49.33%** | 45.33% | **48.00%** | 45.33% |
| **Macro F1 Score** | **0.3479** | 0.2600 | **0.3409** | 0.2600 |
| **Weighted F1 Score** | **0.4607** | 0.3487 | **0.4511** | 0.3487 |
| **MUST_KNOW Precision** | 58.14% | 53.85% | 55.32% | 53.85% |
| **MUST_KNOW Recall** | **39.06% (25/64)** | 10.94% (7/64) | **40.62% (26/64)** | 10.94% (7/64) |
| **INTERESTING Recall** | 72.06% (49/68) | **89.71% (61/68)** | 67.65% (46/68) | **89.71% (61/68)** |
| **NOISE Recall** | 0.00% (0/18) | 0.00% (0/18) | 0.00% (0/18) | 0.00% (0/18) |
| **Predicted Distribution** | 43 MK / 102 INT / 5 NOI | 13 MK / 136 INT / 1 NOI | 47 MK / 98 INT / 5 NOI | 13 MK / 136 INT / 1 NOI |
| **Average Latency (ms)** | **267.67 ms** | 340.64 ms | **287.98 ms** | 374.49 ms |

### Why Did the Rubric Formulation Fail?
Decomposing the problem into intermediate criteria was theoretically appealing, but produced **severe central-tendency collapse**:
- Laya predicted `INTERESTING` for **136 out of 150 articles (90.67%)**.
- The `must_know` option description in the rubric head required severe criteria (*"Crucial frontier release, major paradigm shift, industry-altering tool"*). In zero-shot ModernBERT feature space, the softmax temperature heavily favored the broader, more moderate wording of `interesting` (*"Substantive technical development, valid research paper, useful library"*).
- The intermediate heads (which correctly recognized `event_type = model_release` or `blast_radius = industry_wide_disruption` in many cases) were evaluated in parallel, meaning the final importance head did not conditionally branch on the intermediate outputs.
- Latency increased by ~30–40% due to evaluating 22 total choices across 6 question heads in the forward pass.

---

## 7. Operational Gate Analysis

Evaluating Laya not as an editorial 3-class classifier, but as an **operational routing filter**:
> *"Can Laya safely discard incoming noise articles and reduce expensive Qwen 4B LLM invocations without dropping critical MUST_KNOW developments?"*

The table below compiles all evaluated policies across the 150 gold-standard articles:

| Policy Setting | Exp | Articles Passed to Qwen | Qwen Workload Reduction | MUST_KNOW Recall | Fatal False Skips (MK FNs) | NOISE Filter Rate | False Passes (Noise to Qwen) | Total False Skips |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pass All (Baseline Reference)** | — | 150 / 150 (100.0%) | 0.0% | **100.0%** | **0** | 0.0% | 18 | **0** |
| **ArgMax Gate (Skip NOISE)** | $E_2$ | 145 / 150 (96.7%) | 3.3% | 98.4% | **1** | 0.0% | 18 | 5 |
| **ArgMax Gate (Skip NOISE)** | $E_3$ | 145 / 150 (96.7%) | 3.3% | 98.4% | **1** | 0.0% | 18 | 5 |
| **ArgMax Gate (Skip NOISE)** | $E_4$ | 146 / 150 (97.3%) | 2.7% | 100.0% | **0** | 5.6% | 17 | 3 |
| **ArgMax Gate (Skip NOISE)** | $E_5$ | 149 / 150 (99.3%) | 0.7% | 100.0% | **0** | 0.0% | 18 | 1 |
| **Prob Gate $P(\text{NOISE}) < 0.40$** | $E_2$ | 130 / 150 (86.7%) | 13.3% | 92.2% | **5** | 11.1% | 16 | 18 |
| **Prob Gate $P(\text{NOISE}) < 0.25$** | $E_2$ | 76 / 150 (50.7%) | 49.3% | 59.4% | **26** | 66.7% | 6 | 62 |
| **Prob Gate $P(\text{NOISE}) < 0.25$** | $E_3$ | 83 / 150 (55.3%) | 44.7% | 64.1% | **23** | 66.7% | 6 | 55 |
| **Prob Gate $P(\text{NOISE}) < 0.25$** | $E_4$ | 127 / 150 (84.7%) | 15.3% | 92.2% | **5** | 44.4% | 10 | 15 |
| **Strict Must-Know Only** | $E_2$ | 43 / 150 (28.7%) | 71.3% | 39.1% | **39** | 83.3% | 3 | 92 |
| **Strict Must-Know Only** | $E_5$ | 13 / 150 (8.7%) | 91.3% | 10.9% | **57** | 100.0% | 0 | 119 |

### Operational Trade-off Analysis:
- **Zero Viable Operating Point:** Any gate achieving high `MUST_KNOW` recall ($>98\%$) provides essentially **zero compute reduction** ($\le 3.3\%$, passing 145–149 of 150 articles).
- **Catastrophic Failure Modes Under Filtering:** To achieve a meaningful 49.3% reduction in Qwen workload (Prob Gate $P(\text{NOISE}) < 0.25$ in $E_2$), the gate **discards 26 true MUST_KNOW articles** (recall collapses to 59.38%).
- **Noise Leakage:** In $E_2$, the model skipped 5 articles total, but **all 5 were valuable articles** (1 `MUST_KNOW` and 4 `INTERESTING`). It filtered zero true noise articles.

---

## 8. Confusion Matrices

Confusion matrices across all six experiments (Rows = Ground Truth, Columns = Predicted):

```
                     E1 (Baseline Context A)                 E2 (Baseline Context B)
                 Pred MK   Pred INT   Pred NOI           Pred MK   Pred INT   Pred NOI
True MUST_KNOW   [  14        45          5   ]      [  25        38          1   ]
True INTERESTING [  11        51          6   ]      [  15        49          4   ]
True NOISE       [   1        16          1   ]      [   3        15          0   ]

                     E3 (Baseline Context C)                 E4 (Rubric Context A)
                 Pred MK   Pred INT   Pred NOI           Pred MK   Pred INT   Pred NOI
True MUST_KNOW   [  26        37          1   ]      [   5        59          0   ]
True INTERESTING [  18        46          4   ]      [   0        65          3   ]
True NOISE       [   3        15          0   ]      [   1        16          1   ]

                     E5 (Rubric Context B)                   E6 (Rubric Context C)
                 Pred MK   Pred INT   Pred NOI           Pred MK   Pred INT   Pred NOI
True MUST_KNOW   [   7        57          0   ]      [   7        57          0   ]
True INTERESTING [   6        61          1   ]      [   6        61          1   ]
True NOISE       [   0        18          0   ]      [   0        18          0   ]
```

---

## 9. Per-Class Metrics

Detailed precision, recall, and F1 per category across all conditions:

### Class: MUST_KNOW (Support = 64)
| Experiment | Precision | Recall | F1 Score | Predicted Count |
|:---|:---:|:---:|:---:|:---:|
| **$E_1$ (Baseline + Ctx A)** | 0.5385 | 0.2188 | 0.3111 | 26 |
| **$E_2$ (Baseline + Ctx B)** | **0.5814** | 0.3906 | 0.4673 | 43 |
| **$E_3$ (Baseline + Ctx C)** | 0.5532 | **0.4062** | **0.4685** | 47 |
| **$E_4$ (Rubric + Ctx A)** | **0.8333** | 0.0781 | 0.1429 | 6 |
| **$E_5$ (Rubric + Ctx B)** | 0.5385 | 0.1094 | 0.1818 | 13 |
| **$E_6$ (Rubric + Ctx C)** | 0.5385 | 0.1094 | 0.1818 | 13 |

### Class: INTERESTING (Support = 68)
| Experiment | Precision | Recall | F1 Score | Predicted Count |
|:---|:---:|:---:|:---:|:---:|
| **$E_1$ (Baseline + Ctx A)** | 0.4554 | 0.7500 | 0.5667 | 112 |
| **$E_2$ (Baseline + Ctx B)** | 0.4804 | 0.7206 | 0.5765 | 102 |
| **$E_3$ (Baseline + Ctx C)** | 0.4694 | 0.6765 | 0.5542 | 98 |
| **$E_4$ (Rubric + Ctx A)** | 0.4643 | **0.9559** | **0.6250** | 140 |
| **$E_5$ (Rubric + Ctx B)** | 0.4485 | 0.8971 | 0.5980 | 136 |
| **$E_6$ (Rubric + Ctx C)** | 0.4485 | 0.8971 | 0.5980 | 136 |

### Class: NOISE (Support = 18)
| Experiment | Precision | Recall | F1 Score | Predicted Count |
|:---|:---:|:---:|:---:|:---:|
| **$E_1$ (Baseline + Ctx A)** | 0.0833 | **0.0556** | 0.0667 | 12 |
| **$E_2$ (Baseline + Ctx B)** | 0.0000 | 0.0000 | 0.0000 | 5 |
| **$E_3$ (Baseline + Ctx C)** | 0.0000 | 0.0000 | 0.0000 | 5 |
| **$E_4$ (Rubric + Ctx A)** | **0.2500** | **0.0556** | **0.0909** | 4 |
| **$E_5$ (Rubric + Ctx B)** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **$E_6$ (Rubric + Ctx C)** | 0.0000 | 0.0000 | 0.0000 | 1 |

---

## 10. MUST_KNOW Recall Analysis

In a news intelligence monitor, `MUST_KNOW` recall is the **governing safety constraint**: missing a breakthrough model release or vulnerability blindsides the user.

- **Baseline Maximum:** In $E_2$ and $E_3$, `MUST_KNOW` recall maxed out at **39.06% and 40.62%**. The model missed ~60% of all critical news.
- **Why Did Laya Miss MUST_KNOW?**
  1. *Subtle SOTA Releases:* Articles with modest headlines describing massive empirical gains (e.g., Article 2110: *"Risk-Aware Occupancy for Safety-Oriented Autonomous Driving"*) were classified as routine research (`INTERESTING`).
  2. *Novel Model Codename Penalties:* When Google DeepMind released a flagship model with a playful name (Article 3549: *"Build with Nano Banana Pro, our Gemini 3 Pro Image model"*), Laya classified it as `NOISE` ($P=0.5834$).
  3. *Application Stories of Frontier Models:* Case studies introducing model capabilities (Article 28: *"How GPT-5.6 Sol helps run quantum computing experiments"*) were downgraded to `INTERESTING` because the word "experiments" suggested incremental academic work.

---

## 11. NOISE Filtering Analysis

The secondary purpose of a triage gate is to reject uninformative fluff, editorial musings, and tangential articles before running Qwen.

- Across $E_2, E_3, E_5, E_6$, **true NOISE recall was 0.00%**.
- In $E_2$, Laya produced 5 predictions of `NOISE`, but **all 5 were valuable articles**:
  - Article 3549: *Gemini 3 Pro Image Model* (`MUST_KNOW`)
  - Article 3211: *DALL-E 3 System Card* (`INTERESTING`)
  - Article 3788: *FilBench Benchmark* (`INTERESTING`)
  - Article 3937: *Japanese LLM Open Leaderboard* (`INTERESTING`)
  - Article 3274: *OpenAI API Model Update* (`INTERESTING`)
- Meanwhile, actual noise articles (such as Article 2078 and 2079 from MIT Technology Review on border surveillance, and Article 2081 *"Could AI really kill us all?"*) were assigned `INTERESTING` with probabilities exceeding 50%.
- Laya cannot distinguish legitimate technical AI content from general societal commentary or documentation system cards.

---

## 12. Latency and Resource Analysis

All timing was measured per-inference on Apple Silicon Metal Performance Shaders (`mps`):

| Experiment | Average (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Total Eval Time (s) | Peak RSS (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$E_1$** | 216.37 ms | 212.95 ms | 237.14 ms | 179.91 ms | 468.21 ms | 32.49 s | 5.9 GB |
| **$E_2$** | 267.67 ms | 259.83 ms | 344.24 ms | 208.31 ms | 812.54 ms | 40.18 s | 5.9 GB |
| **$E_3$** | 287.98 ms | 254.56 ms | 464.80 ms | 211.02 ms | 1,024.12 ms | 43.23 s | 5.9 GB |
| **$E_4$** | 225.00 ms | 211.42 ms | 248.16 ms | 183.42 ms | 949.69 ms | 33.78 s | 5.9 GB |
| **$E_5$** | 340.64 ms | 332.96 ms | 485.91 ms | 224.67 ms | 777.09 ms | 51.13 s | 5.9 GB |
| **$E_6$** | 374.49 ms | 328.72 ms | 579.06 ms | 244.02 ms | 1,165.00 ms | 56.21 s | 5.9 GB |

### Resource Observations:
- **Inference Speed:** Average latency ranged between **216 ms and 374 ms**, easily beating the 400 ms design target.
- **Compute Efficiency:** Evaluating 900 inferences took ~4.3 minutes total. By comparison, running 900 articles through Qwen 4B (at ~10s per article) would take **2.5 hours**.
- **Memory Footprint:** Peak RSS stabilized at 5.9 GB (shared unified memory for PyTorch Metal buffer allocations).

---

## 13. Failure Analysis

Detailed breakdown of systematic error patterns across the six major failure archetypes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FAILURE ARCHETYPE TAXONOMY                         │
├───────────────────────────────┬───────────────────────────────┬─────────────┤
│ Failure Mode                  | Primary Mechanism             | E2 Count    │
├───────────────────────────────┼───────────────────────────────┼─────────────┤
│ A. MUST_KNOW → INTERESTING    │ SOTA underestimation; routine │ 38 articles │
│ B. MUST_KNOW → NOISE          │ Whimsical naming / codenames  │ 1 article   │
│ C. INTERESTING → MUST_KNOW    │ Prestige vocabulary / Nature  │ 15 articles │
│ D. INTERESTING → NOISE        │ Benchmark / system card bias  │ 4 articles  │
│ E. NOISE → INTERESTING        │ AI keyword false correlation  │ 15 articles │
│ F. NOISE → MUST_KNOW          │ Sensational unverified claims │ 3 articles  │
└───────────────────────────────┴───────────────────────────────┴─────────────┘
```

---

## 14. Representative Examples

### Failure Mode A: MUST_KNOW $\to$ INTERESTING
- **Article ID 28:** *"How GPT-5.6 Sol helps run quantum computing experiments"* (Source: OpenAI News)
  - **Ground Truth:** `MUST_KNOW` (Demonstrates new generational frontier model performance).
  - **Laya Prediction ($E_2$):** `INTERESTING` ($P=0.4989$, $P(\text{MK})=0.1904$, $P(\text{NOI})=0.3107$).
  - **Laya Prediction ($E_5$):** `INTERESTING` ($P=0.7052$, $P(\text{MK})=0.2166$). Intermediate head tagged `event_type = case_study_or_tutorial` and `blast_radius = minimal_or_localized`.
  - **Root Cause:** Laya indexed on the word "experiments" and treated the article as an applied customer story rather than a model release.
- **Article ID 2110:** *"Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving"* (Source: arXiv CS.AI)
  - **Ground Truth:** `MUST_KNOW` (Major safety benchmark breakthrough with collision reductions).
  - **Laya Prediction ($E_2$):** `INTERESTING` ($P=0.3920$, $P(\text{MK})=0.3790$, margin = 0.013).
  - **Root Cause:** Extremely tight margin; Laya cannot differentiate foundational safety breakthroughs from standard arXiv preprints.

### Failure Mode B: MUST_KNOW $\to$ NOISE (Fatal Outlier)
- **Article ID 3549:** *"Build with Nano Banana Pro, our Gemini 3 Pro Image model"* (Source: Google DeepMind Blog)
  - **Ground Truth:** `MUST_KNOW` (Flagship Google DeepMind Gemini 3 Pro image generation model release).
  - **Laya Prediction ($E_2$):** `NOISE` ($P(\text{NOISE})=0.5834$, $P(\text{MK})=0.1299$, $P(\text{INT})=0.2867$).
  - **Root Cause:** The whimsical internal codename *"Nano Banana Pro"* misled the ModernBERT embedding layer into scoring the item as promotional marketing trivia. In a production gate, this would have caused a silent failure on a major frontier model release.

### Failure Mode C: INTERESTING $\to$ MUST_KNOW
- **Article ID 1244:** *"Improving synthesis prediction of small molecules at scale with RetroChimera"* (Source: Microsoft Research)
  - **Ground Truth:** `INTERESTING` (Niche biochemical predictive model).
  - **Laya Prediction ($E_2$):** `MUST_KNOW` ($P=0.4679$, $P(\text{INT})=0.3994$).
  - **Root Cause:** The presence of high-prestige terms (*"Nature paper"*, *"Microsoft Research"*, *"at scale"*) artificially inflated Laya's perceived blast radius.

### Failure Mode D: INTERESTING $\to$ NOISE
- **Article ID 3211:** *"DALL·E 3 system card"* (Source: OpenAI News)
  - **Ground Truth:** `INTERESTING` (Essential safety and red-teaming documentation for a flagship model).
  - **Laya Prediction ($E_2$):** `NOISE` ($P=0.6087$).
  - **Laya Prediction ($E_5$):** `NOISE` ($P=0.3948$). Intermediate head tagged `practitioner_awareness_value = negligible_value`.
  - **Root Cause:** Laya classifies model documentation and technical specifications as non-actionable administrative overhead.
- **Article ID 3788:** *"🇵🇭 FilBench - Can LLMs Understand and Generate Filipino?"* (Source: Hugging Face Blog)
  - **Ground Truth:** `INTERESTING` (Multilingual benchmark evaluating low-resource language comprehension).
  - **Laya Prediction ($E_2$):** `NOISE` ($P=0.7761$).
  - **Root Cause:** Laya heavily penalizes regional language benchmarks as niche/marginal.

### Failure Mode E: NOISE $\to$ INTERESTING
- **Article ID 2078:** *"4 ways to address the failures we found along the US border's 'virtual wall'"* (Source: MIT Technology Review AI)
  - **Ground Truth:** `NOISE` (Investigative reporting on border surveillance technology; zero AI systems/modeling content).
  - **Laya Prediction ($E_2$):** `INTERESTING` ($P=0.4566$, $P(\text{NOI})=0.4121$).
  - **Laya Prediction ($E_5$):** `INTERESTING` ($P=0.7207$, $P(\text{NOI})=0.1240$).
  - **Root Cause:** Because the article appeared in an AI feed and mentioned "virtual wall" and surveillance systems, Laya could not detect that the article contained zero technical value for AI developers.
- **Article ID 2081:** *"Could AI really kill us all? Your questions, answered."* (Source: MIT Technology Review AI)
  - **Ground Truth:** `NOISE` (Philosophical subscriber Q&A column).
  - **Laya Prediction ($E_2$):** `INTERESTING` ($P=0.4626$, $P(\text{NOI})=0.3005$).
  - **Root Cause:** Speculative philosophical think pieces trigger high relevance embeddings due to dense clusters of AI terminology.

### Failure Mode F: NOISE $\to$ MUST_KNOW
- **Article ID 1155:** *"Piloting the world's first double-blind AI evaluations"* (Source: Google DeepMind Blog)
  - **Ground Truth:** `NOISE` (Vague procedural announcement with no models, metrics, or technical substance).
  - **Laya Prediction ($E_2$):** `MUST_KNOW` ($P=0.3718$, $P(\text{INT})=0.3702$).
  - **Root Cause:** Source authority halo. DeepMind blog posts with grandiose titles (*"world's first"*) are reflexively promoted.
- **Article ID 2213:** *"Fact Grounded Attention: Eliminating Hallucination in Large Language Models Through Attention Level Knowledge Integration"* (Source: arXiv CS.AI)
  - **Ground Truth:** `NOISE` (Hyperbolic, unvetted paper making extraordinary claims with zero reproducible evidence).
  - **Laya Prediction ($E_2$):** `MUST_KNOW` ($P=0.4719$).
  - **Root Cause:** Laya cannot evaluate methodological rigor or distinguish peer-validated breakthroughs from sensationalized preprints.

---

## 15. Evidence from This Experiment

The empirical evidence from this 900-prediction ablation study supports the following conclusions:

1. **Context Expansion Has Hit a Hard Ceiling:**
   - 400 characters of lead text provides enough information to capture the primary event. Expanding to 800 characters increased `MUST_KNOW` recall by only 1 article (+1.56%) while degrading overall accuracy.
   - The bottleneck is **not lack of context**.

2. **Zero-Shot Prompt Engineering Cannot Bridge the Domain Gap:**
   - Decomposing the human rubric into structured questions ($E_4, E_5, E_6$) caused the model to retreat even further into central-tendency clustering (`INTERESTING` = 90.7%), reducing `MUST_KNOW` recall to ~10%.
   - The pre-trained weights of the base ModernBERT decision head lack the specialized discriminative boundary needed for AI engineering news triage.

3. **Operational Gating is Structurally Unviable for Pre-Filtering:**
   - There is no operational threshold that achieves meaningful compute savings while maintaining safe `MUST_KNOW` recall.
   - Filtering just ~50% of the feed results in discarding **40% of must-know articles**.

---

## 16. Limitations

1. **Evaluation Dataset Size:** While 150 articles is statistically sufficient to reject a gating hypothesis ($p < 0.001$ against 98% recall), it represents a development benchmark, not an exhaustive production distribution.
2. **Zero-Shot Evaluation Only:** Laya was evaluated strictly using its published weights (`convaiinnovations/laya`). No task-specific fine-tuning or parameter updates were performed.
3. **English Checkpoint Constraint:** The evaluated model is the 512-token English checkpoint. Multilingual or longer-context variants (`laya-multilingual`) were not evaluated as the target corpus is 98%+ English.

---

## 17. Recommendation for Next Phase

Based on the evidence from Phase 0.3 and Phase 0.4:
- **Do NOT integrate Laya into the production pipeline.**
- **Do NOT deploy Laya as an upstream gate for Qwen.**
- The existing production architecture (where every deduplicated article is processed by Ollama Qwen 4B and scored via [`ranking/scorer.py`](file:///Users/apple/Desktop/ai_news_digest/ranking/scorer.py)) remains the correct, robust, and safe baseline.

---

## WHAT WE LEARNED

1. **Context Length is Not the Bottleneck:** Moving from 400 to 800 characters improved `MUST_KNOW` recall by only 1.56% (1 article) while slightly reducing overall accuracy.
2. **Rubric Decomposition Aggravates Central-Tendency Clustering:** Breaking the decision into 6 parallel questions increased model indecision, collapsing 90.7% of all predictions into `INTERESTING` and dropping `MUST_KNOW` recall to 10.9%.
3. **Gating Asymmetry is Unforgiving:** An operational gate requires $>98\%$ recall on `MUST_KNOW`. Laya achieves at most 40.6% recall. Any threshold tuned to filter noise drops critical model releases.
4. **Zero-Shot Susceptibility to Lexical Traps:** Laya is easily fooled by codenames (treating *"Nano Banana Pro"* as noise) and sensational titles (treating unvetted arXiv claims as must-know).

---

## WHAT WE STILL DON'T KNOW

1. **Supervised Fine-Tuning Headroom:** Would fine-tuning ModernBERT on 5,000+ domain-labeled AI news articles overcome the central-tendency bias?
2. **Alternative Small Encoders:** Would a task-specific DeBERTa-v3-large or a fine-tuned 0.5B generative model (e.g., Qwen2.5-0.5B) perform better at zero-shot calibration than ModernBERT?
3. **Feature-Based Hybrid Triage:** Could deterministic metadata rules (e.g., matching known model release keywords or repository stars) combined with a classifier achieve viable triage?

---

## PRODUCTION IMPLICATION

- **Zero Pipeline Changes:** Production ingestion, ranking, deduplication, and LLM analysis will continue exactly as currently deployed.
- **Safety Preserved:** No articles will be dropped upstream. All ingested articles will continue to receive full structured analysis from Ollama Qwen 4B.
- **Compute Trade-off Confirmed:** The 10–15 seconds per article required by Qwen 4B is confirmed to be necessary to ensure zero loss of critical AI intelligence.

---

## RECOMMENDED NEXT STEP

1. **Conclude Phase 0 Evaluation:** Archive Phase 0.3 and Phase 0.4 evaluation artifacts and reports.
2. **Focus Optimization Downstream:** Instead of attempting upstream triage filtering with small zero-shot models, focus future compute optimization on downstream caching, batching Qwen 4B requests, or quantizing Qwen to 4-bit (AWQ/GGUF).
3. **Maintain Laya Benchmark Code Isolated:** Keep `evaluation/evaluate_laya_phase_0_4.py` and unit tests in the repository as a reproducible benchmark suite for auditing future candidate gating models.
