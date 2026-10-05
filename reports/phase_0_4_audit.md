# Phase 0.4 Error & Result Integrity Audit Report
**Diagnostic Analysis of Laya Ablation Experiments E1–E6**

- **Document Version:** 1.0
- **Audit Date:** October 2, 2026
- **Status:** COMPLETED (Diagnostic Audit Only — Read-Only Inspection)
- **Primary Audited Artifacts:**
  - [`evaluation/evaluate_laya_phase_0_4.py`](file:///Users/apple/Desktop/ai_news_digest/evaluation/evaluate_laya_phase_0_4.py)
  - [`evaluation/phase_0_4_predictions.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_4_predictions.jsonl)
  - [`evaluation/phase_0_4_operational_analysis.json`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_4_operational_analysis.json)
  - [`reports/phase_0_4_laya_ablation.md`](file:///Users/apple/Desktop/ai_news_digest/reports/phase_0_4_laya_ablation.md)
  - [`tests/test_phase_0_4_evaluation.py`](file:///Users/apple/Desktop/ai_news_digest/tests/test_phase_0_4_evaluation.py)
  - [`evaluation/phase_0_2_labels.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_2_labels.jsonl)
  - [`evaluation/phase_0_3_predictions.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_3_predictions.jsonl)
  - Installed Laya Package: `.venv/lib/python3.11/site-packages/laya/` (`v0.3.20`)

---

## 1. Executive Summary

This audit conducted an exhaustive, code-level and data-level investigation into the experimental results reported in Phase 0.4 across conditions E1–E6. 

The audit evaluated whether the reported anomalies — including 0.00% `NOISE` recall in E2, E3, E5, E6, an extreme 90.7% `INTERESTING` prediction bias in E5/E6, low `MUST_KNOW` recall (7.8%–40.6%), and poor operational gating trade-offs — were caused by genuine model limitations, implementation bugs, evaluation script errors, or prompt design flaws.

### Key Audit Conclusions

1. **The reported results are PARTIALLY INVALID.** The experiment was compromised by two distinct implementation bugs and one major data/design artifact:
   - **ArgMax Inconsistency Bug in E1–E3:** In Formulation 1 (Baseline), the evaluation script determined the predicted canonical label by taking the 4-way argmax (`raw_choice` from `['high', 'medium', 'low', 'ignore']`) rather than the argmax of the consolidated 3-class probabilities (`P(MUST_KNOW)`, `P(INTERESTING)`, `P(NOISE)` where `P(NOISE) = P(low) + P(ignore)`). Because the `NOISE` probability mass was split across two distinct choices (`low` and `ignore`), neither individually beat `medium`. This artificially depressed `NOISE` recall in E1, E2, and E3. In E2 and E3, 17 articles had $P(\text{NOISE}) > P(\text{INTERESTING}) > P(\text{MUST_KNOW})$, but were mislabeled as `INTERESTING`.
   - **Context C Ablation Invalidation:** In 120 out of the 150 articles (80.0%), Context C was **100% identical** to Context B. This occurred because only 2 out of 150 articles in the gold-standard dataset have enriched `content`, 45 articles have no body text at all, and 75 articles have RSS summaries shorter than 400 characters (`body[:400] == body[:800]`). The conclusion that "expanding context beyond 400 characters provides diminishing returns" was largely evaluating duplicate strings on 80% of the benchmark.
   - **Rubric Structured Formulation Collapse (E4–E6):** Asking 6 questions in a single `Agent.predict()` call caused all 6 questions to execute as independent, parallel batch rows inside ModernBERT without sequential conditioning or rule-based synthesis. The target question (`final_importance`) was phrased with an extraordinarily high threshold for `must_know` and a broad umbrella for `interesting`, causing the model to collapse onto `interesting` for 136/150 articles (90.7%).
   - **Confidence Field Mismatch:** In `phase_0_4_predictions.jsonl`, the `confidence` field recorded Normalized Shannon Entropy ($1 - H(p)/\log k$) rather than calibrated top-1 probability mass ($\max(p)$ / `answer_confidence`), whereas `phase_0_4_operational_analysis.json` computed `average_confidence` from $\max(p)$.
2. **What Was Correct:**
   - E2 reproduced Phase 0.3 with 100.0% precision (150/150 identical predictions, 0.0 max probability difference).
   - Given the predictions as logged, all accuracy, macro/weighted F1, confusion matrices, and operational gate calculations in `evaluation/phase_0_4_operational_analysis.json` are 100% mathematically correct.
   - Zero sequence truncation occurred at the 512-token limit (max sequence was 437 tokens).

---

## 2. Confirmed Errors

### Error 1: 4-Way Raw ArgMax vs. 3-Way Canonical Probability ArgMax Inconsistency
- **Classification:** D (Output parsing/mapping problem) & E (Experiment configuration bug)
- **Affected Experiments:** E1, E2, E3
- **Mechanism:** In `evaluation/evaluate_laya_phase_0_4.py`, lines 600–607:
  ```python
  raw_choice = target_ans.get("choice", "")
  raw_probs = target_ans.get("probabilities", {})
  mapped_label, canonical_probs = map_prediction_to_canonical(
      formulation, raw_choice, raw_probs
  )
  ```
  In `map_prediction_to_canonical` (lines 249–262):
  ```python
  mapped_label = BASELINE_IMPORTANCE_MAP[clean_choice]
  p_mk = float(raw_probabilities.get("high", 0.0))
  p_int = float(raw_probabilities.get("medium", 0.0))
  p_noi = float(raw_probabilities.get("low", 0.0)) + float(raw_probabilities.get("ignore", 0.0))
  ```
  `clean_choice` is the argmax over 4 classes: `argmax(['ignore', 'low', 'medium', 'high'])`. But `p_noi` is the sum of `low` and `ignore`. When $P(\text{low}) + P(\text{ignore}) > P(\text{medium}) > \max(P(\text{low}), P(\text{ignore}))$, the 4-way argmax picks `medium` (`INTERESTING`), while the 3-way probability argmax is `NOISE`.
- **Empirical Impact:**
  - In E1: **30 articles** had `mapped_label != argmax(canonical_probs)`.
  - In E2: **17 articles** had `mapped_label != argmax(canonical_probs)`.
  - In E3: **17 articles** had `mapped_label != argmax(canonical_probs)`.
  - Example: Article ID 2305 (Human = `NOISE`). Raw probabilities: `ignore`: 0.2059, `low`: 0.2638, `medium`: 0.3223, `high`: 0.2081. Canonical probabilities: `NOISE`: 0.4697, `INTERESTING`: 0.3223, `MUST_KNOW`: 0.2081. Canonical $P(\text{NOISE})$ had a massive 14.7% margin over `INTERESTING`. Yet because `medium` (0.3223) beat `low` (0.2638) individually, it was recorded as `INTERESTING`.

### Error 2: Missing Enriched Content Causing Ineffective Context C Ablation
- **Classification:** B (Input/context construction problem)
- **Affected Experiments:** E3, E6
- **Mechanism:** The evaluation dataset `evaluation/phase_0_2_labels.jsonl` contains:
  - Total records: 150
  - Articles with enriched `content`: **2** (IDs 2081, 2301)
  - Articles with `summary` only: **103**
  - Articles with neither `content` nor `summary`: **45**
  Because 45 articles have no body text, and 75 articles have RSS summaries shorter than 400 characters, `body[:400]` and `body[:800]` evaluate to identical strings.
- **Empirical Impact:**
  - Context A == Context B for 45 articles (30.0%).
  - Context B == Context C for **120 articles (80.0%)**.
  - Only 30 articles (20.0%) actually received more characters in Context C than Context B. Thus, the ablation comparing Context B and Context C was invalid as a general 800-character context test.

### Error 3: Parallel Non-Conditioned Execution of Rubric Decomposition
- **Classification:** C (Laya API misuse / architectural misunderstanding) & B (Input prompt construction)
- **Affected Experiments:** E4, E5, E6
- **Mechanism:** In Formulation 2, six questions (`ai_centrality`, `new_development`, `event_type`, `blast_radius`, `practitioner_awareness_value`, `final_importance`) were passed simultaneously to `agent.predict(state_text, q_schema)`.
  In Laya's architecture, `Agent.predict()` collates questions into batch rows of a single parallel forward pass. There is **no autoregressive generation, no sequential conditioning, and no rule synthesis**. Question 6 (`final_importance`) had zero visibility into questions 1–5.
- **Empirical Impact:** The target question was scored zero-shot directly against the article text. Because the criterion for `interesting` was broad ("Substantive technical development, valid research paper, useful library, or instructive case study") and `must_know` was narrow ("Crucial frontier release, major paradigm shift, industry-altering tool, or critical incident"), 90.7% of articles collapsed into `interesting`.

### Error 4: Semantic Ambiguity in Saved `confidence` Field
- **Classification:** C (Laya API misuse) & D (Output parsing problem)
- **Affected Experiments:** E1–E6
- **Mechanism:** In `evaluate_laya_phase_0_4.py`, line 602:
  `conf = target_ans.get("confidence", 0.0)`
  In Laya, `target_ans["confidence"]` is Normalized Shannon Entropy: $1 - H(p)/\log(k)$.
  Calibrated answer probability is stored under `target_ans["answer_confidence"]` ($\max(p)$).
  In `phase_0_4_predictions.jsonl`, `confidence` stored entropy (e.g., 0.11), while in `phase_0_4_operational_analysis.json`, `average_confidence` was computed from $\max(p)$ (e.g., 0.45). This causes users inspecting `phase_0_4_predictions.jsonl` to believe the model had ~10% probability on its choice.

---

## 3. Suspected Issues

1. **Extreme Asymmetry in `final_importance` Prompt Criteria:**
   - In E4–E6, the prompt definition for `must_know` demanded "major paradigm shift" or "critical incident", which is a higher bar than human labeling standards (where any major model release from a frontier lab was labeled `MUST_KNOW`). This prompted ModernBERT to suppress `must_know` probabilities (averaging 0.30 vs 0.58 for `interesting`).
2. **Subjectivity & Domain Skepticism in Gold-Standard Labels:**
   - In 3 out of 18 human `NOISE` articles, Laya predicted `MUST_KNOW` (IDs 1155, 2213, 3824). In ID 2213 ("Fact Grounded Attention: Eliminating Hallucination in LLMs"), the human annotator marked it `NOISE` due to skepticism of the paper's claims ("Highly sensational claims... insufficient evidence"). A 150M parameter classifier cannot evaluate empirical rigor or perform peer review on academic abstracts.

---

## 4. Results That Are Correct Despite Looking Suspicious

1. **Phase 0.3 Parity:**
   - Phase 0.4 E2 reproduced Phase 0.3 exactly: Accuracy 49.33% (74/150), MUST_KNOW recall 39.06% (25/64), NOISE recall 0.00% (0/18), INTERESTING recall 72.06% (49/68). Max probability difference across all 150 articles was 0.0000.
2. **Operational Gate Arithmetic:**
   - The reported operational gate numbers in `reports/phase_0_4_laya_ablation.md` are 100% verified:
     - E2 ArgMax Gate: 145/150 passed (96.67%), 3.33% Qwen reduction, 98.44% MUST_KNOW recall, 0.00% NOISE filter rate.
     - E2 P(NOISE) < 0.25: 76/150 passed (50.67%), 49.33% Qwen reduction, 59.38% MUST_KNOW recall (26 MUST_KNOW false negatives), 66.67% NOISE filter rate.
3. **Metric Calculation Correctness:**
   - All precision, recall, F1, macro, weighted, and confusion matrix numbers in `evaluation/phase_0_4_operational_analysis.json` match manual recomputation from `evaluation/phase_0_4_predictions.jsonl` with zero discrepancies.
4. **Token Limits & Truncation:**
   - Zero token truncation occurred at the 512-token limit. In Context C, maximum sequence length was 378 tokens for Baseline and 437 tokens for Rubric.
5. **Latency Measurements:**
   - Latency was measured around `agent.predict()`. It properly includes tokenization, batch collation, MPS tensor transfer, one forward pass, CPU sync, and logit decoding. E4–E6 latencies (225–374 ms) are higher than E1–E3 (216–287 ms) because batch size was 6 questions instead of 4, triggering FP16 AMP and larger matrix operations on MPS.

---

## 5. Laya API Verification

We inspected the installed source code of `laya==0.3.20` in `.venv/lib/python3.11/site-packages/laya/`:

### Detailed Verification of Output Semantics

1. **`choice`:** Key of the criterion corresponding to $\operatorname{argmax}(z / T)$. In multi-choice questions, it is strictly the 1-best argmax over the specific option keys provided in that question definition.
2. **`probabilities`:** Softmax distribution over option logits scaled by temperature:
   $$p = \frac{\exp((z - \max(z)) / T)}{\sum \exp((z - \max(z)) / T)}$$
   These sum to 1.0 across the choices of that specific question.
3. **`confidence` vs. `answer_confidence`:**
   - `confidence`: Normalized Shannon Entropy: $1 - \frac{H(p)}{\log(k)}$. Measures how peaked the distribution is, on $[0, 1]$.
   - `answer_confidence`: Calibrated probability mass on the reported choice: $\max(p)$. This is the quantity fitted by temperature scaling.
4. **`score`:** For question type `score`, the expectation $\sum i \cdot p_i$.
5. **`noul`:** For binary statements, returns $P(\text{true})$.
6. **Margins:** Not computed natively by Laya; computed in evaluation scripts as $P_{\text{top1}} - P_{\text{top2}}$.

### Answers to 9 Mandatory API Verification Questions

| # | Question | Finding | Evidence / Citation |
|:---:|:---|:---|:---|
| 1 | Are probabilities actually class probabilities? | **Yes** | Computed via `softmax(logits / T)` over question options (`laya/agent.py:279-281`). |
| 2 | Are they calibrated probabilities? | **Yes, under temperature scaling** | Temperature parameters are loaded from model config and clamped (`laya/agent.py:112-114`). |
| 3 | Is confidence computed correctly? | **Yes, but dual definition** | `confidence` is entropy; `answer_confidence` is $\max(p)$. The evaluation script used the former. |
| 4 | Is the top-two margin calculation correct? | **Yes** | Calculated as `sorted_probs[0] - sorted_probs[1]` (`evaluate_laya_phase_0_4.py:330-333`). |
| 5 | Does Laya return probabilities in assumed order? | **Yes** | Keys maintain Python dict insertion order matching criteria (`laya/common.py:46-52`). |
| 6 | Is the mapping valid for the questions used? | **Partially** | Valid for 1-to-1, but invalid for 4-way to 3-way where NOISE was split into `low` and `ignore`. |
| 7 | Are structured questions interpreted correctly? | **Yes, syntactically** | Each question was tokenized and scored as a valid choice question. |
| 8 | Does final decision come from intended head? | **Yes** | Extracted from `answers["final_importance"]` (`evaluate_laya_phase_0_4.py:599`). |
| 9 | Do intermediate decisions accidentally influence final? | **No** | Questions are evaluated in parallel in one forward pass; rows have zero cross-attention. |

---

## 6. Input/Context Verification

### Sample Audit Table (10 Representative Articles)

| Article ID | Source | Human Label | Content Chars | Summary Chars | Context A (chars/tok) | Context B (chars/tok) | Context C (chars/tok) | B == C? | Text Source Type |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2081** | MIT Tech Review | NOISE | 10,862 | 317 | 88 ch / 21 tok | 490 ch / 114 tok | 890 ch / 201 tok | False | `content` |
| **2301** | Hacker News AI | INTERESTING | 12,124 | 192 | 101 ch / 27 tok | 503 ch / 130 tok | 902 ch / 247 tok | False | `content` |
| **2099** | arXiv CS.AI | INTERESTING | 0 | 1,548 | 149 ch / 39 tok | 551 ch / 133 tok | 951 ch / 224 tok | False | `summary` |
| **2110** | arXiv CS.AI | MUST_KNOW | 0 | 1,218 | 104 ch / 31 tok | 506 ch / 113 tok | 906 ch / 191 tok | False | `summary` |
| **2115** | arXiv CS.AI | INTERESTING | 0 | 1,775 | 111 ch / 27 tok | 513 ch / 102 tok | 913 ch / 172 tok | False | `summary` |
| **7** | OpenAI News | INTERESTING | 0 | 163 | 75 ch / 20 tok | 240 ch / 54 tok | 240 ch / 54 tok | **True** | `summary` |
| **28** | OpenAI News | MUST_KNOW | 0 | 143 | 82 ch / 21 tok | 227 ch / 56 tok | 227 ch / 56 tok | **True** | `summary` |
| **1155** | DeepMind Blog | NOISE | 0 | 54 | 90 ch / 20 tok | 146 ch / 34 tok | 146 ch / 34 tok | **True** | `summary` |
| **1152** | DeepMind Blog | MUST_KNOW | 0 | 0 | 84 ch / 23 tok | 84 ch / 23 tok | 84 ch / 23 tok | **True** | `title_source` |
| **3098** | OpenAI News | MUST_KNOW | 0 | 0 | 63 ch / 20 tok | 63 ch / 20 tok | 63 ch / 20 tok | **True** | `title_source` |

### Context Findings Summary
1. **Fallback Logic:** Summary fallback operated correctly (`content` $\to$ `summary` $\to$ `title_source`).
2. **Data Availability Bottleneck:** 148 of 150 articles lacked enriched content; 45 lacked summaries entirely.
3. **Identity of B and C:** For **120 out of 150 articles (80.0%)**, Context B and Context C were **identical strings**.
4. **Token Truncation:** Zero truncation occurred. Sequence lengths ranged from 120 to 437 tokens across all conditions, well below ModernBERT's 512-token ceiling.

---

## 7. Structured-Question Verification (E4–E6)

We audited the output and probability distributions of all six structured questions across E4, E5, and E6:

### Distribution Matrix across All Intermediate Questions

```
========================================================================================
Question                       Choice                          E4 (Count / %)   E5 (Count / %)   E6 (Count / %)
========================================================================================
ai_centrality                  core_focus                      103 (68.7%)      118 (78.7%)      119 (79.3%)
                               secondary_focus                  46 (30.7%)       32 (21.3%)       31 (20.7%)
                               tangential_or_unrelated           1 ( 0.7%)        0 ( 0.0%)        0 ( 0.0%)
----------------------------------------------------------------------------------------
new_development                concrete_artifact_or_event      113 (75.3%)       70 (46.7%)       73 (48.7%)
                               ongoing_evolution                21 (14.0%)       72 (48.0%)       70 (46.7%)
                               speculative_or_editorial         16 (10.7%)        8 ( 5.3%)        7 ( 4.7%)
----------------------------------------------------------------------------------------
event_type                     model_release                    37 (24.7%)       46 (30.7%)       44 (29.3%)
                               developer_tool_or_infra           6 ( 4.0%)        9 ( 6.0%)       10 ( 6.7%)
                               research_paper                   85 (56.7%)       78 (52.0%)       79 (52.7%)
                               security_or_vulnerability         6 ( 4.0%)        7 ( 4.7%)        7 ( 4.7%)
                               industry_or_business              4 ( 2.7%)        3 ( 2.0%)        3 ( 2.0%)
                               case_study_or_tutorial           11 ( 7.3%)        5 ( 3.3%)        5 ( 3.3%)
                               opinion_or_commentary             1 ( 0.7%)        2 ( 1.3%)        2 ( 1.3%)
----------------------------------------------------------------------------------------
blast_radius                   industry_wide_disruption         35 (23.3%)       71 (47.3%)       72 (48.0%)
                               domain_specific_utility           9 ( 6.0%)       15 (10.0%)       15 (10.0%)
                               minimal_or_localized            106 (70.7%)       64 (42.7%)       63 (42.0%)
----------------------------------------------------------------------------------------
practitioner_awareness_value   essential_attention               2 ( 1.3%)       13 ( 8.7%)       24 (16.0%)
                               informative_reference            71 (47.3%)      106 (70.7%)       95 (63.3%)
                               negligible_value                 77 (51.3%)       31 (20.7%)       31 (20.7%)
----------------------------------------------------------------------------------------
final_importance               must_know                         6 ( 4.0%)       13 ( 8.7%)       13 ( 8.7%)
                               interesting                     140 (93.3%)      136 (90.7%)      136 (90.7%)
                               noise                             4 ( 2.7%)        1 ( 0.7%)        1 ( 0.7%)
========================================================================================
```

### Analysis of the Collapse
1. In `ai_centrality`, `tangential_or_unrelated` was chosen 0 times in E5/E6.
2. In `practitioner_awareness_value`, `informative_reference` dominated at 70.7% in E5.
3. In `final_importance`, `interesting` mean probability was **0.5778**, while `noise` mean probability was **0.1205** (maximum was only 0.3948 across the entire dataset). `noise` was literally incapable of winning the argmax in E5 and E6.
4. Because the questions were evaluated in parallel, `final_importance` never used the intermediate answers; it merely reflected the unbalanced criteria definitions.

---

## 8. Label-Mapping Verification

### The 4-Way to 3-Way ArgMax Inconsistency

In Formulation 1 (E1–E3), Laya was queried with 4 options: `high`, `medium`, `low`, `ignore`.
The script mapped labels using:
```python
mapped_label = BASELINE_IMPORTANCE_MAP[raw_choice]
```
where `BASELINE_IMPORTANCE_MAP = {"high": "MUST_KNOW", "medium": "INTERESTING", "low": "NOISE", "ignore": "NOISE"}`.
Simultaneously, probabilities were consolidated:
$$P(\text{MUST\_KNOW}) = P(\text{high})$$
$$P(\text{INTERESTING}) = P(\text{medium})$$
$$P(\text{NOISE}) = P(\text{low}) + P(\text{ignore})$$

### Simulation of True 3-Way Probability ArgMax on E1–E3

If `predicted_final_label` had been determined by $\operatorname{argmax}(P(\text{MUST\_KNOW}), P(\text{INTERESTING}), P(\text{NOISE}))$, the results change significantly:

| Experiment | Metric | Original (Reported) | Corrected (3-Way ArgMax) | Delta |
|:---:|:---|:---:|:---:|:---:|
| **E1** | Accuracy | 44.00% | 34.67% | -9.33 pp |
| | MUST_KNOW Recall | 21.88% (14/64) | 20.31% (13/64) | -1.57 pp |
| | **NOISE Recall** | **5.56% (1/18)** | **33.33% (6/18)** | **+27.77 pp** |
| | Predicted NOISE Count | 12 | 42 | +30 |
| **E2** | Accuracy | 49.33% | 42.00% | -7.33 pp |
| | MUST_KNOW Recall | 39.06% (25/64) | 37.50% (24/64) | -1.56 pp |
| | **NOISE Recall** | **0.00% (0/18)** | **5.56% (1/18)** | **+5.56 pp** |
| | Predicted NOISE Count | 5 | 22 | +17 |
| **E3** | Accuracy | 48.00% | 40.67% | -7.33 pp |
| | MUST_KNOW Recall | 40.62% (26/64) | 39.06% (25/64) | -1.56 pp |
| | **NOISE Recall** | **0.00% (0/18)** | **5.56% (1/18)** | **+5.56 pp** |
| | Predicted NOISE Count | 5 | 22 | +17 |

**Finding:** The reported 0.00% NOISE recall in E2 and E3 was partly an artifact of probability splitting. In E2, Article ID 2305 was assigned 47% probability to NOISE ($P(\text{low})=0.2638, P(\text{ignore})=0.2059$), but labeled `INTERESTING` because neither option individually beat `medium` (0.3223). However, even under true 3-way argmax, Laya generated 21 false-positive noise predictions on valuable articles, dropping accuracy to 42.00%.

---

## 9. Investigation of All 18 Human NOISE Articles

Below is the complete audit table for all 18 human NOISE articles under Experiment E2:

| Article ID | Title | Human Label | Predicted (E2) | P(MK) | P(INT) | P(NOI) | Confidence (Entropy) | Margin | Raw Probs (ign, low, med, high) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1155** | Piloting the world's first double-blind AI ev... | NOISE | MUST_KNOW | 0.3718 | 0.3702 | 0.2581 | 0.1109 | 0.0016 | 0.0665, 0.1916, 0.3702, 0.3718 |
| **2078** | 4 ways to address the failures we found along... | NOISE | INTERESTING | 0.1313 | 0.4566 | 0.4121 | 0.0994 | 0.0445 | 0.1323, 0.2798, 0.4566, 0.1313 |
| **2079** | The US spent billions on border surveillance.... | NOISE | INTERESTING | 0.1177 | 0.5201 | 0.3622 | 0.1491 | 0.1579 | 0.1071, 0.2551, 0.5201, 0.1177 |
| **2081** | Could AI really kill us all? Your questions,... | NOISE | INTERESTING | 0.2369 | 0.4626 | 0.3005 | 0.0961 | 0.1621 | 0.1042, 0.1963, 0.4626, 0.2369 |
| **2082** | Building the materials foundation for AI | NOISE | INTERESTING | 0.3300 | 0.4622 | 0.2077 | 0.1513 | 0.1322 | 0.0631, 0.1446, 0.4622, 0.3300 |
| **2083** | Roundtables: Could AI really kill us all? | NOISE | INTERESTING | 0.1998 | 0.5225 | 0.2777 | 0.1552 | 0.2448 | 0.0678, 0.2099, 0.5225, 0.1998 |
| **2213** | Fact Grounded Attention: Eliminating Hallucin... | NOISE | MUST_KNOW | 0.4719 | 0.3543 | 0.1738 | 0.1860 | 0.1176 | 0.0479, 0.1259, 0.3543, 0.4719 |
| **2303** | Study: Young users (9 to 18Y) ditch Google fo... | NOISE | INTERESTING | 0.3286 | 0.4000 | 0.2714 | 0.1015 | 0.0714 | 0.0744, 0.1970, 0.4000, 0.3286 |
| **2305** | How AI is impacting who gets to cross borders | NOISE | INTERESTING | 0.2081 | 0.3223 | **0.4697** | 0.0129 | 0.1474 | 0.2059, 0.2638, 0.3223, 0.2081 |
| **2605** | ChatGPT for marketing teams | NOISE | INTERESTING | 0.3141 | 0.4937 | 0.1922 | 0.1779 | 0.1796 | 0.0504, 0.1418, 0.4937, 0.3141 |
| **2924** | OpenAI nonprofit jam | NOISE | INTERESTING | 0.1195 | 0.5429 | 0.3376 | 0.1708 | 0.2053 | 0.0917, 0.2459, 0.5429, 0.1195 |
| **3039** | Adebayo Ogunlesi joins OpenAI’s Board of Dire... | NOISE | INTERESTING | 0.3410 | 0.4472 | 0.2118 | 0.1650 | 0.1062 | 0.0396, 0.1722, 0.4472, 0.3410 |
| **3054** | Data-driven beauty and creativity with ChatGP... | NOISE | INTERESTING | 0.3737 | 0.3947 | 0.2316 | 0.1222 | 0.0210 | 0.0718, 0.1598, 0.3947, 0.3737 |
| **3102** | Awakening Sleeping Beauties at The Met | NOISE | INTERESTING | 0.1112 | 0.5200 | 0.3689 | 0.1769 | 0.1511 | 0.0770, 0.2919, 0.5200, 0.1112 |
| **3210** | Simplifying contract reviews with AI | NOISE | INTERESTING | 0.2121 | 0.4746 | 0.3133 | 0.1386 | 0.1613 | 0.0565, 0.2568, 0.4746, 0.2121 |
| **3824** | Learn the Hugging Face Kernel Hub in 5 Minute... | NOISE | MUST_KNOW | 0.4175 | 0.3703 | 0.2122 | 0.1398 | 0.0472 | 0.0652, 0.1470, 0.3703, 0.4175 |
| **3870** | The New and Fresh analytics in Inference Endp... | NOISE | INTERESTING | 0.3344 | 0.3936 | 0.2719 | 0.1025 | 0.0592 | 0.0716, 0.2003, 0.3936, 0.3344 |
| **4332** | Introducing Hugging Face for Education 🤗 | NOISE | INTERESTING | 0.2927 | 0.4505 | 0.2568 | 0.1267 | 0.1578 | 0.0626, 0.1942, 0.4505, 0.2927 |

---

## 10. Investigation of Specific Failure Cases

### 1. ID 3549: "Build with Nano Banana Pro, our Gemini 3 Pro Image model"
- **Human Label:** MUST_KNOW | **Reason:** "Major Gemini 3 image-model release..."
- **Laya Prediction:** E1–E3: `NOISE` (choice: `low`, raw probs: `ign` 0.21, `low` 0.37, `med` 0.29, `high` 0.13). E4–E6: `INTERESTING` (P(MK)=0.3800, P(INT)=0.4113).
- **Context Provided:** `title_source` ONLY (92 characters). `content` and `summary` were both `None`.
- **Root Cause:** **INPUT DEFICIENCY + MODEL ZERO-SHOT LIMITATION**. The input text was literally 92 characters. Without body text explaining that "Nano Banana Pro" is Google's new frontier image generation model, the model saw a whimsical title and assigned 58% probability to `low` + `ignore`.

### 2. ID 28: "How GPT-5.6 Sol helps run quantum computing experiments"
- **Human Label:** MUST_KNOW | **Reason:** "It's about a new model performance"
- **Laya Prediction:** E1–E3: `INTERESTING` (choice: `medium`, P(INT)=0.4989, P(MK)=0.1904). E4–E6: `INTERESTING` (P=0.7052).
- **Context Provided:** Title + Source + 143-character summary.
- **Root Cause:** **HUMAN LABEL MISMATCH / SENSITIVITY**. The title frames the article as a domain application / case study ("helps run quantum computing experiments"). Under the rubric, applied case studies belong in `INTERESTING`. The human annotator elevated it to `MUST_KNOW` due to the keyword "GPT-5.6". Laya categorized it as an application/case study (`medium`).

### 3. IDs 2078 & 2079: Border Surveillance Articles (MIT Technology Review AI)
- **Human Label:** NOISE | **Reason:** "Not related" / "Unnecessary details"
- **Laya Prediction:**
  - ID 2078: `INTERESTING` across E1–E6. In E2: P(INT)=0.4566, P(NOI)=0.4121.
  - ID 2079: In E1 (Title only): `NOISE` (P=0.6752). In E2 (Summary added): `INTERESTING` (P=0.5201).
- **Root Cause:** **INPUT CONTEXT DILUTION & MAPPING BUG**. In ID 2079, Laya correctly classified Title-only as `NOISE` (E1). But when the 332-character RSS summary containing surveillance technology jargon was appended (E2), Laya shifted to `INTERESTING`. In ID 2078, Laya assigned 41.2% mass to NOISE, but `medium` won the 4-way argmax.

### 4. ID 1155: "Piloting the world's first double-blind AI evaluations"
- **Human Label:** NOISE | **Reason:** "Nothing useful about AI"
- **Laya Prediction:** E1: `INTERESTING` (P(NOI)=0.4629, P(INT)=0.3523). E2: `MUST_KNOW` (raw `high`=0.3718, `medium`=0.3702, margin=0.0016).
- **Root Cause:** **HUMAN LABEL SENSITIVITY & RAZOR-THIN MARGIN**. The human labeler considered policy/governance non-technical and labeled it `NOISE`. For a machine classifier, a DeepMind announcement of "world's first double-blind evaluations" carries strong frontier benchmark signals. Furthermore, in E2 the margin between `MUST_KNOW` and `INTERESTING` was 0.0016 (a near-perfect tie).

---

## 11. Metric & Operational Gate Arithmetic Verification

### Metric Recomputation Audit
We wrote an independent Python verification engine from scratch (bypassing `sklearn`) to recompute accuracy, precision, recall, and F1 across all conditions directly from `evaluation/phase_0_4_predictions.jsonl`:
- **Result:** **0 discrepancies found.** Every number in `evaluation/phase_0_4_operational_analysis.json` matches independent calculations down to the 4th decimal place.
- **Sklearn Zero-Division Check:** In E2, E3, E5, E6, `NOISE` precision was 0.0 because Laya predicted NOISE 5 times in E2/E3 (0 TP, 5 FP) and 1 time in E5/E6 (0 TP, 1 FP). The denominator ($TP + FP$) was non-zero; precision was mathematically 0.0, not a zero-division fallback artifact.

### Operational Gate Arithmetic Audit
We audited all 30 operational policies directly from `phase_0_4_predictions.jsonl`:
- **Result:** **0 discrepancies found.**
  - `articles_passed + articles_skipped == 150` for all policies.
  - `pct_passed + qwen_reduction_pct == 100.00%` for all policies.
  - In E2 ArgMax Gate: 145 passed, 5 skipped, MK passed = 63/64 (98.44%), NOISE filtered = 0/18 (0.00%).
  - In E2 P(NOISE) < 0.25: 76 passed, 74 skipped, MK passed = 38/64 (59.38%), MK false negatives = 26, NOISE filtered = 12/18 (66.67%).

---

## 12. Latency & Hardware Profile Verification

### Latency Measurement Scope
Latency is recorded in `evaluate_laya_phase_0_4.py` lines 589–592:
```python
t_start = time.perf_counter()
res = agent.predict(state_text, q_schema)
lat_ms = (time.perf_counter() - t_start) * 1000.0
```
This block measures:
1. State text tokenization (`Agent.tok`).
2. Sequence formatting and option marker insertion for each question (`build_sequence`).
3. Collation into tensor batch (`collate_items`).
4. Transfer of batch tensors to Apple Silicon Metal (`.to(device)`).
5. Single forward pass through ModernBERT (`model.forward()`).
6. Transfer of logits to CPU numpy arrays (`.cpu().numpy()`).
7. Temperature scaling, softmax, entropy computation (`_decode_answers`).

It **excludes** model weights loading (measured once at startup as 2.52 seconds) and tokenizer compilation (cached).

### Why E4–E6 Latency Was Higher Than E1–E3
1. **Batch Size Scaling:**
   - E1–E3 evaluated 4 questions $\implies$ batch size = 4 rows.
   - E4–E6 evaluated 6 questions $\implies$ batch size = 6 rows (+50% matrix multiply volume).
2. **MPS Precision Threshold:**
   In Laya (`laya/agent.py:144`), MPS FP16 autocast is governed by `mps_amp_min_rows = 5`.
   - E1–E3 (4 rows) ran in **FP32** because $4 < 5$.
   - E4–E6 (6 rows) ran in **FP16** with autocast overhead.
3. **Context Length Scaling:**
   - E4 (Context A, ~140 tokens): 225.0 ms avg.
   - E5 (Context B, ~220 tokens): 340.6 ms avg.
   - E6 (Context C, ~260 tokens): 374.5 ms avg.
Attention complexity scales with $O(L^2)$ and $N \times L$. The measurements are completely consistent with single-call parallel execution.

---

## 13. Phase 0.3 vs. Phase 0.4 Parity Verification

We performed a bit-for-bit comparison between `evaluation/phase_0_3_predictions.jsonl` and Phase 0.4 E2:
- **Prediction Match:** **150 / 150 (100.00%) identical predictions.**
- **Probability Match:** Maximum absolute difference in raw probabilities across all 150 articles was **0.000000**.
- **Explanation:** E2 reused the exact same 4-question prompt, the same `convaiinnovations/laya` checkpoint, the same `mps` device, and the same context construction logic (`build_laya_input`). Phase 0.4 successfully and perfectly reproduced Phase 0.3. However, it also inherited the 4-way argmax mapping bug from Phase 0.3.

---

## 14. Test Coverage Assessment

We inspected [`tests/test_phase_0_4_evaluation.py`](file:///Users/apple/Desktop/ai_news_digest/tests/test_phase_0_4_evaluation.py) (8 tests, all passing):

### Strengths
- Enforces dataset integrity (150 unique rows, 64/68/18 label counts).
- Verifies deterministic context string formatting on synthetic strings.
- Verifies arithmetic invariants on synthetic mock dictionaries (passed + skipped == total).
- Verifies artifact existence and JSON schema completeness.

### Critical Blind Spots & Test Gaps
1. **Did Not Test 4-Way vs. 3-Way ArgMax Consistency:** The test `test_map_prediction_to_canonical` passed synthetic `high` and `low` strings, verifying that `low` maps to `NOISE`. It never checked whether `raw_choice` agreed with `argmax(canonical_probs)`.
2. **Did Not Test Confidence Semantics:** The tests never asserted whether `confidence` in the predictions matched $\max(p)$ or normalized entropy.
3. **Did Not Detect Dataset Content Scarcity:** Tests verified string slicing on synthetic mock strings (`X * 1200`), but never validated how many real dataset records actually contained enriched content versus empty strings.
4. **Did Not Detect Parallel Intermediate Isolation:** Tests never validated whether intermediate questions in Formulation 2 had any causal connection to the final importance decision.

---

## 15. Environment & Checkpoint Comparison

| Component | Phase 0.3 | Phase 0.4 | Status |
|:---|:---|:---|:---:|
| **Python** | 3.11.15 | 3.11.15 | Identical |
| **PyTorch** | 2.14.0 | 2.14.0 | Identical |
| **Transformers** | 5.17.0 | 5.17.0 | Identical |
| **Safetensors** | 0.8.0 | 0.8.0 | Identical |
| **Hugging Face Hub**| 1.33.0 | 1.33.0 | Identical |
| **Laya** | 0.3.20 | 0.3.20 | Identical |
| **Model ID** | `convaiinnovations/laya` | `convaiinnovations/laya` | Identical |
| **Base Architecture**| ModernBERT-large (421M) | ModernBERT-large (421M) | Identical |
| **Device Backend** | `mps` (Metal) | `mps` (Metal) | Identical |

The software, hardware, and weight environments were 100% identical. No environmental shifts occurred.

---

## 16. Root-Cause Analysis Table

| Issue | Observed Behavior | Expected Behavior | Evidence | Root Cause | Severity | Confidence |
|:---|:---|:---|:---|:---|:---:|:---:|
| **ArgMax Inconsistency** | Predicted label does not match argmax of 3-class canonical probabilities in 17 articles (E2). | Predicted label should match highest class probability in canonical distribution. | `evaluate_laya_phase_0_4.py:251` uses `BASELINE_IMPORTANCE_MAP[raw_choice]` before summing `p_low + p_ignore`. Article 2305: P(NOI)=0.47 vs P(INT)=0.32, labeled INT. | Split probability mass across 4 options vs consolidated 3-class distribution. | **HIGH** | **CONFIRMED** |
| **Zero NOISE Recall (E2/E3)** | 0 out of 18 human NOISE articles predicted as NOISE in E2/E3. | Model should recall clear noise / off-topic articles. | 15 predicted INT, 3 predicted MK. 1 article (2305) was a direct victim of ArgMax Inconsistency bug; 17 had INT probability ~0.45. | Combination of ArgMax bug, semantic overlap of AI feeds, and lack of content. | **HIGH** | **CONFIRMED** |
| **Context C Ineffective Ablation** | E3/E6 showed no meaningful performance improvement over E2/E5. | 800 characters of context should provide additional discriminative signal over 400 characters. | For 120/150 articles (80.0%), Context B and C were character-for-character identical. Only 2 articles had enriched content. | Dataset content scarcity (`phase_0_2_labels.jsonl` contains 148 empty content fields). | **HIGH** | **CONFIRMED** |
| **Extreme INTERESTING Bias (E4–E6)** | 136/150 articles (90.7%) predicted as INTERESTING in E5/E6. | Balanced or discriminative distribution reflecting true labels (64 MK, 68 INT, 18 NOI). | `final_importance` evaluated independently in parallel. `must_know` had extreme barrier ("paradigm shift"); `interesting` had broad umbrella. | Prompt asymmetry + parallel execution without intermediate conditioning. | **CRITICAL** | **CONFIRMED** |
| **Confidence Field Mismatch** | `confidence` values in JSONL are ~0.10–0.25, while analysis report cites avg confidence ~0.45. | Consistent definition of confidence across all artifacts. | Predictions recorded `confidence_from_probs` (entropy). Analysis computed `average_confidence` from `np.max(p)`. | Laya API semantic confusion between entropy and calibrated answer probability. | **MEDIUM** | **CONFIRMED** |
| **Operational Gating Failure** | No threshold achieved safe MK recall ($\ge 98\%$) with meaningful Qwen reduction ($\ge 20\%$). | Model serves as high-recall pre-filter to drop cheap noise. | E2 P(NOI)<0.25 dropped 26 MK articles; ArgMax gate dropped Article 3549 (Gemini 3 Pro) and filtered 0% noise. | Laya assigns significant probability mass to both valuable and non-valuable AI news. | **CRITICAL** | **CONFIRMED** |

---

## 17. Audit Verdict & Next Actions

### AUDIT VERDICT

# **B. RESULTS ARE PARTIALLY INVALID — specific implementation/evaluation bugs affected results**

#### Justification:
1. **Experiments E1, E2, E3 (Baseline Formulation) are PARTIALLY INVALID:**
   The reported metrics in E1–E3 were distorted by the 4-way argmax mapping bug. By taking the argmax over 4 classes (`high`, `medium`, `low`, `ignore`) where the negative class was split into two options, `NOISE` recall was artificially suppressed to 0.00% (affecting 30 articles in E1 and 17 articles in E2/E3). When evaluated under true 3-way probability argmax, NOISE recall is 5.56% in E2/E3 and 33.33% in E1.
2. **Experiments E3 and E6 (Context C Ablation) are INVALID as a true 800-character test:**
   For 80.0% of the dataset (120/150 articles), Context C was literally identical text to Context B due to the lack of enriched article content in `phase_0_2_labels.jsonl`. The claim that "expanding context beyond 400 characters yields diminishing returns" was largely measuring duplicate inputs.
3. **Experiments E4, E5, E6 (Rubric-Aligned Formulation) are INVALID as a multi-step rubric test:**
   The experimental architecture assumed that defining 6 questions would "decompose the human decision procedure". Because Laya runs all 6 questions in parallel with zero sequential conditioning, the target question operated zero-shot on an asymmetric prompt, causing a complete collapse onto `INTERESTING` (90.7%).
4. **The Operational Gate Calculations are MATHEMATICALLY VALID on the generated predictions, but PRACTICALLY COMPROMISED:**
   The formulas, pass counts, and reductions calculated in `phase_0_4_operational_analysis.json` are 100% exact representations of `phase_0_4_predictions.jsonl`. However, the operational gate conclusions regarding noise filtering are distorted by the underlying probability-splitting bug.

---

### NEXT ACTION (Diagnostic Recommendations Only — Do Not Implement Yet)

1. **Do NOT promote Laya to production yet.** Even when correcting the argmax bug, zero-shot Laya achieves only 42.00% 3-way accuracy, generates 21 false noise skips on valuable articles, and fails to safely triage frontier models without full text.
2. **Correct the Evaluation Mapper in Phase 0.4.1 (When Authorized):**
   Update `map_prediction_to_canonical` to determine the predicted label strictly from the argmax of the consolidated canonical probabilities:
   ```python
   mapped_label = max(canonical_probs.items(), key=lambda x: x[1])[0]
   ```
3. **Record Both Calibrated Confidence and Entropy:**
   Update the prediction logger to explicitly record:
   - `answer_confidence`: $\max(p)$ (calibrated choice probability)
   - `entropy_confidence`: $1 - H(p)/\log(k)$ (distribution concentration)
4. **Conduct True Context Ablation After Multi-Source Enrichment:**
   Prior to re-evaluating context length (400 vs 800 vs 1500 chars), the 150 gold-standard articles in `phase_0_2_labels.jsonl` must have their `content` field populated via the Task 1.2 enrichment pipeline so that Context C actually contains 800 distinct characters of article body.
5. **Redesign Structured Rubric Evaluation as a Decision Tree or Rule Synthesis:**
   If structured questions (`ai_centrality`, `new_development`, `event_type`, `blast_radius`, `practitioner_awareness_value`) are evaluated, the final importance label should either be:
   - Synthesized deterministically via Python business logic directly implementing `docs/importance_rubric.md`, OR
   - Evaluated sequentially where intermediate answers are formatted into the prompt state for the final decision.
