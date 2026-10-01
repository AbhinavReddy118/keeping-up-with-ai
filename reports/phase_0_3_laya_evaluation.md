# Phase 0.3: Laya vs. Human Gold-Standard Evaluation Report

**Document Version:** 1.0  
**Phase:** Phase 0.3 — Evaluation of Local Laya Decision Model against Human Gold-Standard  
**Evaluation Dataset:** [`evaluation/phase_0_2_labels.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_2_labels.jsonl) (150 articles)  
**Predictions Output:** [`evaluation/phase_0_3_predictions.jsonl`](file:///Users/apple/Desktop/ai_news_digest/evaluation/phase_0_3_predictions.jsonl)  
**Runner Module:** [`evaluation/evaluate_laya.py`](file:///Users/apple/Desktop/ai_news_digest/evaluation/evaluate_laya.py)  
**Canonical Rubric:** [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md)  
**Date:** October 1, 2026  
**Status:** Evaluation Completed (Read-Only; Production Code & Schema Untouched)

---

## 1. Objective

The objective of Phase 0.3 is to conduct a **rigorous, reproducible, empirical evaluation** of the local Laya decision model (`convaiinnovations/laya` running on Apple Silicon Metal Performance Shaders - `mps`) against the **150 human-labeled gold-standard articles** produced in Phase 0.2 Stage B.

The core question being tested is:
> *Can the lightweight (421M parameter) Laya model safely and accurately serve as a first-pass triage/gating filter to separate `MUST_KNOW` and `INTERESTING` developments from `NOISE` prior to heavy LLM analysis (Ollama Qwen3 4B)?*

This evaluation strictly compares human ground-truth labels against Laya's deterministic predictions without modifying any production pipeline code, database schemas, or ranking algorithms.

---

## 2. Dataset

The evaluation was executed over the full 150-article dataset generated in Phase 0.2:
- **Total Articles:** 150
- **Human Label Distribution:**
  - **`MUST_KNOW`**: 64 articles (42.7%)
  - **`INTERESTING`**: 68 articles (45.3%)
  - **`NOISE`**: 18 articles (12.0%)
- **Data Integrity:**
  - 100% of the 150 articles were assigned human labels and non-empty reasons in accordance with [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md).
  - Zero duplicate article IDs.
  - Zero modifications to human labels during evaluation.

---

## 3. Laya Configuration

The model and runtime environment were configured to match the verified setup from Task 1.5A and 1.5B:
- **Model Identifier:** `convaiinnovations/laya`
- **Underlying Architecture:** `answerdotai/ModernBERT-large` (421M parameters) with dual/cross-encoder multi-task classification heads
- **Context Constraints:** `cfg['max_len'] = 512` tokens, `cfg['head_max_len'] = 192` tokens
- **Inference Runtime:** Python 3.11.15 in `.venv`
- **Dependencies:** `laya==0.3.20`, `transformers==5.17.0`, `torch==2.14.0`, `scikit-learn==1.9.1`
- **Execution Backend:** Apple Silicon `mps` (Metal Performance Shaders, fp16/bf16 mixed precision)
- **Parallel Decision Heads Evaluated:**
  1. `ai_relevance` (`not_relevant`, `weakly_relevant`, `moderately_relevant`, `highly_relevant`)
  2. `event_type` (`model_release`, `research`, `tool_release`, `company_news`, `industry_news`, `opinion_editorial`, `other`)
  3. `novelty` (`breakthrough`, `significant`, `incremental`, `unknown`)
  4. `importance` (`high`, `medium`, `low`, `ignore`)

---

## 4. Input Strategy

To respect Laya's 512-token context ceiling without fetching external web content or running enrichment passes, inputs were constructed using the bounded hierarchy established in Phase 0.2:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT STRATEGY HIERARCHY                           │
├─────────────────┬────────────────────────────────────────────┬──────────────┤
│ Strategy        │ Construction Rule                          │ Article Count│
├─────────────────┼────────────────────────────────────────────┼──────────────┤
│ content_lead    │ Source + Title + First ~400 chars content  │ 2 (1.3%)     │
│ summary         │ Source + Title + First ~400 chars summary  │ 103 (68.7%)  │
│ title_source    │ Source + Title (no text body available)    │ 45 (30.0%)   │
├─────────────────┴────────────────────────────────────────────┴──────────────┤
│ TOTAL                                                        │ 150 (100.0%) │
└─────────────────────────────────────────────────────────────────────────────┘
```

All 150 input representations consumed between 21 and 294 tokens, well within ModernBERT's 512-token boundary. No context overflow or truncation errors occurred.

---

## 5. Label Mapping

Laya's native importance prompt queries four bounded choices:
- `high`: *"Crucial update, must-read for AI engineers and researchers"*
- `medium`: *"Solid value, relevant to practitioners and researchers"*
- `low`: *"Marginal interest, routine news, or niche topic"*
- `ignore`: *"Noise, completely irrelevant, or no technical value"*

These criteria map directly and defensibly to the three human classes defined in [`docs/importance_rubric.md`](file:///Users/apple/Desktop/ai_news_digest/docs/importance_rubric.md):

```
┌───────────────────────────┬───────────────────────────┬────────────────────────────────────────┐
│ Laya Native Choice        │ Canonical Human Class     │ Semantic Equivalence Rationale         │
├───────────────────────────┼───────────────────────────┼────────────────────────────────────────┤
│ high                      │ MUST_KNOW                 │ Crucial / frontier development.        │
│ medium                    │ INTERESTING               │ Solid technical value / awareness.     │
│ low                       │ NOISE                     │ Marginal / niche routine news.         │
│ ignore                    │ NOISE                     │ Irrelevant / zero technical value.     │
└───────────────────────────┴───────────────────────────┴────────────────────────────────────────┘
```

The mapped prediction is deterministic: $\text{ArgMax}(\{ \text{high}, \text{medium}, \text{low}, \text{ignore} \})$. Full probability distributions were also recorded for every article.

---

## 6. Overall Metrics

Calculated using `sklearn.metrics` across all 150 gold-standard articles:

| Metric | Measured Value | Raw Ratio |
|:---|:---:|:---:|
| **Overall Accuracy** | **49.33%** | 74 / 150 |
| **Exact Correct Predictions** | **74** | — |
| **Exact Incorrect Predictions** | **76** | — |
| **Macro Precision** | **0.3539** | — |
| **Macro Recall** | **0.3704** | — |
| **Macro F1 Score** | **0.3479** | — |
| **Weighted Precision** | **0.4658** | — |
| **Weighted Recall** | **0.4933** | — |
| **Weighted F1 Score** | **0.4607** | — |

---

## 7. Per-Class Metrics

Detailed precision, recall, F1, and support per category:

| Class | Precision | Recall | F1 Score | Support (True Count) | Predicted Count |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`MUST_KNOW`** | **0.5814** (58.1%) | **0.3906** (39.1%) | **0.4673** | 64 | 43 |
| **`INTERESTING`** | **0.4804** (48.0%) | **0.7206** (72.1%) | **0.5765** | 68 | 102 |
| **`NOISE`** | **0.0000** (0.0%) | **0.0000** (0.0%) | **0.0000** | 18 | 5 |

### Key Observations:
1. **`MUST_KNOW` Precision is Moderate (58.1%):** When Laya predicts `MUST_KNOW`, it is correct nearly 6 out of 10 times.
2. **`MUST_KNOW` Recall is Very Low (39.1%):** Laya missed 39 out of 64 must-know articles.
3. **`INTERESTING` Dominates Predictions:** Laya predicted `INTERESTING` for **102 out of 150 articles (68.0%)**, creating a massive central-tendency clustering.
4. **`NOISE` is Almost Never Predicted:** Laya predicted `NOISE` for only 5 articles across the entire dataset, and was incorrect on all 5. It achieved **0.0% recall on true NOISE**.

---

## 8. Confusion Matrix

Class order: `MUST_KNOW`, `INTERESTING`, `NOISE`.

### Tabular Representation:
```
                      Predicted Label (Laya)
Human Gold Standard   MUST_KNOW   INTERESTING   NOISE   Total
─────────────────────────────────────────────────────────────
MUST_KNOW                    25            38       1      64
INTERESTING                  15            49       4      68
NOISE                         3            15       0      18
─────────────────────────────────────────────────────────────
Total Predicted              43           102       5     150
```

### Visual Breakdown:
- **True Positives:**
  - `MUST_KNOW`: **25**
  - `INTERESTING`: **49**
  - `NOISE`: **0**
- **False Positives:**
  - `MUST_KNOW`: $15 + 3 = \mathbf{18}$
  - `INTERESTING`: $38 + 15 = \mathbf{53}$
  - `NOISE`: $1 + 4 = \mathbf{5}$
- **False Negatives:**
  - `MUST_KNOW`: $38 + 1 = \mathbf{39}$
  - `INTERESTING`: $15 + 4 = \mathbf{19}$
  - `NOISE`: $3 + 15 = \mathbf{18}$

---

## 9. MUST_KNOW Recall & False Negatives

Because the intended use of Laya in a two-stage architecture is a gating filter before expensive LLM analysis, **`MUST_KNOW` Recall is the most critical safety metric**.

- **Measured `MUST_KNOW` Recall:** **39.06%** (25 / 64)
- **`MUST_KNOW` False Negative Rate:** **60.94%** (39 / 64)

Of the 64 articles that human annotators judged critical for AI practitioners, **Laya failed to identify 39 of them**.

### Inventory of All 39 MUST_KNOW False Negatives

Below is the complete manifest of critical articles dropped by Laya, including prediction probabilities:

| ID | Source | Laya Prediction | Probabilities | Article Title |
|:---|:---|:---:|:---:|:---|
| **28** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.19, P_{\text{med}}=0.50$ | How GPT-5.6 Sol helps run quantum computing experiments |
| **2110** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.38, P_{\text{med}}=0.39$ | Risk-Aware Occupancy for Safety-Oriented Autonomous Driving |
| **2143** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.23, P_{\text{med}}=0.48$ | Bio-MF: Low-Latency EEG-to-fNIRS Generation for BCI |
| **2154** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.12, P_{\text{med}}=0.53$ | SWE-Proof: Resolving Real-World Issues with Machine Proofs |
| **2179** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.29, P_{\text{med}}=0.47$ | VidOmni-Bench: Benchmark for Fine-Grained Video Understanding |
| **2204** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.25, P_{\text{med}}=0.49$ | Benchmarking Explanatory Quality of Open VLMs in Face Analysis |
| **2224** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.25, P_{\text{med}}=0.51$ | Visual Foundation Model for Medical Image Recognition |
| **2247** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.34, P_{\text{med}}=0.43$ | HERMES: Risk-Aware Multimodal System for Long-Tail Driving |
| **2297** | arXiv CS.AI | `INTERESTING` | $P_{\text{high}}=0.23, P_{\text{med}}=0.50$ | ASLEval: Measuring Privacy Exposure Displacement in Agent Sessions |
| **2316** | Hacker News | `INTERESTING` | $P_{\text{high}}=0.21, P_{\text{med}}=0.35$ | A Summer of AI Optimization |
| **2415** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.14, P_{\text{med}}=0.58$ | Working with APA on youth mental health and AI |
| **2484** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.34, P_{\text{med}}=0.41$ | Introducing LifeSciBench |
| **2524** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.23, P_{\text{med}}=0.51$ | OpenAI’s Frontier Governance Framework |
| **2669** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.27, P_{\text{med}}=0.50$ | An update on our mental health-related work |
| **2753** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.37, P_{\text{med}}=0.46$ | Advancing science and math with GPT-5.2 |
| **2780** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.17, P_{\text{med}}=0.55$ | GPT-5 and the future of mathematical discovery |
| **2937** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.27, P_{\text{med}}=0.49$ | Responding to The New York Times data demands to protect privacy |
| **2986** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.17, P_{\text{med}}=0.48$ | Security on the path to AGI |
| **3035** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.27, P_{\text{med}}=0.55$ | Trading inference-time compute for adversarial robustness |
| **3098** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.33, P_{\text{med}}=0.34$ | Fine-tuning now available for GPT-4o |
| **3242** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.24, P_{\text{med}}=0.46$ | GPTs are GPTs: Labor market impact potential of LLMs |
| **3252** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.33, P_{\text{med}}=0.47$ | OpenAI and Microsoft extend partnership |
| **3266** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.32, P_{\text{med}}=0.44$ | Our approach to alignment research |
| **3306** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.37, P_{\text{med}}=0.40$ | GPT-3 powers the next generation of apps |
| **3309** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.30, P_{\text{med}}=0.51$ | Scaling Kubernetes to 7,500 nodes |
| **3314** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.26, P_{\text{med}}=0.42$ | Generative language modeling for automated theorem proving |
| **3335** | OpenAI News | `INTERESTING` | $P_{\text{high}}=0.28, P_{\text{med}}=0.49$ | Emergent tool use from multi-agent interaction |
| **3488** | Google DeepMind | `INTERESTING` | $P_{\text{high}}=0.25, P_{\text{med}}=0.38$ | Introducing computer use in Gemini 3.5 Flash |
| **3508** | Google DeepMind | `INTERESTING` | $P_{\text{high}}=0.14, P_{\text{med}}=0.46$ | Uniting biological toolkits for a new approach to ALS |
| **3519** | Google DeepMind | `INTERESTING` | $P_{\text{high}}=0.38, P_{\text{med}}=0.41$ | Gemini Robotics-ER 1.6: Real-world robotics via embodied reasoning |
| **3549** | Google DeepMind | `NOISE` | $P_{\text{high}}=0.13, P_{\text{low}}=0.45$ | Build with Nano Banana Pro, our Gemini 3 Pro Image model |
| **3551** | Google DeepMind | `INTERESTING` | $P_{\text{high}}=0.30, P_{\text{med}}=0.40$ | Start building with Gemini 3 |
| **3559** | Google DeepMind | `INTERESTING` | $P_{\text{high}}=0.21, P_{\text{med}}=0.42$ | Accelerating discovery with the AI for Math Initiative |
| **3567** | Microsoft Research | `INTERESTING` | $P_{\text{high}}=0.40, P_{\text{med}}=0.43$ | MindTopo reveals VLMs’ spatial reasoning abilities |
| **3749** | Hugging Face | `INTERESTING` | $P_{\text{high}}=0.31, P_{\text{med}}=0.39$ | Building the Open Agent Ecosystem: Introducing OpenEnv |
| **3944** | Hugging Face | `INTERESTING` | $P_{\text{high}}=0.31, P_{\text{med}}=0.36$ | Introducing SynthID Text |
| **3953** | Hugging Face | `INTERESTING` | $P_{\text{high}}=0.37, P_{\text{med}}=0.37$ | Fixing Gradient Accumulation |
| **4038** | Hugging Face | `INTERESTING` | $P_{\text{high}}=0.16, P_{\text{med}}=0.34$ | Welcome Llama 3 - Meta's new open LLM |
| **4146** | Hugging Face | `INTERESTING` | $P_{\text{high}}=0.17, P_{\text{med}}=0.35$ | Fine-tune Llama 2 with DPO |

---

## 10. Error Analysis

The 76 misclassifications group into six distinct error patterns:

```
┌───────────────────────────────────────┬──────────────┬───────────────┐
│ Error Category                        │ Count        │ % of Errors   │
├───────────────────────────────────────┼──────────────┼───────────────┤
│ A. MUST_KNOW  -> INTERESTING          │ 38           │ 50.0%         │
│ B. MUST_KNOW  -> NOISE                │ 1            │ 1.3%          │
│ C. INTERESTING -> MUST_KNOW           │ 15           │ 19.7%         │
│ D. INTERESTING -> NOISE               │ 4            │ 5.3%          │
│ E. NOISE      -> MUST_KNOW            │ 3            │ 3.9%          │
│ F. NOISE      -> INTERESTING          │ 15           │ 19.7%         │
├───────────────────────────────────────┼──────────────┼───────────────┤
│ TOTAL ERRORS                          │ 76           │ 100.0%        │
└───────────────────────────────────────┴──────────────┴───────────────┘
```

### Pattern A: MUST_KNOW Downgraded to INTERESTING (38 articles)
This is the single largest error mode (50% of all mistakes).
- **Substantive Pattern:** Laya consistently recognizes that frontier releases (Llama 3, Gemini 3 API, DPO fine-tuning, GPT-4o fine-tuning) have "solid value" ($P_{\text{medium}} \approx 0.35 - 0.50$), but fails to elevate them to "crucial / must-read" ($P_{\text{high}}$).
- **Razor-Thin Margins:** In articles like 2110 (Autonomous Driving), 3098 (GPT-4o fine-tuning), 3519 (Gemini Robotics), 3567 (MindTopo), and 3953 (Gradient Accumulation), $P_{\text{high}}$ and $P_{\text{medium}}$ were separated by less than **0.03**.

### Pattern B: MUST_KNOW Dropped to NOISE (1 article)
- **Article 3549:** *Build with Nano Banana Pro, our Gemini 3 Pro Image model*
- **Observed Pattern:** Laya predicted `low` ($P=0.45$). The model likely interpreted the whimsical name "Nano Banana" as consumer/casual noise rather than an enterprise Gemini 3 Pro image generation release.

### Pattern C: INTERESTING Over-Promoted to MUST_KNOW (15 articles)
- **Representative Articles:**
  - 1244: *Improving synthesis prediction of small molecules with RetroChimera* ($P_{\text{high}}=0.49$)
  - 2311: *Create AI Image Free – No Sign-Up Needed* ($P_{\text{high}}=0.44$)
  - 2541: *A new personal finance experience in ChatGPT* ($P_{\text{high}}=0.45$)
- **Observed Pattern:** Laya triggered strongly on high-sounding technical terms ("synthesis prediction at scale", "personal finance in ChatGPT", "reinforcement learning") even when the articles describe narrow domain papers, consumer features, or third-party wrappers.

### Pattern D: INTERESTING Dropped to NOISE (4 articles)
- **Representative Articles:**
  - 3211: *DALL·E 3 system card* ($P_{\text{low}}=0.41$)
  - 3788: *FilBench - Can LLMs Understand and Generate Filipino?* ($P_{\text{ignore}}=0.45$)
  - 3937: *Introducing the Open Leaderboard for Japanese LLMs!* ($P_{\text{low}}=0.44$)
  - 4137: *AudioLDM 2, but faster* ($P_{\text{low}}=0.43$)
- **Observed Pattern:** Regional language benchmarks (Filipino, Japanese) and specialized media models (AudioLDM 2) were classified as `low` (marginal interest/niche), whereas human annotators marked them as `INTERESTING` technical artifacts.

### Pattern E: NOISE Over-Promoted to MUST_KNOW (3 articles)
- **Representative Articles:**
  - 1155: *Piloting the world's first double-blind AI evaluations* ($P_{\text{high}}=0.37$)
  - 2213: *Fact Grounded Attention: Eliminating Hallucination in Large Language Models* ($P_{\text{high}}=0.46$)
  - 3824: *Learn the Hugging Face Kernel Hub in 5 Minutes* ($P_{\text{high}}=0.40$)
- **Observed Pattern:** Strong keyword triggers ("double-blind AI evaluations", "eliminating hallucination", "Kernel Hub") caused Laya to flag these as must-read updates, missing the fact that 3824 is a beginner tutorial and 1155 is an early announcement without deliverables.

### Pattern F: NOISE Promoted to INTERESTING (15 articles)
- **Representative Articles:**
  - 2081: *Could AI really kill us all? Your questions, answered.* ($P_{\text{med}}=0.46$)
  - 2083: *Roundtables: Could AI really kill us all?* ($P_{\text{med}}=0.52$)
  - 2605: *ChatGPT for marketing teams* ($P_{\text{med}}=0.49$)
  - 3039: *Adebayo Ogunlesi joins OpenAI’s Board of Directors* ($P_{\text{med}}=0.45$)
  - 3054: *Data-driven beauty and creativity with ChatGPT* ($P_{\text{med}}=0.39$)
- **Observed Pattern:** Editorial subscriber Q&As, executive board announcements, and corporate marketing promos published by major labs were assigned `medium` (solid value) simply because they mention major AI entities. Laya failed to recognize them as non-technical PR or opinion chatter.

---

## 11. Confidence & Uncertainty Analysis

Laya’s calibrated probabilities reveal significant internal uncertainty across the 150 evaluations:

- **Mean Top Probability:** **0.4574** (barely above random guessing for 4 classes)
- **Median Top Probability:** **0.4520**
- **Min / Max Top Probability:** 0.3104 / 0.7072
- **Mean Margin ($P_{\text{top}} - P_{\text{second}}$):** **0.1573**
- **Median Margin:** **0.1366**
- **Uncertain / Borderline Predictions ($\text{Margin} < 0.10$):** **59 articles (39.3%)**

In nearly 40% of all articles, Laya's top choice beat its second choice by less than 10 percentage points. The class probability distribution across the entire dataset heavily favored `medium`:
$$\overline{P}(\text{medium}) = 40.20\%, \quad \overline{P}(\text{high}) = 32.22\%, \quad \overline{P}(\text{low}) = 19.17\%, \quad \overline{P}(\text{ignore}) = 8.41\%$$

---

## 12. Latency and Runtime

Measurements taken on Apple Silicon (M-series, unified memory, `mps` backend):

- **Model Load Time (Disk Cache to MPS):** **21.21 seconds** (first cold load)
- **Total Batch Execution Time (150 articles):** **38.42 seconds**
- **Average Inference Latency:** **256.14 ms** per article (4 parallel question heads)
- **Median Inference Latency:** **242.43 ms**
- **95th Percentile Latency (P95):** **327.72 ms**
- **Min Latency:** **195.69 ms**
- **Max Latency:** **440.63 ms**
- **Peak Process Memory Footprint (RSS):** **2,840.2 MB (~2.84 GB)**

Operationally, Laya is exceptionally fast: evaluating 150 multi-decision records took under 39 seconds total, compared to the ~35 minutes that Ollama Qwen3 4B would require for equivalent article counts.

---

## 13. Important Failure Patterns

1. **Severe Central-Tendency Bias:**  
   Laya defaulted to `medium` (INTERESTING) in 68% of inferences. It functions primarily as a cautious classifier that assumes any article mentioning AI has "solid value."
2. **Complete Inability to Filter NOISE:**  
   With a NOISE recall of **0.0%**, Laya cannot be used to drop fluff, marketing promos, or generic roundtables.
3. **Fuzzy Boundary Between "Crucial" and "Solid":**  
   The semantic separation between `high` ("Crucial update, must-read") and `medium` ("Solid value, relevant to practitioners") proved too fine-grained for zero-shot ModernBERT embeddings without fine-tuning on domain-specific importance thresholds.
4. **Vulnerability to Domain Entities:**  
   When articles are sourced from `OpenAI News`, `Google DeepMind`, or `Hugging Face`, Laya's internal prior strongly suppresses `low` or `ignore` regardless of whether the article is an executive hire or an Estée Lauder marketing partnership.

---

## 14. Limitations of this Evaluation

1. **Fixed Prompt Phrasing:** The evaluation used the exact 4-level importance question from Task 1.5B. Alternate prompt phrasings (e.g. binary 2-class triage prompts) were not evaluated.
2. **ArgMax Decision Rule:** The evaluation mapped choices strictly using $\text{ArgMax}$. It did not explore calibrated threshold gating (e.g. passing to Qwen if $P_{\text{high}} > 0.25$).
3. **Bounded Context:** 45 articles had only headlines and links available. While humans easily evaluated these from titles like *"Introducing Gemini 3.8 Flash"*, Laya's performance on headline-only inputs is known to be weaker than on 400-character excerpts.

---

## 15. Conclusion

The empirical measurements from this benchmark demonstrate:

1. **Laya is NOT suitable as an out-of-the-box primary importance gate using argmax classification.**  
   If configured to pass only articles it classifies as `high` (MUST_KNOW), the pipeline would **drop 60.94% of human-labeled must-know articles**, representing an unacceptable intelligence failure.
2. **Laya cannot serve as an effective noise filter.**  
   Because Laya achieved **0% recall on NOISE**, an upstream gate configured to discard `low`/`ignore` would filter out **zero** noise articles while dropping legitimate releases like Gemini 3 Pro Image (Article 3549).
3. **Operational Speed is Confirmed:**  
   Inference speed (256 ms average) and low resource consumption (~2.8 GB RAM) fulfill all performance prerequisites for real-time monitoring, proving that the runtime architecture is viable if classification accuracy can be addressed.
4. **Future Architectural Direction:**  
   To utilize Laya in production, future phases must investigate either:
   - **Threshold-tuned gating:** Using probability cutoffs rather than argmax (e.g. pass if $P_{\text{high}} \ge 0.20$).
   - **Binary specialized prompts:** Formulating a dedicated two-class triage prompt specifically aligned with `docs/importance_rubric.md`.
   - **Retaining Qwen for first-pass filtering:** Until a gating model demonstrates high `MUST_KNOW` recall ($\ge 95\%$), Qwen remains the only reliable extraction layer.
