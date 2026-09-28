# AI News Intelligence Monitor: Article Importance Rubric

**Document Version:** 1.0  
**Phase:** Phase 0.1 — Evaluation Foundation  
**Audience:** Human Labelers, Evaluation Engineers, System Architects  
**Scope:** Human Ground-Truth Annotation for Gating Model Evaluation  
**Status:** Canonical Evaluation Standard (Read-Only Specification)

---

## 1. Purpose & System Context

### 1.1 What This Rubric Is For
This document defines the canonical operational rubric for manually classifying artificial intelligence news articles into three discrete categories: **`MUST_KNOW`**, **`INTERESTING`**, and **`NOISE`**. 

This rubric is designed to be used by human annotators to label a gold-standard reference dataset of approximately 150 real-world database articles. This labeled dataset will serve as the empirical ground truth for evaluating whether lightweight decision models (specifically the local 421M-parameter **Laya** ModernBERT model) can safely act as a high-speed pre-filtering and triage gate in the AI News Intelligence pipeline.

### 1.2 How It Relates to the AI News Intelligence Monitor
The AI News Intelligence Monitor ingests hundreds of raw RSS/Atom feed items daily from diverse sources: foundation research labs (OpenAI, Google DeepMind, Anthropic), open-source repositories and hubs (Hugging Face), community aggregators (Hacker News), and technology journalism (MIT Technology Review).

Today, every ingested article that passes basic deduplication is processed through heavy local LLM analysis (Ollama with Qwen3 4B), which takes 10–15 seconds per article (~20–25 minutes per 100 articles). An upstream triage layer aims to instantly discard irrelevant noise and low-priority chatter in under 400 milliseconds, reserving deep LLM extraction only for articles that deliver meaningful intelligence.

To determine whether an automated gating model is safe and effective, we must first establish what a discerning human expert considers critical versus ignorable.

```
[ Ingested RSS Articles ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│               PHASE 0 EVALUATION SCOPE                 │
│                                                        │
│  Human Annotators Apply THIS Importance Rubric         │
│  to 150 Representative Articles                        │
│                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │  MUST_KNOW   │   │ INTERESTING  │   │   NOISE    │  │
│  └──────────────┘   └──────────────┘   └────────────┘  │
│         │                  │                  │        │
│         └──────────────────┼──────────────────┘        │
│                            ▼                           │
│       Gold-Standard Benchmark Reference Set            │
│                            │                           │
│                            ▼                           │
│      Compare Laya Predictions vs. Human Labels         │
│     (Measure MUST_KNOW Recall & Noise Rejection)       │
└────────────────────────────────────────────────────────┘
```

### 1.3 Critical Distinction: Ground Truth vs. Production Ranking
- **This Rubric is NOT a Production Ranking Formula:**  
  Production ranking in this repository is governed by the deterministic 5-factor scoring engine in [`ranking/scorer.py`](file:///Users/apple/Desktop/ai_news_digest/ranking/scorer.py) ($\text{Final Score} = 0.40 \cdot \text{relevance} + 0.25 \cdot \text{novelty} + 0.20 \cdot \text{recency} + 0.15 \cdot \text{confidence}$). That formula computes a continuous float $\in [0.0, 1.0]$ with time-decay for dashboard ordering and Telegram delivery.
- **This Rubric IS an Evaluation Ground Truth:**  
  This rubric produces static, time-independent, categorical judgments representing the intrinsic informational value of an article at its moment of publication. It establishes the benchmark against which automated gating and filtering decisions are quantitatively audited.

---

## 2. Operational Definitions of the Three Labels

To achieve high inter-annotator reliability (ensuring two independent annotators assign identical labels to the same article), labels must not rely on subjective impressions of "usefulness." They are grounded in concrete operational criteria:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             LABEL SPECTRUM                                  │
├──────────────────────────┬────────────────────────────┬─────────────────────┤
│        MUST_KNOW         │        INTERESTING         │        NOISE        │
├──────────────────────────┼────────────────────────────┼─────────────────────┤
│ Frontier Capability      │ Incremental Progress       │ Zero Technical News │
│ Workflow Altering        │ Domain-Specific Depth      │ Speculative Opinion │
│ Paradigm Shift           │ Educational / Instructive  │ Repetitive Fluff    │
│ Major Vulnerability      │ Niche Tool / Library       │ SEO / Marketing     │
│ Immediate Attention      │ Passive Awareness          │ Safe to Discard     │
└──────────────────────────┴────────────────────────────┴─────────────────────┘
```

### 2.1 MUST_KNOW
An article is **`MUST_KNOW`** if an AI engineer, researcher, technical founder, or machine learning practitioner would be materially disadvantaged or technologically behind if they missed it during their weekly briefing.

**Operational Criteria (At least ONE must apply):**
1. **Frontier Model / Architecture Release:** Launch of a new state-of-the-art foundation model, new weights, or major generational architecture (e.g., GPT-5, Gemini 3, Claude 3.5 Sonnet, Llama 4, DeepSeek V3/R1).
2. **Workflow-Altering Developer Platform:** Release or major revision of a tool, framework, or runtime that changes standard development practices across the industry (e.g., PyTorch 3.0, Hugging Face `tokenizers v1` with 10x throughput, CUDA breakthrough compiler).
3. **Pioneering Research Breakthrough:** Publication of a demonstrable theoretical or empirical paradigm shift (e.g., FlashAttention, LoRA, chain-of-thought prompting origin, DeepSeek architectural efficiency).
4. **Critical Safety, Security, or Vulnerability Incident:** Exploitable zero-day in widely used AI infrastructure, systemic model jailbreak affecting production deployments, or critical model poisoning incident.
5. **Macro Industry or Regulatory Disruption:** Sweeping legal, corporate, or geopolitical actions that directly dictate how models are built, trained, licensed, or deployed (e.g., export ban on key GPUs, landmark copyright injunction halting training, major lab acquisition/restructuring).

*Key Test:* "Does this news force teams to re-evaluate their tech stack, roadmap, architecture, or operational risk?"

### 2.2 INTERESTING
An article is **`INTERESTING`** if it provides substantive, verified technical value, context, or domain insight, but does not disrupt broad industry workflows or demand immediate strategic realignment.

**Operational Criteria (At least ONE must apply):**
1. **Solid Research & Incremental Papers:** Legitimate academic research offering valid empirical findings, optimizations, or niche architectures that advance specialized subfields (e.g., a 2% accuracy improvement on visual document QA, novel loss formulation for 3D Gaussian splatting).
2. **Valuable Tooling & Library Updates:** Meaningful releases of utility packages, developer extensions, UI wrappers, or specialized libraries that are useful to specific teams but not foundational to the entire industry (e.g., a new LangChain community connector, a CLI prompt tester).
3. **Technical Case Studies & Post-Mortems:** Substantive engineering blogs describing how a company scaled an inference cluster, debugged distributed training, or optimized context retrieval (e.g., Rakuten reducing MTTR with Codex, Discord scaling embeddings).
4. **Ecosystem & Industry Dynamics:** Standard corporate partnerships, series A/B venture rounds, talent moves, or minor hardware roadmaps that provide valuable situational awareness.
5. **High-Quality Technical Deep Dives & Explanations:** Rigorous analyses breaking down existing complex models, reverse-engineering bot behavior, or dissecting a paper's mechanics.

*Key Test:* "Is this informative, technically sound, and worth reading for awareness, without requiring immediate operational changes?"

### 2.3 NOISE
An article is **`NOISE`** if it contains no meaningful new technical development, offers zero actionable information for practitioners, or consists primarily of generic commentary, recycled PR, or speculative fluff.

**Operational Criteria (Any of the following apply):**
1. **Generic Speculation & Think-Pieces:** Unsubstantiated op-eds, philosophical musings, existential doomerism/utopianism without empirical data, or standard "How AI Will Change Marketing/Law/Finance" articles.
2. **SEO Content & List-icles:** Low-effort list-icles ("Top 10 AI Tools to Boost Productivity in 2026", "Best AI Prompts for Resumes"), affiliate roundups, and generic clickbait.
3. **Routine Maintenance & Trivial Updates:** Patch releases with minor bug fixes (v2.1.3 fixing typo in docstring), cosmetic UI changes, minor website redesigns, or routine server status updates.
4. **Corporate Marketing & Pure PR:** Press releases announcing vague "commitments to AI excellence," ceremonial partnerships with no technical deliverables, or re-announcements of existing tools under new branding.
5. **Zero Substance / Content Truncation:** Scraped links where the body contains only cookie banners, login prompts, subscription paywalls without summaries, or empty RSS placeholders with no discernable information.
6. **Tangential Mentions:** General technology, business, or gaming articles that mention AI in passing without presenting any AI-centric development (e.g., a stock market wrap-up mentioning NVIDIA's share price movement).

*Key Test:* "Would an AI practitioner lose anything of value if this article were permanently deleted and never seen?"

---

## 3. Step-by-Step Decision Procedure for Human Labelers

When annotating an article from the database, follow this sequential 5-step procedure. Do not skip steps.

```mermaid
flowchart TD
    Start([Read Title, Source, & Article Content]) --> S1{Step 1: Substance Check<br/>Does article contain substantive,<br/>coherent information?}
    S1 -- No --> L_Noise[Assign NOISE]
    S1 -- Yes --> S2{Step 2: AI Core Check<br/>Is the primary subject directly<br/>about AI models, code, data,<br/>hardware, or research?}
    S2 -- No --> L_Noise
    S2 -- Yes --> S3{Step 3: New Development Check<br/>Does it report a real event, release,<br/>finding, or substantive case study?<br/>(Not generic op-ed / SEO fluff)}
    S3 -- No --> L_Noise
    S3 -- Yes --> S4{Step 4: Paradigm Shift Check<br/>Is it a frontier model, industry-altering tool,<br/>major breakthrough, critical CVE,<br/>or macro disruption?}
    S4 -- Yes --> L_MustKnow[Assign MUST_KNOW]
    S4 -- No --> S5{Step 5: Awareness Value Check<br/>Does it offer concrete technical utility,<br/>solid research, or instructive case study?}
    S5 -- Yes --> L_Interesting[Assign INTERESTING]
    S5 -- No --> L_Noise
```

### Step 1: Substantive Content Check
Inspect the title and extracted article body (e.g., from `content` or `summary`).
- If the content is an empty string, an uninformative error page ("Please enable JavaScript", "Subscribe to continue"), or less than 2 coherent sentences with no factual assertions: **Assign `NOISE`**.
- Otherwise, proceed to Step 2.

### Step 2: Primary AI Subject Check
Determine whether artificial intelligence is the *core subject* or merely a passing keyword.
- If AI is only referenced tangentially (e.g., a macroeconomics piece on inflation mentioning tech stocks, a consumer gadget review where a phone camera has an "AI filter"): **Assign `NOISE`**.
- If the development directly concerns AI architectures, training, models, evaluation, tooling, infrastructure, or AI regulation: proceed to Step 3.

### Step 3: Verifiable Development vs. Fluff Check
Examine whether the text communicates a specific, verifiable new event or technical artifact, or if it is purely speculative commentary.
- If the article is an opinion column, an interview asking generic questions, a rehashed high-level overview of known facts, or marketing hype with no technical substance: **Assign `NOISE`**.
- If the article describes a concrete artifact (a released model, code repository, paper preprint, benchmark result, CVE security disclosure, corporate funding round, or production case study): proceed to Step 4.

### Step 4: The Blast Radius & Workflow Test (MUST_KNOW Evaluation)
Evaluate the impact and scope of the development:
- Does this article report on:
  - A frontier foundation model or significant open weights (e.g., OpenAI o3, Gemini Live, Llama 4)?
  - A foundational infrastructure or tooling leap (e.g., FlashAttention, Triton release, Tokenizers v1)?
  - A major theoretical or architectural discovery that alters current scaling/inference assumptions?
  - A critical vulnerability, exploit, or regulatory shock?
- **If YES $\to$ Assign `MUST_KNOW`**.
- If NO $\to$ proceed to Step 5.

### Step 5: Informational & Technical Utility Test (INTERESTING Evaluation)
Evaluate whether the development is worth knowing for an engineer working in or tracking the subfield:
- Does it present a solid research paper, a useful niche library, an informative technical post-mortem, an enterprise adoption case study, or a meaningful community development?
- **If YES $\to$ Assign `INTERESTING`**.
- **If NO $\to$ Assign `NOISE`** (e.g., routine patch, minor conference announcement, minor marketing event).

---

## 4. Classification Criteria by Category

The following table provides explicit operational rules for 20 common content categories encountered in the intelligence pipeline:

| Category | Typical Label | Explicit Criteria & Exceptions |
|:---|:---:|:---|
| **Major Foundation Model Releases** | `MUST_KNOW` | Initial launch or major version release of flagship models (OpenAI GPT/o-series, Google Gemini, Anthropic Claude, Meta Llama, DeepSeek). Even if third-party reported. |
| **Major Model Updates / Checkpoints** | `MUST_KNOW` or `INTERESTING` | `MUST_KNOW` if introducing major new capabilities (e.g., reasoning/thinking mode, native audio/vision, doubling context to 1M). `INTERESTING` if routine fine-tune or minor checkpoint refresh. |
| **New AI Products & Platforms** | `INTERESTING` | Standard enterprise product launches (e.g., "Company X adds an AI assistant"). Upgraded to `MUST_KNOW` only if redefining a major category (e.g., original launch of ChatGPT or GitHub Copilot). |
| **Developer Tools & Libraries** | `INTERESTING` | High-quality open-source packages, agent frameworks, visualization tools. Upgraded to `MUST_KNOW` if it becomes ubiquitous standard infrastructure (e.g., Hugging Face Transformers v1, vLLM initial release). |
| **Open-Source Weight Drops** | `MUST_KNOW` or `INTERESTING` | `MUST_KNOW` if competitive with proprietary frontier models (e.g., Llama 70B, DeepSeek R1). `INTERESTING` if domain-specific, small scale (<7B without novel architecture), or synthetic dataset release. |
| **Major Research Breakthroughs** | `MUST_KNOW` | Papers introducing paradigm shifts (e.g., Transformer attention mechanism, LoRA, DPO, Mixture-of-Depths). Must demonstrate reproducible, substantial architectural advantages. |
| **Ordinary Research Papers** | `INTERESTING` | Standard arXiv preprints with sound methodology, 1–5% incremental benchmark gains, or specialized ablation studies. Default is `INTERESTING`. |
| **Benchmarks & Evaluations** | `INTERESTING` | Comprehensive new benchmark suites (e.g., SWE-bench, Humanity's Last Exam) or major leaderboard reshuffles. Minor evaluations on private datasets are `NOISE`. |
| **AI Safety & Security Incidents** | `MUST_KNOW` or `INTERESTING` | `MUST_KNOW` for active remote-code execution in widely deployed ML packages, major model exfiltration, or systemic prompt injections in production systems. `INTERESTING` for theoretical/academic jailbreak papers. |
| **Corporate Announcements** | `INTERESTING` or `NOISE` | Standard executive hires, minor partnerships, or office openings are `NOISE`. Major strategic pivots or restructuring of leading frontier labs are `INTERESTING`. |
| **Acquisitions & Mergers** | `INTERESTING` or `MUST_KNOW` | Most talent/tech acquisitions are `INTERESTING`. Upgraded to `MUST_KNOW` only if it consolidates dominant market power or drastically alters foundation model availability. |
| **Venture Funding & Valuations** | `INTERESTING` or `NOISE` | Seed/Series A rounds < $50M are `NOISE` unless technical breakthrough is detailed. Mega-rounds ($\ge \$500\text{M}$) for frontier labs (Anthropic, xAI, Mistral) are `INTERESTING`. |
| **Hardware & Infrastructure** | `MUST_KNOW` or `INTERESTING` | `MUST_KNOW` for new accelerator architectures (NVIDIA Blackwell, Google TPU v6) or cluster interconnect breakthroughs. `INTERESTING` for routine server rack specs or minor hardware benchmarks. |
| **Developer Ecosystem Changes** | `MUST_KNOW` or `INTERESTING` | `MUST_KNOW` if major providers slash API pricing by 50%+, deprecate critical endpoints, or drastically alter Terms of Service. `INTERESTING` if adding localized data centers. |
| **Tutorials & How-To Guides** | `INTERESTING` or `NOISE` | In-depth engineering walkthroughs (e.g., "Building an FP8 kernel from scratch in Triton") are `INTERESTING`. Generic "Getting Started with ChatGPT Prompts" are `NOISE`. |
| **Opinion & Editorial Content** | `NOISE` | Default is `NOISE`. Substantive, mathematically rigorous essays by renowned pioneers (e.g., LeCun, Karpathy) framing major architectural debates can be `INTERESTING`. Never `MUST_KNOW`. |
| **General AI News & Journalism** | `NOISE` or `INTERESTING` | Mainstream non-technical articles ("Will AI take our jobs?") are `NOISE`. Well-researched investigative reporting uncovering systemic corporate misconduct or supply chain fraud is `INTERESTING`. |
| **Minor Updates & Patches** | `NOISE` | Minor version bumps (v1.4.2), documentation fixes, localized typo repairs, and standard dependency maintenance are always `NOISE`. |
| **Duplicate Event Coverage** | Evaluate Independently | Label the information present in the specific article without penalizing it for being redundant to earlier articles. (See Section 7). |
| **Announcements vs. Meta-Analysis** | Differentiated | The primary announcement of an event receives its natural label (e.g., `MUST_KNOW`). Later meta-commentary discussing the announcement is usually `INTERESTING` or `NOISE` unless introducing new empirical benchmarks. |

---

## 5. Borderline Cases & Explicit Tie-Breaking Rules

In real-world labeling, annotators encounter borderline cases at category boundaries. To prevent subjective drift, apply these explicit tie-breaking rules:

### 5.1 Borderline Case: MUST_KNOW vs. INTERESTING
*The Dilemma:* An article describes a genuinely impressive development, but the annotator is uncertain whether it is truly critical or simply very interesting.

```
                  MUST_KNOW vs. INTERESTING DECISION MATRIX
                  
     Can an AI engineer defer reading this for 2 weeks
     without experiencing workflow or architectural obsolescence?
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
             [YES]                      [NO]
               │                         │
     Assign INTERESTING          Assign MUST_KNOW
```

**Explicit Tie-Breaking Rules:**
1. **The "Workflow Disruption" Rule:** If the development changes how developers build software *today* (e.g., new state-of-the-art weights released, a breaking API deprecation, or a 10x faster core library), classify as **`MUST_KNOW`**. If it is a proof-of-concept that will take years to commercialize or affects only a narrow niche, classify as **`INTERESTING`**.
2. **The "Frontier SOTA" Rule:** If a model claims State-of-the-Art performance across universally tracked benchmarks (MMLU, SWE-bench, GSM8K, MATH) and is backed by a credible lab or open weights, classify as **`MUST_KNOW`**. If it achieves SOTA only on a narrow sub-benchmark (e.g., Hungarian medical text translation), classify as **`INTERESTING`**.
3. **The "Sub-Checkpoint" Rule:** Base model releases (e.g., Llama 3) are **`MUST_KNOW`**. Routine community fine-tunes or quantized variants (e.g., Llama-3-8B-Instruct-GGUF-Q4) are **`INTERESTING`**.
4. **Tie-Break Default for Gating Safety:** When genuinely stuck between `MUST_KNOW` and `INTERESTING`, **assign `MUST_KNOW`**. (See Section 12 for the rationale on false negatives).

### 5.2 Borderline Case: INTERESTING vs. NOISE
*The Dilemma:* An article contains some technical terms or discusses a legitimate company, but has very little depth or actionable content.

```
                   INTERESTING vs. NOISE DECISION MATRIX
                   
        Does the article contain at least ONE verifiable technical
        fact, metric, code artifact, or substantive case study?
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
             [YES]                      [NO]
               │                         │
     Assign INTERESTING             Assign NOISE
```

**Explicit Tie-Breaking Rules:**
1. **The "Extractable Fact" Rule:** Ask: *"Can I extract at least one concrete technical learning, architecture detail, performance number, or verifiable event from this article?"* If yes, classify as **`INTERESTING`**. If the article merely repeats high-level sentiment, re-announces known products without detail, or shares subjective opinions, classify as **`NOISE`**.
2. **The "Customer Case Study" Rule:** A corporate case study describing a specific deployment with concrete architectural details or measurable ROI (e.g., "How Rakuten integrated Codex into CI/CD and reduced MTTR by 50%") is **`INTERESTING`**. A generic sales puff piece ("How Company X transformed its culture with AI") with zero technical metrics is **`NOISE`**.
3. **The "Preprint Rigor" Rule:** An arXiv preprint with clear math, architecture diagrams, or empirical tables is **`INTERESTING`**. An unvetted preprint that makes extraordinary claims (e.g., "We solved AGI with 10 lines of Python") with no benchmark code, weights, or verifiable rigor is **`NOISE`**.
4. **Tie-Break Default:** When genuinely stuck between `INTERESTING` and `NOISE`, ask whether reading the article was a waste of 3 minutes. If it provided zero new information, **assign `NOISE`**.

---

## 6. Publisher and Source Independence

A core principle of this rubric is **Source Independence**:
> **The label must reflect the intrinsic significance of the underlying information, NOT the prestige or reputation of the publishing outlet.**

### 6.1 The "Blind Source" Principle
Imagine the article text was published on an anonymous plain-text bulletin board. Does the development itself matter to the AI community?

- **High-Prestige Outlet with Low-Value Content $\to$ NOISE:**  
  Prestigious publications (e.g., MIT Technology Review, The New York Times, Nature) regularly publish subscriber roundtables, introductory explainers for general audiences, or non-technical editorial commentary. These articles do not become `MUST_KNOW` simply because of their masthead.
  - *Example:* An MIT Technology Review article titled *"Could AI really kill us all? Your questions, answered"* is an editorial Q&A with subscriber questions. Despite coming from a tier-1 outlet, it contains no new technical development or policy action $\to$ **`NOISE`** (or at best borderline `INTERESTING` for general awareness).
- **Independent / Obscure Outlet with High-Value Development $\to$ MUST_KNOW:**  
  A major frontier announcement or critical vulnerability broken by an independent researcher's blog, a personal Substack, or an anonymous GitHub gist is `MUST_KNOW` if the underlying artifact is real.
  - *Example:* An independent security researcher discloses a working remote code execution vulnerability in PyTorch on a personal blog $\to$ **`MUST_KNOW`**.
- **Secondary News Reporting Frontier Drops $\to$ MUST_KNOW:**  
  If TechCrunch or a third-party blog reports that Anthropic launched Claude 3.7 Sonnet with hybrid reasoning capabilities, the article is **`MUST_KNOW`**, even if it is not the primary press release from Anthropic.

---

## 7. Duplicate Coverage and Multi-Outlet Handling

In live RSS feeds, major announcements (e.g., a new OpenAI model or European AI Act enforcement) trigger dozens of articles across multiple feeds within hours.

### 7.1 The "Island" Evaluation Rule
For this evaluation dataset:
1. **Evaluate Each Article Individually:** Annotate the article based strictly on the information contained within that single document.
2. **Do NOT Artificially Downgrade Duplicates:** Do not label the second or third article covering a major model launch as `NOISE` simply because an earlier article already reported it. If an article accurately reports that DeepSeek-R1 was released with full weights and competitive reasoning benchmarks, that article is **`MUST_KNOW`**, regardless of whether three other feeds in the dataset also reported it.
3. **Do NOT Assume Upstream Clustering Exists:** The purpose of Phase 0 is to evaluate whether a gating model (Laya) can recognize the *intrinsic importance* of an incoming article. Deduplication and story clustering are handled in separate pipeline stages ([`deduplication.py`](file:///Users/apple/Desktop/ai_news_digest/deduplication.py)). Artificially downgrading duplicates in the evaluation dataset would corrupt the model's training signal for recognizing important topics.

---

## 8. Evidence Requirements and Dealing with Uncertainty

Labelers must ground their evaluations strictly in the text provided in the article record (enriched text or summary).

### 8.1 Evidence Standards
- **Sufficient Evidence:** The article explicitly cites concrete model names, benchmark numbers, repository links, direct quotes from creators, or verifiable policy documents.
- **Speculative Claims / Rumors:** Articles discussing unverified rumors (e.g., "Leaker claims GPT-5 will launch next Tuesday with 10 trillion parameters") must be classified as **`NOISE`** unless the rumor itself triggered an official confirmation or verified market reaction.
- **Marketing Superlatives:** Disregard self-serving corporate superlatives ("revolutionary", "unprecedented", "game-changing"). Focus exclusively on what was actually built, measured, or proven.

### 8.2 Handling Truncated or Missing Content
In our SQLite database, some articles have full scraped `content`, while others have only an RSS `summary`.
- If an article's `summary` clearly announces a definitive major event (e.g., *"OpenAI announces GPT-4o with real-time audio and vision capabilities, rolling out free to all users"*), the title and summary provide sufficient evidence to assign **`MUST_KNOW`**.
- If the text is so severely truncated that it is impossible to determine what happened (e.g., *"Click here to read more..."*), and enrichment has failed: **Assign `NOISE`** (unusable content cannot be intelligence).

---

## 9. Explicit Exclusions (Default NOISE Rules)

The following archetypes must be classified as **`NOISE`** by default, unless they meet the explicit override conditions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXPLICIT NOISE ARCHETYPES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Generic AI Op-Eds ("Will AI replace software engineers?")                │
│ 2. Clickbait & Productivity Listicles ("10 Prompts to 10x Your Life")       │
│ 3. Corporate Executive Appointments & Office Expansions                     │
│ 4. Minor Patch Releases & Dependency Maintenance Bumps                      │
│ 5. Stock Market & Valuation Chatter with No Technical Context               │
│ 6. Vague Corporate "AI Strategy" Pledges without Deliverables               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Override Condition
An excluded archetype can only be elevated to `INTERESTING` or `MUST_KNOW` if the article introduces a **concrete, verifiable technical artifact or landmark precedent**:
- *Example:* An executive departure article is normally `NOISE`. However, if the Chief Scientist of a frontier lab resigns specifically over verified safety protocol violations and releases an open letter detailing suppressed risk evaluations, the event elevates to **`MUST_KNOW`**.

---

## 10. Consistency and Inter-Rater Reliability Rules

To ensure high consistency between different human labelers, follow these five golden rules:

1. **The "No Prior Knowledge" Rule:** Judge the article based on what an informed reader learns from the text and its verified context at publication time. Do not penalize an article because a technology later failed six months after publication.
2. **The "Dispassionate Engineer" Perspective:** Adopt the viewpoint of a pragmatic senior AI systems engineer. Do not let personal philosophical stances (e.g., strict open-source advocacy vs. proprietary safety caution) influence whether an event is important.
3. **The "Actionability" Anchor:** When torn between categories, ask what action a reader takes:
   - Must change technical roadmap or inspect code immediately? $\to$ **`MUST_KNOW`**
   - Files away in mental knowledge base for reference? $\to$ **`INTERESTING`**
   - Closes tab immediately and forgets? $\to$ **`NOISE`**
4. **Distinguish Writing Style from Event Substance:** A poorly written or dry press release announcing a breakthrough model is still **`MUST_KNOW`**. A beautifully written, evocative essay reflecting on AI consciousness that contains zero technical facts is still **`NOISE`**.
5. **Self-Calibration Check:** After labeling 20 articles, check your distribution. In a natural, uncurated AI feed, an expected healthy distribution is approximately:
   - **`MUST_KNOW`**: 10% – 20%
   - **`INTERESTING`**: 30% – 45%
   - **`NOISE`**: 40% – 55%  
   If your `MUST_KNOW` count exceeds 35%, your threshold is too lenient. If your `NOISE` count is below 25%, you are tolerating low-signal fluff.

---

## 11. Concrete Reference Examples

The following 18 diverse scenarios illustrate the operational boundaries across various news categories:

| # | Article Headline & Core Context | Category | Label | Operational Rationale |
|:---|:---|:---|:---:|:---|
| 1 | *Meta Releases Llama 3 70B and 405B with Open Weights and Technical Report* | Open Weights | **`MUST_KNOW`** | Frontier open-weights release rivaling proprietary frontier models; immediate industry-wide fine-tuning and deployment impact. |
| 2 | *Hugging Face Releases tokenizers v1: 10x Speedup via Workspace Splitting* | Developer Tool | **`MUST_KNOW`** | Foundational ML infrastructure release with order-of-magnitude performance gain affecting almost all transformer training/inference pipelines. |
| 3 | *Critical RCE Vulnerability Discovered in Ray Distributed Cluster Management (CVE-2023-48022)* | AI Security | **`MUST_KNOW`** | Actively exploited remote code execution vulnerability in widely adopted AI cluster infrastructure; requires immediate patching. |
| 4 | *DeepSeek Introduces Multi-Head Latent Attention (MLA) in DeepSeek-V2 Paper* | Research | **`MUST_KNOW`** | Novel architectural breakthrough dramatically compressing KV-cache memory during inference with minimal quality loss. |
| 5 | *OpenAI Announces SearchGPT Prototype with Real-Time Web Indexing* | Product | **`MUST_KNOW`** | Major competitive shift in search and retrieval architectures by leading frontier lab. |
| 6 | *Rakuten Fixes Engineering Issues Twice as Fast Using OpenAI Codex API* | Case Study | **`INTERESTING`** | Substantive enterprise case study detailing concrete metrics (50% MTTR reduction), but is a customer story, not a new tool release. |
| 7 | *Benchmarking Long-Context Retrieval Across 12 Open-Weight LLMs* | Evaluation | **`INTERESTING`** | High-utility empirical evaluation providing actionable guidance for RAG practitioners, but does not introduce a new model. |
| 8 | *Fine-Tuning Mistral 7B on Financial Earnings Calls Using QLoRA* | Tutorial / How-To | **`INTERESTING`** | Solid, reproducible technical walkthrough with code snippets; useful to domain practitioners, but not an industry-wide event. |
| 9 | *An AI Bot Appeared on the Competitive StarCraft Ladder: Reverse-Engineering Replays* | Analysis | **`INTERESTING`** | Fascinating technical investigation into player fingerprinting and automated game agents; strong technical depth but niche scope. |
| 10 | *AI Startup Celestial Raises $35M Series A to Build Specialized Chip Interconnects* | Venture Funding | **`INTERESTING`** | Meaningful funding for hardware infrastructure addressing interconnect bottlenecks, but hardware is pre-silicon. |
| 11 | *FlashAttention-3 Released: Tripling Training Speed on Hopper GPUs via Asynchronous Warp Scheduling* | Compute / Infra | **`MUST_KNOW`** | Major architectural breakthrough in kernel computation that immediately alters frontier training efficiency. |
| 12 | *Could AI Really Kill Us All? Your Questions Answered by MIT Tech Review Panel* | Editorial Q&A | **`NOISE`** | Conversational subscriber roundtable discussing speculative risks; zero technical data, code, or verified policy events. |
| 13 | *Top 15 ChatGPT Prompts That Will Supercharge Your Marketing Workflow in 2026* | SEO Listicle | **`NOISE`** | Low-information, non-technical productivity fluff aimed at consumer users. |
| 14 | *LangChain Version 0.2.14 Bugfix Release (Resolves KeyError in BasePromptTemplate)* | Minor Patch | **`NOISE`** | Routine minor bug fix; normal library maintenance with zero strategic importance. |
| 15 | *Tech Industry Executive Argues AI is Transforming Corporate Culture in Keynote Speech* | Generic News | **`NOISE`** | Generic corporate speech; no technical deliverables, code, or measurable breakthroughs. |
| 16 | *Anthropic Slashes Claude 3 Haiku Input Pricing by 50%* | Ecosystem Pricing | **`MUST_KNOW`** | Massive cost reduction on high-volume production model; directly changes unit economics of production agent architectures. |
| 17 | *Evaluating the Impact of Dropout on Multimodal Cross-Attention in Vision-Language Models* | Research Paper | **`INTERESTING`** | Legitimate, sound academic research paper; valuable for model architects, but an incremental ablation study. |
| 18 | *Stock Market Report: NVIDIA Rises 2.3% as S&P Tech Sector Posts Gains* | Financial News | **`NOISE`** | Routine daily equity market fluctuation; zero technical AI developments. |

---

## 12. Evaluation Philosophy: Why MUST_KNOW Recall is King

When using human labels to evaluate an automated triage/gating model (such as Laya), the cost of classification errors is **highly asymmetric**:

```
                       ERROR ASYMMETRY IN GATING
                       
                   Actual Label: MUST_KNOW
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Model Predicts: NOISE            Model Predicts: PASS (Must-Know)
   ═════════════════════            ═════════════════════════════════
   FATAL TYPE II ERROR              CORRECT DECISION
   - Article is dropped             - Sent to Qwen for analysis
   - Critical intelligence missed   - Highlighted in daily digest
   - User is blindsided             - Intelligence delivered
   - Pipeline FAILS primary mission
```

### 12.1 The Fatal Cost of False Negatives (Type II Error)
- If an automated gate mistakenly marks a **`MUST_KNOW`** article as `NOISE`, the article is discarded or suppressed without downstream LLM analysis.
- The user of the AI News Intelligence Monitor suffers a **silent intelligence failure**: a major foundation model release or critical vulnerability occurred, and the monitor failed to alert them.
- Therefore, **`MUST_KNOW` Recall** ($\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$) is the primary governing safety metric of any gating model. A triage gate is acceptable only if its `MUST_KNOW` recall is near 100% (target $\ge 98\%$).

### 12.2 The Minor Cost of False Positives (Type I Error)
- If an automated gate mistakenly marks a **`NOISE`** or **`INTERESTING`** article as `MUST_KNOW`, the article is simply forwarded to Stage 2 (Ollama Qwen3 4B).
- In Stage 2, Qwen performs detailed structured extraction, and [`ranking/scorer.py`](file:///Users/apple/Desktop/ai_news_digest/ranking/scorer.py) assigns deterministic relevance, novelty, and recency scores. Low-value articles are filtered or pushed to the bottom of the dashboard.
- The only penalty for a false positive is a small amount of extra local compute time (~10 seconds on GPU).

### 12.3 What This Implies for Evaluation
1. When evaluating a gating model against this human rubric, the primary threshold will be tuned for high sensitivity / recall.
2. Labelers must be disciplined: do not artificially elevate `NOISE` into `MUST_KNOW` out of caution, but do ensure that anything with legitimate industry-shifting potential is labeled `MUST_KNOW` so the model's safety perimeter can be tested accurately.
3. This rubric does not assume that Laya or any other model is already accurate; it exists to create an objective yardstick to determine *whether* and *where* the model fails.

---

## 13. Relationship to Existing System Architecture

This rubric introduces an evaluation standard without altering any existing production software components. The table below clarifies how this rubric interfaces with existing database fields, ranking scoring, and future pipeline stages:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ARCHITECTURAL ALIGNMENT MATRIX                                  │
├──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────────┤
│ Dimension            │ Phase 0 Evaluation   │ Existing Production  │ Future Production        │
│                      │ Human Ground Truth   │ Pipeline (Current)   │ Triage Gate (Proposed)   │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Role**             │ Benchmark Reference  │ Feature Extraction   │ Fast Pre-Filter          │
│                      │ Dataset (150 rows)   │ & Deterministic Rank │ Gating Layer             │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Engine**           │ Human Domain Expert  │ Ollama Qwen3 4B      │ Laya ModernBERT-large    │
│                      │ Manual Annotation    │ + ranking/scorer.py  │ (Apple Silicon MPS)      │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Output Type**      │ Categorical Enum:    │ Structured JSON +    │ Categorical Logits +     │
│                      │ - MUST_KNOW          │ Float [0.0, 1.0]     │ Calibrated Probabilities │
│                      │ - INTERESTING        │ in article_analysis  │ (e.g. P(high) > 0.40)    │
│                      │ - NOISE              │                      │                          │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Execution Time**   │ Manual (~1 min/item) │ 10 – 15 seconds      │ 300 – 400 milliseconds   │
│                      │                      │ per article          │ per batch                │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Database Impact**  │ Stored in dedicated  │ Written to           │ Potential future flag    │
│                      │ evaluation benchmark │ article_analysis     │ in articles table        │
│                      │ JSON/CSV file        │ SQLite table         │ (NOT implemented now)    │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────┤
│ **Code Status**      │ Specification only   │ Active Production    │ Under Investigation      │
│                      │ (This Document)      │ Codebase             │ (Task 1.5B / ADR 1.6)    │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────────┘
```

### Architectural Guarantees
- **No Ranking Changes:** The deterministic ranking formula ($w_r \cdot \text{rel} + w_n \cdot \text{nov} + w_t \cdot \text{rec} + w_c \cdot \text{conf}$) in [`ranking/scorer.py`](file:///Users/apple/Desktop/ai_news_digest/ranking/scorer.py) remains unchanged.
- **No Schema Changes:** The SQLite schema in [`database.py`](file:///Users/apple/Desktop/ai_news_digest/database.py) remains unchanged.
- **No Pipeline Disruption:** Ingestion, enrichment, Qwen batch analysis, and Telegram delivery continue operating normally.

---

## 14. Document Change Control

| Version | Date | Author / Role | Changes / Notes |
|:---|:---|:---|:---|
| `1.0` | 2026-09-28 | Systems Architecture | Initial canonical specification for Task Phase 0.1. |
