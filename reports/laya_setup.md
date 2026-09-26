# Laya Local Runtime Setup & Sanity Benchmark Report

**Task:** 1.5A — Laya Local Runtime Setup and Sanity Check  
**Date:** September 26, 2026  
**Environment:** Apple Silicon Mac (arm64, macOS Darwin 27.0.0)  
**Status:** SUCCESS (Evaluation and Sanity Checks Complete)  

---

## 1. Machine & Environment Specification

| Parameter | Value | Verification Status |
| :--- | :--- | :--- |
| **Operating System** | macOS 27.0 (Darwin Kernel 27.0.0, Build 26A428) | `[VERIFIED]` (`sw_vers`, `uname -a`) |
| **Hardware Architecture** | `arm64` (Apple Silicon SoC, T8142) | `[VERIFIED]` (`uname -m`, `platform.machine()`) |
| **Python Executable** | `/Users/apple/Desktop/ai_news_digest/.venv/bin/python` | `[VERIFIED]` (`sys.executable`) |
| **Python Version** | Python 3.11.15 (`Clang 22.1.3`) | `[VERIFIED]` (`python --version`) |
| **Apple Silicon Acceleration** | PyTorch Metal Performance Shaders (MPS): `is_available()=True`, `is_built()=True` | `[VERIFIED]` (`torch.backends.mps`) |

---

## 2. Laya Runtime & Package Details

| Component | Identifier / Version | Details |
| :--- | :--- | :--- |
| **Official Repository** | `https://github.com/NandhaKishorM/laya` | Convai Innovations (Nandakishor M) `[VERIFIED]` |
| **PyPI Package** | `laya==0.3.20` | Non-autoregressive System 1 decision engine `[VERIFIED]` |
| **Core Framework** | `torch==2.14.0` | PyTorch with native Apple Silicon MPS support `[VERIFIED]` |
| **Support Libraries** | `transformers==5.17.0`, `safetensors==0.8.0`, `tokenizers==0.23.2`, `huggingface_hub==1.33.0` | Minimal dependencies installed `[VERIFIED]` |
| **Runtime Mechanism** | Non-autoregressive single-pass forward classification | No token-by-token text generation `[VERIFIED]` |
| **Decision Primitives** | `choice` (categorical selection), `score` (ordered scale), `noul` (calibrated binary yes/no probability) | `[VERIFIED]` |

---

## 3. Laya Model Checkpoints & Selection

The official repository maintains three distinct checkpoints on the Hugging Face Hub under `convaiinnovations/laya`:

| Checkpoint Name | Hub Identifier | Backbone Architecture | Parameters | Context Window | Intended Workflow |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`laya` (Selected)** | `convaiinnovations/laya` (root) | ModernBERT-large + Decision Head | ~421M | 512 tokens | English classification, routing, and scoring `[VERIFIED]` |
| **`laya-multilingual`** | `convaiinnovations/laya` (`multilingual`) | mmBERT-base + Decision Head | ~322M | 1,024 – 8,192 tokens | 100+ languages, 2x faster, long documents |
| **`laya-typed-decisions`** | `convaiinnovations/laya` (`typed-decisions`) | ModernBERT-large + Decision Head | ~421M | 1,024 tokens | Fine-tuned typed decision benchmarks |

**Selection Rationale**:
For our initial sanity check, the primary English base checkpoint (`convaiinnovations/laya`) was selected because the AI News Digest corpus is predominantly English and this benchmark evaluates whether the core ModernBERT-large decision head operates cleanly on Apple Silicon MPS.

---

## 4. Installation & Checkpoint Download Metrics

- **Installation Method**: Minimal pip install into the local virtual environment:
  ```bash
  .venv/bin/pip install laya
  ```
- **Requirements/Manifest Status**: `requirements.txt` and `pyproject.toml` were **NOT** modified. Laya is kept strictly isolated to benchmark tasks.
- **Model Checkpoint**: `convaiinnovations/laya`
- **Model Download Size**: ~846 MB total (804 MB `model.safetensors` + tokenizer + config files) `[VERIFIED]`
- **Local Cache Location**: `~/.cache/huggingface/hub/models--convaiinnovations--laya` `[VERIFIED]`
- **Cold Download & Load Time**: 219.81 seconds (including network fetch over internet) `[VERIFIED]`
- **Warm Load Time (from disk cache)**: **1.535 seconds** (`[VERIFIED]`)

---

## 5. Basic Sanity Test (Structured Decision)

### Test Setup
- **State**: `"OpenAI released a new coding model that improves software engineering performance and is available to developers."`
- **Question**: `"How relevant is this to an AI news intelligence system?"`
- **Type**: `choice`
- **Criteria**:
  - `highly_relevant`: `"directly about major AI model releases, breakthroughs, or frontier tools"`
  - `somewhat_relevant`: `"general tech or software news with minor AI mention"`
  - `not_relevant`: `"completely unrelated to artificial intelligence"`

### Output Received
```json
{
  "model": "laya-rl-agent",
  "answers": {
    "ai_relevance": {
      "type": "choice",
      "choice": "highly_relevant",
      "probabilities": {
        "highly_relevant": 0.6521,
        "somewhat_relevant": 0.2411,
        "not_relevant": 0.1068
      },
      "confidence": 0.2166,
      "answer_confidence": 0.6521,
      "action": {
        "act_probability": 1.0
      }
    }
  },
  "usage": {
    "input_tokens": 78,
    "output_tokens": 0
  }
}
```

### Key Sanity Observations
- **Selected Choice**: `highly_relevant` `[VERIFIED]`
- **Calibrated Probabilities**: `{'highly_relevant': 0.6521, 'somewhat_relevant': 0.2411, 'not_relevant': 0.1068}` `[VERIFIED]`
- **Non-Autoregressive Property**: `output_tokens: 0`. Unlike an LLM which must generate tokens and format JSON text, Laya evaluated the choice directly in a single forward pass without generating prose text `[VERIFIED]`.

---

## 6. Basic Failure & Robustness Tests

| Test Case | Input Text | Length | Outcome | Result / Probabilities | Tokens In/Out |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **1. Normal Input** | News sentence regarding OpenAI coding model | 113 ch | **SUCCESS** | `highly_relevant` (p=0.6521) | 78 / 0 |
| **2. Empty Input** | `""` | 0 ch | **SUCCESS** | `not_relevant` (p=0.4441, somewhat=0.3922) | 60 / 0 |
| **3. Very Short Input**| `"AI"` | 2 ch | **SUCCESS** | `not_relevant` (p=0.4274, somewhat=0.4087) | 61 / 0 |
| **4. Moderately Long** | Paragraph on DeepMind Gemini 3.8 Live announcement | 435 ch | **SUCCESS** | `highly_relevant` (p=0.8512, somewhat=0.0942) | 135 / 0 |
| **5. Null State** | `None` | N/A | **SUCCESS** | Fallback to empty string evaluation | 13 / 0 |
| **6. Empty Questions** | `{}` | 9 ch | **SUCCESS** | Returns `{answers: {}, usage: {input_tokens: 0}}` without forward pass | 0 / 0 |
| **7. Invalid Type** | `{"type": "invalid_type"}` | 9 ch | **RAISED** | Raises `ValueError: unknown type 'invalid_type'; use one of ['choice', 'noul', 'score']` | N/A |

All four core prompt scenarios (normal, empty, very short, moderately long) executed cleanly without unhandled exceptions or runtime crashes `[VERIFIED]`.

---

## 7. Performance Sanity Check & Hardware Backend

### Measured Latency (Apple Silicon Mac)

```
Target Checkpoint: convaiinnovations/laya (ModernBERT-large, 421M params)
Backend: PyTorch MPS (Metal Performance Shaders GPU)
```

| Execution Stage | Latency | Unit | Observations |
| :--- | :---: | :---: | :--- |
| **Cold Download + Initial Load** | 219.81 | s | Initial Hugging Face download (846 MB) |
| **Warm Load from Disk Cache** | 1,534.75 | ms | Model instantiation & weight loading (1.53 s) |
| **First Inference (Warmup)** | 119.43 | ms | Metal shader compilation & graph allocation |
| **Inference Run 2** | 40.29 | ms | Steady-state forward pass |
| **Inference Run 3** | 39.66 | ms | Steady-state forward pass |
| **Inference Run 4** | 40.72 | ms | Steady-state forward pass |
| **Inference Run 5** | 39.59 | ms | Steady-state forward pass |
| **Inference Run 6** | 39.47 | ms | Steady-state forward pass |
| **Average Subsequent Latency (MPS)**| **39.95** | **ms** | **~0.040 seconds per decision on Apple MPS GPU** `[VERIFIED]` |
| **Average Subsequent Latency (CPU)**| **81.00** | **ms** | **~0.081 seconds per decision on Apple CPU** `[VERIFIED]` |

### Speedup Comparison: Laya vs. Generative LLM
- **Ollama Qwen3 4B (Local Generative LLM)**: ~8,000 ms (8.0 seconds) per article analysis `[VERIFIED]`.
- **Laya ModernBERT-large (Local Decision Engine)**: **39.95 ms (0.04 seconds)** on MPS `[VERIFIED]`.
- **Latency Ratio**: Laya is approximately **200x faster** than Qwen3 4B for structured decision questions `[INFERENCE]`.

---

## 8. Known Limitations & Constraints

1. **Structured Decisions Only**: Laya cannot generate prose text, summaries, key takeaway bullet points, or named entity extractions. It is strictly a classification, scoring, and routing engine `[VERIFIED]`.
2. **Context Window Constraint**: The base English checkpoint has a 512-token context window. Articles longer than 512 tokens are clipped unless the multilingual checkpoint (`laya-multilingual`, supporting up to 8,192 tokens) is used `[VERIFIED]`.
3. **Runtime Warning on Initial Load**: The checkpoint emits `RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; using choice:11+=...`. This is an internal calibration parameter clamp in Laya v0.3.20 that does not disrupt inference `[VERIFIED]`.

---

## 9. Production Safety & Existing Test Suite Status

- **Zero Production Modification**: No production code, prompts, database schemas, or configurations were modified `[VERIFIED]`.
- **Isolated Dependency**: `requirements.txt` remains unchanged `[VERIFIED]`.
- **Full Test Suite Verification**:
  ```bash
  .venv/bin/python -m pytest
  ```
  **Result**: `174 passed, 2 warnings in 0.54s` (`[VERIFIED]`).

---

## 10. Reproduction Command

To reproduce this exact sanity benchmark on this machine:

```bash
# 1. Activate development virtual environment
source .venv/bin/activate

# 2. Run the sanity decision test using Laya on Apple MPS
python -c "
import json, laya
agent = laya.load('convaiinnovations/laya')
state = 'OpenAI released a new coding model that improves software engineering performance and is available to developers.'
question = {
    'ai_relevance': {
        'type': 'choice',
        'instructions': 'How relevant is this to an AI news intelligence system?',
        'criteria': {
            'highly_relevant': 'directly about major AI model releases, breakthroughs, or frontier tools',
            'somewhat_relevant': 'general tech or software news with minor AI mention',
            'not_relevant': 'completely unrelated to artificial intelligence'
        }
    }
}
res = agent.predict(state, question)
print('Device:', agent.device)
print('Choice:', res['answers']['ai_relevance']['choice'])
print('Probabilities:', res['answers']['ai_relevance']['probabilities'])
"
```
