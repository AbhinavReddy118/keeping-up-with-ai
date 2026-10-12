# Keeping Up With AI

> A personal AI intelligence pipeline that continuously tracks new models, research, tools, companies, and major developments across the AI ecosystem.

AI changes faster than it is practical to follow manually.

New model releases, research papers, developer tools, open-source projects, company announcements, and industry developments appear across many different sources.

Keeping Up With AI is a local, continuously running intelligence pipeline built to answer one question:

> **What changed in AI, and what actually matters?**

The system continuously collects, processes, analyzes, ranks, and optionally delivers important AI developments without requiring manual execution.

---

## What It Does

The pipeline continuously runs:

**Ingest → Normalize → Deduplicate → Enrich → Analyze → Rank → Deliver**

| Stage | What it does |
|---|---|
| Ingestion | Polls configured AI news and research sources |
| Normalization | Converts RSS/Atom data into a consistent article structure |
| Deduplication | Prevents the same development from being stored repeatedly |
| Web Enrichment | Extracts additional content when feed metadata is insufficient |
| AI Analysis | Uses Qwen 3 4B through Ollama to analyze articles |
| Ranking | Calculates a deterministic importance score |
| Storage | Tracks every article through a SQLite-backed pipeline state machine |
| Dashboard | Provides a local searchable view of collected intelligence |
| Telegram | Optionally delivers important developments |

The system is designed to run continuously in the background on macOS.

---

## Current Architecture

```text
                    AI ECOSYSTEM
                         │
             ┌───────────┴───────────┐
             │                       │
          RSS / Atom            Web Content
             │                       │
             └───────────┬───────────┘
                         ↓
                    INGESTION
                         ↓
                  NORMALIZATION
                         ↓
                   DEDUPLICATION
                         ↓
                  WEB ENRICHMENT
                         ↓
                    SQLite DB
                         ↓
                  PIPELINE STATE
                         ↓
                  QWEN 3 4B / OLLAMA
                         ↓
                  STRUCTURED ANALYSIS
                         ↓
                DETERMINISTIC RANKING
                         │
                 ┌───────┴────────┐
                 ↓                ↓
             DASHBOARD         TELEGRAM
