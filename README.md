# Keeping Up With AI

> A personal AI intelligence pipeline that continuously tracks new models, research, tools, companies, and major developments across the AI ecosystem.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

## The Problem

AI changes faster than it is practical to follow manually.

New model releases, research papers, developer tools, open-source projects, company announcements, and industry developments appear across hundreds of sources.

I built this project to answer one question:

### **"What changed in AI today, and what actually matters?"**

---

## What It Does

The system continuously:

**Collects** → **Cleans** → **Deduplicates** → **Analyzes** → **Ranks** → **Delivers**

| Stage | What happens |
|---|---|
| 📡 Ingestion | Collects updates from configured AI sources |
| 🧹 Normalization | Converts different source formats into a common structure |
| 🔎 Deduplication | Detects the same development appearing across multiple sources |
| 🤖 AI Analysis | Classifies and summarizes important developments |
| 📊 Ranking | Determines which developments deserve attention |
| 🌐 Web | Provides a searchable view of collected intelligence |
| 📱 Telegram | Sends important updates directly to Telegram |

---

## Architecture

```text
                  AI ECOSYSTEM
                       │
          ┌────────────┴────────────┐
          │                         │
       RSS / APIs              Other Sources
          │                         │
          └────────────┬────────────┘
                       ↓
                 DATA INGESTION
                       ↓
                 NORMALIZATION
                       ↓
                 DEDUPLICATION
                       ↓
                SQLite Database
                       ↓
                 AI ANALYSIS
                       ↓
              IMPORTANCE RANKING
                       │
              ┌────────┴────────┐
              ↓                 ↓
        WEB INTERFACE        TELEGRAM
