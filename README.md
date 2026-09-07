Keeping Up With AI

An automated system that continuously monitors the AI ecosystem and keeps me updated on important developments in AI.

It collects information from multiple sources, filters duplicate stories, analyzes and ranks important developments, and delivers updates through a web interface and Telegram.

Why?

AI is changing extremely quickly. New models, research, tools, companies, products, and techniques appear every day.

Following everything manually across dozens of sources is time-consuming, so I built a system to do the monitoring for me.

How It Works

AI Sources
    ↓
Data Ingestion
    ↓
Normalization
    ↓
Deduplication
    ↓
AI Analysis
    ↓
Importance Ranking
    ↓
Web Interface + Telegram

The pipeline continuously collects new information, converts it into a common format, removes duplicate developments, uses an LLM to analyze the content, and surfaces the information that matters most.

Tech Stack

* Python
* SQLite
* RSS / APIs
* Pydantic
* LLM APIs
* Embeddings
* Telegram Bot API
* Web interface
* pytest
* Git / GitHub

Current Status

* Project setup
* SQLite database
* RSS/Atom ingestion
* Source configuration
* Data normalization
* Database ingestion
* Deduplication
* LLM analysis
* Importance ranking
* Telegram notifications
* Web interface
* Continuous automation

Running Locally

git clone <repository-url>
cd ai_news_digest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py

Run tests:

pytest

Project Goal

The goal is to build a practical end-to-end data pipeline that turns a constantly changing stream of AI information into a useful personal intelligence feed.

Built by Abhinav Reddy.
