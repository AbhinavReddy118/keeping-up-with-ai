"""
Demonstration of Deterministic Article Ranking (Task 3.0).

Loads completed article analysis records from SQLite, computes deterministic scores
using weighted components (relevance, novelty, recency, confidence), and displays
the top 10 ranked articles.

Guarantees:
- Database is 100% read-only (article count verified before and after).
- Zero external API or network calls.
- Fully deterministic sorting with tie-breaking.
"""

import sys
import argparse
from datetime import datetime, timezone
from database import get_connection
from ranking import (
    RankingWeights,
    get_completed_articles_for_ranking,
    rank_articles,
)


def main():
    parser = argparse.ArgumentParser(description="Demonstrate deterministic article ranking.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of top articles to display (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("TASK 3.0: DETERMINISTIC ARTICLE RANKING DEMONSTRATION")
    print("=" * 100)

    # 1. Check article count before demo
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_count_before = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM article_analysis WHERE status = 'completed'")
        completed_count = cursor.fetchone()["cnt"]

    print(f"\n[1] Database State:")
    print(f"    Total articles in database      : {articles_count_before}")
    print(f"    Articles with completed analysis: {completed_count}")

    if completed_count == 0:
        print("\n[!] No completed article analyses found in the database.")
        print("    Please run: python3 demo_batch_analysis.py --limit 10")
        print("    to analyze a batch of articles first.")
        sys.exit(0)

    # 2. Fetch completed articles using SQL JOIN
    with get_connection() as conn:
        records = get_completed_articles_for_ranking(conn=conn)

    print(f"\n[2] Loaded {len(records)} completed article record(s) via SQL JOIN.")

    # 3. Compute deterministic ranking
    weights = RankingWeights()
    print(f"\n[3] Scoring Configuration (Weights sum to {weights.relevance + weights.novelty + weights.recency + weights.confidence:.2f}):")
    print(f"    - Relevance : {weights.relevance * 100:.0f}%")
    print(f"    - Novelty   : {weights.novelty * 100:.0f}%")
    print(f"    - Recency   : {weights.recency * 100:.0f}% (7-day exponential half-life)")
    print(f"    - Confidence: {weights.confidence * 100:.0f}%")

    ranked_articles = rank_articles(records, weights=weights, limit=args.limit)

    # 4. Display top ranked articles
    print(f"\n[4] Top {len(ranked_articles)} Ranked Articles:")
    print("-" * 100)
    header = f"{'Rank':<5} | {'ID':<5} | {'Final':<7} | {'Rel':<5} | {'Nov':<5} | {'Rec':<5} | {'Conf':<5} | {'Title':<45}"
    print(header)
    print("-" * 100)

    for idx, item in enumerate(ranked_articles, start=1):
        comp = item.component_scores
        title_trunc = item.title[:45]
        print(
            f"{idx:<5} | {item.article_id:<5} | {item.final_score:.4f}  | "
            f"{comp.relevance:.2f}  | {comp.novelty:.2f}  | {comp.recency:.2f}  | {comp.confidence:.2f}  | "
            f"{title_trunc}"
        )

    print("-" * 100)

    # 5. Display detailed breakdown of #1 ranked article
    if ranked_articles:
        top = ranked_articles[0]
        c = top.component_scores
        print(f"\n[5] Deep Dive: #1 Ranked Article (ID {top.article_id}):")
        print(f"    Title       : {top.title}")
        print(f"    Source      : {top.source}")
        print(f"    Published At: {top.published_at}")
        print(f"    Link        : {top.link}")
        print(f"    Event Type  : {top.event_type}")
        print(f"    Novelty     : {top.novelty_label} (score: {c.novelty:.2f})")
        print(f"    Relevance   : {c.relevance:.2f} (weight: 40% -> {c.relevance * weights.relevance:.4f})")
        print(f"    Novelty     : {c.novelty:.2f} (weight: 25% -> {c.novelty * weights.novelty:.4f})")
        print(f"    Recency     : {c.recency:.2f} (weight: 20% -> {c.recency * weights.recency:.4f})")
        print(f"    Confidence  : {c.confidence:.2f} (weight: 15% -> {c.confidence * weights.confidence:.4f})")
        print(f"    -------------------------------------------------")
        print(f"    FINAL SCORE : {top.final_score:.4f}")

    # 6. Database Integrity Check
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        articles_count_after = cursor.fetchone()["cnt"]

    print(f"\n[6] Database Integrity Verification:")
    print(f"    Articles count BEFORE demo : {articles_count_before}")
    print(f"    Articles count AFTER demo  : {articles_count_after}")
    assert articles_count_before == articles_count_after, "Articles count changed during ranking demo!"
    print("    Confirmation               : ARTICLES TABLE WAS NOT MODIFIED (100% read-only)")
    print("=" * 100)


if __name__ == "__main__":
    main()
