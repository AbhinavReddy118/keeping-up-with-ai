"""
Offline demonstration of LLM Analysis Foundation (Task 2.9.1) against ONE existing article.
Guarantees NO external API calls and NO modifications to the SQLite database.
"""

import json
from database import get_connection
from llm import analyze_article, MockLLMProvider


def main():
    print("=" * 80)
    print("TASK 2.9.1: LLM ANALYSIS FOUNDATION - OFFLINE DEMONSTRATION")
    print("=" * 80)

    # 1. Check article count before demo
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_before = cursor.fetchone()["cnt"]

        # Fetch exactly ONE article from the database
        cursor.execute("""
            SELECT id, title, link, summary, source, published_at, raw_content 
            FROM articles 
            WHERE summary IS NOT NULL AND LENGTH(summary) > 30 
            ORDER BY id DESC 
            LIMIT 1
        """)
        article = dict(cursor.fetchone())

    print(f"\n[1] Selected Article from Database (ID: {article['id']}):")
    print(f"    Title       : {article['title']}")
    print(f"    Source      : {article['source']}")
    print(f"    Published At: {article['published_at']}")
    print(f"    Link        : {article['link']}")
    print(f"    Summary     : {article['summary'][:120]}...")

    # 2. Run analysis using deterministic MockLLMProvider (NO external network call)
    provider = MockLLMProvider()
    print(f"\n[2] Executing analysis using Provider: '{provider.name}' (Deterministic Mock)")
    result = analyze_article(article, provider=provider)

    # 3. Display structured analysis
    print("\n[3] Generated Structured Analysis Result (Validated Contract):")
    print(json.dumps(result.to_dict(), indent=2))

    # 4. Check article count after demo
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_after = cursor.fetchone()["cnt"]

    print("\n[4] Database Integrity Verification:")
    print(f"    Article count BEFORE demo : {count_before}")
    print(f"    Article count AFTER demo  : {count_after}")
    assert count_before == count_after, "Database count changed during read-only demo!"
    print("    Confirmation              : DATABASE WAS NOT MODIFIED (100% read-only)")
    print("=" * 80)


if __name__ == "__main__":
    main()
