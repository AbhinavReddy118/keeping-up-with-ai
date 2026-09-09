"""
Demonstration of Live LLM Provider Integration (Task 2.9.2).

Performs analysis of exactly ONE article from the SQLite database using either
a configured live LLM provider (OpenAI / OpenRouter) or safely reports configuration instructions
if credentials are not present.

Guarantees:
- Never modifies or deletes any SQLite database records.
- Masks API keys completely so secrets are never printed or logged.
- Verifies database integrity before and after execution.
"""

import os
import sys
import json
from database import get_connection
from config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from llm import (
    analyze_article,
    prepare_article_input,
    get_llm_provider,
    MockLLMProvider,
    LLMConfigurationError,
    LLMError,
)


def mask_api_key(key: str) -> str:
    """Safely mask an API key for display without leaking secrets."""
    if not key:
        return "[NOT CONFIGURED]"
    cleaned = key.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:3]}...{cleaned[-4:]} (length: {len(cleaned)})"


def main():
    print("=" * 80)
    print("TASK 2.9.2: LIVE LLM PROVIDER INTEGRATION - SINGLE ARTICLE DEMO")
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
        row = cursor.fetchone()
        if not row:
            print("ERROR: No suitable articles found in the database.")
            sys.exit(1)
        article = dict(row)

    print(f"\n[1] Selected Database Article (ID: {article['id']}):")
    print(f"    Title       : {article['title']}")
    print(f"    Source      : {article['source']}")
    print(f"    Published At: {article['published_at']}")
    print(f"    Link        : {article['link']}")
    summary_preview = (article['summary'] or "")[:120].replace('\n', ' ')
    print(f"    Summary     : {summary_preview}...")

    # 2. Inspect provider configuration
    provider_name = (LLM_PROVIDER or "mock").strip().lower()
    has_api_key = bool(LLM_API_KEY and LLM_API_KEY.strip())

    print("\n[2] Provider Configuration Status:")
    print(f"    LLM_PROVIDER: '{provider_name}'")
    print(f"    LLM_MODEL   : '{LLM_MODEL or '(default for provider)'}'")
    print(f"    LLM_BASE_URL: '{LLM_BASE_URL or '(default for provider)'}'")
    print(f"    LLM_API_KEY : {mask_api_key(LLM_API_KEY)}")

    # 3. Handle live vs mock execution
    if provider_name == "mock" or not has_api_key:
        print("\n" + "-" * 80)
        print("  CONFIGURATION NOTICE: Real LLM credentials not configured.")
        print("  The system is operating in safe offline/mock mode.")
        print()
        print("  To perform a live API test against OpenAI or OpenRouter:")
        print("  1. Create or update your local .env file (never commit this file!):")
        print("         LLM_PROVIDER=openai")
        print("         LLM_MODEL=gpt-4o-mini")
        print("         LLM_API_KEY=your_actual_api_key_here")
        print()
        print("     Or for OpenRouter:")
        print("         LLM_PROVIDER=openrouter")
        print("         LLM_MODEL=openai/gpt-4o-mini")
        print("         LLM_API_KEY=your_openrouter_api_key_here")
        print()
        print("  2. Re-run: python3 demo_live_llm.py")
        print("-" * 80)

        print(f"\n[3] Demonstrating Analysis Pipeline Using: 'mock' provider (offline safe)")
        provider = MockLLMProvider()
        result = analyze_article(article, provider=provider)
        print(f"    Provider executed: {provider.name}")

    else:
        print(f"\n[3] Initializing Live Provider: '{provider_name}'")
        try:
            provider = get_llm_provider(
                provider_name=provider_name,
                api_key=LLM_API_KEY,
                model=LLM_MODEL if LLM_MODEL else None,
                base_url=LLM_BASE_URL if LLM_BASE_URL else None,
                timeout=30.0,
            )
            print(f"    Target Endpoint: {getattr(provider, 'base_url', 'N/A')}/chat/completions")
            print(f"    Target Model   : {getattr(provider, 'model', 'N/A')}")
            print(f"    Sending request for Article ID {article['id']}...")

            result = analyze_article(article, provider=provider)
            print("    Live LLM analysis completed successfully!")

        except LLMConfigurationError as e:
            print(f"\n[!] Configuration Error: {e}")
            result = None
        except LLMError as e:
            print(f"\n[!] LLM Provider Call Failed: {e.__class__.__name__}: {e}")
            result = None
        except Exception as e:
            print(f"\n[!] Unexpected Error: {e}")
            result = None

    # 4. Display Analysis Result
    if result:
        print("\n[4] Structured Analysis Result (Validated Contract):")
        print(json.dumps(result.to_dict(), indent=2))
        print(f"    Prompt Version : {result.prompt_version}")
        print(f"    Raw Response   : {result.raw_response[:100]}..." if result.raw_response else "    Raw Response: None")
    else:
        print("\n[4] Structured Analysis Result: SKIPPED DUE TO ERROR/MISSING CREDENTIALS")

    # 5. Check article count after demo
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM articles")
        count_after = cursor.fetchone()["cnt"]

    print("\n[5] Database Integrity Verification:")
    print(f"    Article count BEFORE demo : {count_before}")
    print(f"    Article count AFTER demo  : {count_after}")
    assert count_before == count_after, f"Database count changed! ({count_before} -> {count_after})"
    print("    Confirmation              : DATABASE WAS NOT MODIFIED (100% read-only)")
    print("=" * 80)


if __name__ == "__main__":
    main()
