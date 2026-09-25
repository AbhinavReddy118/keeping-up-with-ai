"""
Article Content Enrichment Layer (Task 1.2).

Detects articles with missing or insufficient RSS content, fetches their original
webpages with robust HTTP error handling, extracts the readable article body text
while stripping boilerplate/navigation/scripts, and idempotently stores the extracted
content in the database without modifying original RSS metadata.
"""

import json
import logging
import re
import sqlite3
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from database import get_connection
from storage import get_article_by_id, update_article_enrichment

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "AI-News-Digest-Enricher/1.0 "
    "(Mozilla/5.0 compatible; AI Intelligence Monitor; +https://github.com/AbhinavReddy118/Ai_news_digest)"
)
DEFAULT_TIMEOUT = 15.0

# Tags to strip entirely before text extraction
NON_CONTENT_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "svg",
    "dialog",
    "iframe",
]

# Class and ID patterns that indicate non-article noise (cookies, ads, comments, sidebars)
UNWANTED_CONTAINER_REGEX = re.compile(
    r"(comment|discussion|sidebar|cookie|banner|newsletter|modal|advertisement|social-share|share-button)",
    re.IGNORECASE,
)


def is_usable_content(
    text: Optional[str],
    min_chars: int = 100,
    min_words: int = 15,
) -> bool:
    """
    Determine whether text constitutes substantive, usable article content.

    Rules:
    - Empty, whitespace-only, or None is insufficient.
    - RSS metadata JSON string alone is insufficient.
    - Very short strings (< min_chars or < min_words) are insufficient.
    - Non-string types are insufficient.
    """
    if not text or not isinstance(text, str):
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # Detect JSON-serialized RSS metadata (e.g. raw_content payloads)
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return False
        except (json.JSONDecodeError, ValueError):
            pass

    if len(stripped) < min_chars:
        return False

    if len(stripped.split()) < min_words:
        return False

    return True


def extract_article_text_from_html(html: str) -> Optional[str]:
    """
    Extract the main readable article text from raw HTML.

    Prioritizes article/main content over navigation, headers, footers,
    cookie notices, scripts, styles, and unrelated page elements.
    """
    if not html or not isinstance(html, str) or not html.strip():
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.warning(f"Failed to parse HTML with BeautifulSoup: {exc}")
        return None

    # Decompose script, style, navigation, headers, footers, etc.
    for tag_name in NON_CONTENT_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Decompose known noise containers (cookie notices, comment sections, sidebars)
    for el in soup.find_all(attrs={"class": UNWANTED_CONTAINER_REGEX}):
        el.decompose()
    for el in soup.find_all(attrs={"id": UNWANTED_CONTAINER_REGEX}):
        el.decompose()

    # Resolve primary content container in order of specificity
    container = None

    # 1. Look for targeted content containers (Tailwind/prose/blog classes)
    content_classes = [
        "blog-content",
        "article-content",
        "post-content",
        "entry-content",
        "article-body",
        "story-body",
        "prose",
    ]
    for cls in content_classes:
        candidate = soup.find(class_=re.compile(rf"\b{cls}\b", re.IGNORECASE))
        if candidate:
            # Strip non-prose helper divs inside blog content (e.g. HF comment widgets)
            for np in candidate.find_all(class_="not-prose"):
                np.decompose()
            text_candidate = candidate.get_text(separator="\n\n", strip=True)
            if is_usable_content(text_candidate):
                container = candidate
                break

    # 2. Look for semantic <article>
    if not container:
        articles = soup.find_all("article")
        for art in articles:
            text_candidate = art.get_text(separator="\n\n", strip=True)
            if is_usable_content(text_candidate):
                container = art
                break

    # 3. Look for <main> or [role="main"]
    if not container:
        main = soup.find("main") or soup.find(attrs={"role": "main"})
        if main:
            # Strip any lingering not-prose or sidebar-like elements inside main
            for np in main.find_all(class_="not-prose"):
                np.decompose()
            text_candidate = main.get_text(separator="\n\n", strip=True)
            if is_usable_content(text_candidate):
                container = main

    # 4. Fallback to <body>
    if not container:
        container = soup.body

    if not container:
        return None

    raw_text = container.get_text(separator="\n\n", strip=True)

    # Normalize whitespace: collapse 3+ newlines to 2, collapse spaces/tabs
    cleaned_text = re.sub(r"[ \t]+", " ", raw_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    if not is_usable_content(cleaned_text):
        return None

    return cleaned_text


def fetch_webpage_content(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch the webpage at URL and extract readable article text.

    Returns:
        (extracted_text, error_message): Extracted text if successful, or error details.
        Never raises network or parsing exceptions.
    """
    if not url or not isinstance(url, str) or not url.strip().startswith(("http://", "https://")):
        return None, f"Invalid URL: {url}"

    headers = {"User-Agent": user_agent}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get(url)

        if response.status_code >= 400:
            return None, f"HTTP error {response.status_code}: {response.reason_phrase}"

        extracted_text = extract_article_text_from_html(response.text)
        if not extracted_text:
            return None, "No substantive article content found on webpage"

        return extracted_text, None

    except httpx.TimeoutException as exc:
        logger.warning(f"Timeout fetching article content from {url} ({timeout}s): {exc}")
        return None, f"HTTP Timeout after {timeout}s: {exc}"
    except httpx.HTTPStatusError as exc:
        logger.warning(f"HTTP status error fetching {url}: {exc}")
        return None, f"HTTP Error {exc.response.status_code}: {exc}"
    except httpx.RequestError as exc:
        logger.warning(f"Network request error fetching {url}: {exc}")
        return None, f"Network Request Error: {exc}"
    except Exception as exc:
        logger.warning(f"Unexpected error enriching article from {url}: {exc}")
        return None, f"Extraction Error: {exc}"


def enrich_article(
    article_id: int,
    conn: Optional[sqlite3.Connection] = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    """
    Enrich an article record with readable webpage content if RSS content is insufficient.

    Idempotent:
    - If article already has usable content or is marked 'enriched', skips fetch.
    - If RSS summary is already usable and sufficient, marks 'not_needed' and skips fetch.
    - If fetch fails, updates enrichment_status to 'failed' with error details.
    - Preserves existing raw RSS metadata in raw_content.

    Args:
        article_id: Primary key ID of the article.
        conn: Optional active SQLite database connection.
        force: If True, force re-enrichment even if already enriched.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Updated article dictionary record.

    Raises:
        ValueError: If article_id is not found in database.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        article = get_article_by_id(article_id, conn=conn)
        if not article:
            raise ValueError(f"Article ID {article_id} not found in database.")

        # 1. Check idempotency: already enriched with usable content
        existing_content = article.get("content")
        existing_status = article.get("enrichment_status")

        if not force and existing_status == "enriched" and is_usable_content(existing_content):
            logger.debug(f"Article {article_id} already enriched with usable content; skipping.")
            return article

        # 2. Check if original RSS summary is already usable and sufficient
        existing_summary = article.get("summary")
        if not force and is_usable_content(existing_summary):
            if existing_status != "not_needed":
                update_article_enrichment(
                    article_id,
                    content=None,
                    status="not_needed",
                    error=None,
                    conn=conn,
                )
                article["enrichment_status"] = "not_needed"
            logger.debug(f"Article {article_id} has sufficient RSS summary; enrichment not needed.")
            return article

        # 3. Content is missing/insufficient -> Fetch article link
        link = article.get("link")
        if not link:
            update_article_enrichment(
                article_id,
                content=None,
                status="failed",
                error="Article has no link URL to fetch",
                conn=conn,
            )
            article["enrichment_status"] = "failed"
            article["enrichment_error"] = "Article has no link URL to fetch"
            return article

        logger.info(f"Enriching article {article_id} from {link}...")
        extracted_text, error_message = fetch_webpage_content(
            link, timeout=timeout, user_agent=user_agent
        )

        if extracted_text and is_usable_content(extracted_text):
            update_article_enrichment(
                article_id,
                content=extracted_text,
                status="enriched",
                error=None,
                conn=conn,
            )
            article["content"] = extracted_text
            article["enrichment_status"] = "enriched"
            article["enrichment_error"] = None
            logger.info(
                f"Successfully enriched article {article_id}: {len(extracted_text)} chars, "
                f"{len(extracted_text.split())} words."
            )
        else:
            err = error_message or "Failed to extract usable article content"
            update_article_enrichment(
                article_id,
                content=None,
                status="failed",
                error=err,
                conn=conn,
            )
            article["content"] = None
            article["enrichment_status"] = "failed"
            article["enrichment_error"] = err
            logger.warning(f"Enrichment failed for article {article_id}: {err}")

        return article

    finally:
        if should_close:
            conn.close()


def enrich_articles_batch(
    article_ids: list[int],
    conn: Optional[sqlite3.Connection] = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """
    Enrich a list of articles by ID.

    Returns:
        Summary dict: {"total": int, "enriched": int, "skipped": int, "failed": int}
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    summary = {"total": len(article_ids), "enriched": 0, "skipped": 0, "failed": 0}

    try:
        for aid in article_ids:
            try:
                res = enrich_article(aid, conn=conn, force=force, timeout=timeout)
                st = res.get("enrichment_status")
                if st == "enriched":
                    summary["enriched"] += 1
                elif st in ("not_needed", "skipped"):
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1
            except Exception as exc:
                logger.error(f"Error enriching article {aid}: {exc}")
                summary["failed"] += 1

        return summary
    finally:
        if should_close:
            conn.close()
