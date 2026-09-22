"""
Read-only database query layer for the web presentation interface (Task 3.1 & Modifications).
Guarantees zero mutations to the SQLite database.
Formats summaries, importance star ratings, and key takeaways for human-friendly presentation.
"""

import re
import json
import math
import sqlite3
from typing import Optional, Any
from database import get_connection
from ranking import rank_articles, score_article, RankingWeights


def score_to_stars(score: Optional[float]) -> str:
    """
    Deterministically convert a decimal ranking score into a 5-star importance indicator.
    Mapping (Task 3.1 Modification Requirement 4):
        0.80–1.00 -> ★★★★★
        0.60–0.79 -> ★★★★☆
        0.40–0.59 -> ★★★☆☆
        0.20–0.39 -> ★★☆☆☆
        0.00–0.19 -> ★☆☆☆☆
    """
    if score is None:
        return ""
    try:
        s = float(score)
    except (ValueError, TypeError):
        return ""

    if s >= 0.80:
        return "★★★★★"
    elif s >= 0.60:
        return "★★★★☆"
    elif s >= 0.40:
        return "★★★☆☆"
    elif s >= 0.20:
        return "★★☆☆☆"
    else:
        return "★☆☆☆☆"


def score_to_star_count(score: Optional[float]) -> int:
    """Return integer star count (1-5) for visual styling."""
    if score is None:
        return 0
    try:
        s = float(score)
    except (ValueError, TypeError):
        return 0

    if s >= 0.80:
        return 5
    elif s >= 0.60:
        return 4
    elif s >= 0.40:
        return 3
    elif s >= 0.20:
        return 2
    else:
        return 1


def clean_human_summary(
    raw_summary: Optional[str],
    title: str = "",
    source: str = "",
    event_type: Optional[str] = None,
) -> str:
    """
    Format and clean summary text so it reads like a human-friendly news explanation.
    Strips raw scraper/LLM boilerplate (e.g. 'Analysis of...', 'Article URL...', etc.).
    Returns 'Analysis not available yet.' if summary is empty or unanalyzed.
    Does not invent summaries (Step 13).
    """
    if not raw_summary or not isinstance(raw_summary, str) or not raw_summary.strip():
        return "Analysis not available yet."

    text = raw_summary.strip()

    # 1. Strip standard boilerplate patterns
    patterns_to_remove = [
        r"^Analysis of ['\"].*?['\"]\.\s*",
        r"^Analysis of .*?:\s*",
        r"^According to the provided article,?\s*",
        r"^The model determined that\s*",
        r"^I analyzed .*? and found that\s*",
        r"Article URL:\s*https?://\S+",
        r"Comments URL:\s*https?://\S+",
        r"Points:\s*\d+\s*#\s*Comments:\s*\d+",
        r"Points:\s*\d+",
        r"#\s*Comments:\s*\d+",
    ]
    for pat in patterns_to_remove:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    # Clean up whitespace and orphaned punctuation
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lstrip(".- :")

    return text if text else "Analysis not available yet."


def clean_key_takeaways(key_points: Any, title: str = "", source: str = "") -> list[str]:
    """
    Ensure key takeaways contain actual informative points from article_analysis.
    Strips metadata boilerplate. Returns empty list if no takeaways available.
    Does not invent fake takeaways (Step 13).
    """
    raw_list = key_points if isinstance(key_points, list) else []
    cleaned = []

    for pt in raw_list:
        if not pt or not isinstance(pt, str):
            continue
        p = pt.strip()
        # Filter out boilerplate like "Article from ... titled '...'"
        if p.startswith("Article from ") and "titled " in p:
            continue
        if p.startswith("Primary topic identified as "):
            topic = p.replace("Primary topic identified as ", "").rstrip(".")
            cleaned.append(f"Focuses on advancements and technical discussions in {topic}.")
            continue
        cleaned.append(p)

    return cleaned


# ==============================================================================
# DATABASE QUERIES
# ==============================================================================

def get_summary_stats(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Retrieve headline statistics for the dashboard summary cards."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM articles")
        total_articles = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as analyzed FROM article_analysis WHERE status = 'completed'")
        analyzed_articles = cursor.fetchone()["analyzed"]

        cursor.execute("SELECT COUNT(DISTINCT source) as sources FROM articles")
        distinct_sources = cursor.fetchone()["sources"]

        cursor.execute("SELECT MAX(published_at) as last_updated FROM articles")
        row = cursor.fetchone()
        last_updated = row["last_updated"] if row and row["last_updated"] else None

        not_yet_analyzed = max(0, total_articles - analyzed_articles)

        return {
            "total_articles": total_articles,
            "analyzed_articles": analyzed_articles,
            "not_yet_analyzed": not_yet_analyzed,
            "distinct_sources": distinct_sources,
            "last_updated": last_updated,
        }
    finally:
        if should_close:
            conn.close()


def get_event_type_counts(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """Get distribution of event types among completed analyses for the sidebar."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    EVENT_CATEGORY_CONFIG = [
        {"key": "research_paper", "label": "research_paper", "color": "#3b82f6"},
        {"key": "model_release", "label": "model_release", "color": "#10b981"},
        {"key": "industry_news", "label": "industry_news", "color": "#f97316"},
        {"key": "company_news", "label": "company_news", "color": "#8b5cf6"},
        {"key": "tool_release", "label": "tool_release", "color": "#ef4444"},
        {"key": "regulation", "label": "regulation", "color": "#eab308"},
        {"key": "other", "label": "other", "color": "#94a3b8"},
    ]

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT event_type, COUNT(*) as cnt 
            FROM article_analysis 
            WHERE status = 'completed' AND event_type IS NOT NULL
            GROUP BY event_type
        """)
        db_counts = {row["event_type"]: row["cnt"] for row in cursor.fetchall()}

        results = []
        for cat in EVENT_CATEGORY_CONFIG:
            k = cat["key"]
            if k == "company_news":
                cnt = db_counts.get("company_news", 0) + db_counts.get("business_partnership", 0)
            elif k == "regulation":
                cnt = db_counts.get("regulation", 0) + db_counts.get("regulation_policy", 0)
            elif k == "other":
                cnt = (
                    db_counts.get("other", 0)
                    + db_counts.get("opinion_editorial", 0)
                    + db_counts.get("unknown", 0)
                )
            else:
                cnt = db_counts.get(k, 0)

            results.append({
                "key": k,
                "label": cat["label"],
                "color": cat["color"],
                "count": cnt,
            })

        return results
    finally:
        if should_close:
            conn.close()


def get_distinct_sources(conn: Optional[sqlite3.Connection] = None) -> list[str]:
    """Get sorted list of all active article sources."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source FROM articles ORDER BY source ASC")
        return [row["source"] for row in cursor.fetchall() if row["source"]]
    finally:
        if should_close:
            conn.close()


def get_ranked_articles_for_dashboard(
    limit: int = 10,
    source: Optional[str] = None,
    event_type: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """
    Retrieve top ranked articles via deterministic ranking engine.
    Applies human-friendly summary formatting and 5-star importance indicators.
    Completely removes internal confidence scores from user view (Requirement 3).
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        query = """
            SELECT 
                a.id as article_id,
                a.title,
                a.link,
                a.source,
                a.published_at,
                an.relevance_score,
                an.novelty,
                an.confidence,
                an.event_type,
                an.summary,
                an.status
            FROM articles a
            JOIN article_analysis an ON a.id = an.article_id
            WHERE an.status = 'completed'
        """
        params = []
        if source and source.strip() and source != "All Sources":
            query += " AND a.source = ?"
            params.append(source.strip())

        if event_type and event_type.strip() and event_type != "All Event Types":
            clean_et = event_type.strip()
            if clean_et == "company_news":
                query += " AND (an.event_type = 'company_news' OR an.event_type = 'business_partnership')"
            elif clean_et == "regulation":
                query += " AND (an.event_type = 'regulation' OR an.event_type = 'regulation_policy')"
            elif clean_et == "other":
                query += " AND (an.event_type = 'other' OR an.event_type = 'opinion_editorial' OR an.event_type = 'unknown')"
            else:
                query += " AND an.event_type = ?"
                params.append(clean_et)

        query += " ORDER BY a.id ASC"
        cursor.execute(query, params)
        records = [dict(r) for r in cursor.fetchall()]

        ranked_list = rank_articles(records, limit=limit)
        results = []
        for r in ranked_list:
            item = r.to_dict()
            # Convert decimal score to 5 stars (Requirement 4)
            item["importance_stars"] = score_to_stars(item["final_score"])
            item["importance_count"] = score_to_star_count(item["final_score"])
            # Format clean human explanation (Requirement 5)
            item["summary"] = clean_human_summary(
                item.get("summary"),
                title=item["title"],
                source=item["source"],
                event_type=item.get("event_type"),
            )
            # Remove internal ML confidence from user view (Requirement 3)
            if "component_scores" in item and "confidence" in item["component_scores"]:
                del item["component_scores"]["confidence"]
            results.append(item)

        return results
    finally:
        if should_close:
            conn.close()


def get_latest_articles(
    page: int = 1,
    per_page: int = 20,
    source: Optional[str] = None,
    event_type: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Retrieve newest articles by publication date, including unanalyzed articles.
    Returns pagination metadata and list of article dictionaries.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        where_clauses = ["1=1"]
        params = []

        if source and source.strip() and source != "All Sources":
            where_clauses.append("a.source = ?")
            params.append(source.strip())

        if event_type and event_type.strip() and event_type != "All Event Types":
            clean_et = event_type.strip()
            if clean_et == "company_news":
                where_clauses.append("(an.event_type = 'company_news' OR an.event_type = 'business_partnership')")
            elif clean_et == "regulation":
                where_clauses.append("(an.event_type = 'regulation' OR an.event_type = 'regulation_policy')")
            elif clean_et == "other":
                where_clauses.append("(an.event_type = 'other' OR an.event_type = 'opinion_editorial' OR an.event_type = 'unknown')")
            else:
                where_clauses.append("an.event_type = ?")
                params.append(clean_et)

        where_sql = " AND ".join(where_clauses)

        # Count total matching for pagination
        count_sql = f"""
            SELECT COUNT(*) as cnt
            FROM articles a
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE {where_sql}
        """
        cursor.execute(count_sql, params)
        total_count = cursor.fetchone()["cnt"]

        # Calculate pagination limits
        page = max(1, page)
        total_pages = max(1, math.ceil(total_count / per_page))
        if page > total_pages:
            page = total_pages
        offset = (page - 1) * per_page

        # Fetch articles
        data_sql = f"""
            SELECT 
                a.id,
                a.title,
                a.link,
                a.source,
                a.published_at,
                an.status,
                an.event_type,
                an.relevance_score
            FROM articles a
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE {where_sql}
            ORDER BY 
                CASE WHEN a.published_at IS NULL THEN 1 ELSE 0 END,
                a.published_at DESC, 
                a.id DESC
            LIMIT ? OFFSET ?
        """
        fetch_params = list(params) + [per_page, offset]
        cursor.execute(data_sql, fetch_params)
        articles = [dict(r) for r in cursor.fetchall()]

        return {
            "articles": articles,
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
        }
    finally:
        if should_close:
            conn.close()


def get_article_detail(
    article_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[dict]:
    """
    Retrieve an individual article record joined with article_analysis if available.
    Formats human-friendly explanations, key takeaways, and 5-star importance.
    Strictly removes internal confidence and raw LLM metadata from visible fields.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        query = """
            SELECT 
                a.id,
                a.title,
                a.link,
                a.source,
                a.summary as raw_summary,
                a.published_at,
                a.created_at as ingested_at,
                an.relevance_score,
                an.summary as analyzed_summary,
                an.key_points,
                an.topics,
                an.technologies,
                an.companies,
                an.people,
                an.models,
                an.event_type,
                an.novelty,
                an.confidence,
                an.entities,
                an.status as analysis_status,
                an.provider as llm_provider,
                an.model as llm_model,
                an.error_type,
                an.error_message,
                an.updated_at as analyzed_at
            FROM articles a
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE a.id = ?
        """
        cursor.execute(query, (article_id,))
        row = cursor.fetchone()
        if not row:
            return None

        data = dict(row)

        # Deserialize JSON fields
        json_fields = ["key_points", "topics", "technologies", "companies", "people", "models", "entities"]
        for f in json_fields:
            raw = data.get(f)
            if raw and isinstance(raw, str):
                try:
                    data[f] = json.loads(raw)
                except json.JSONDecodeError:
                    data[f] = []
            elif not raw:
                data[f] = [] if f != "entities" else {}

        # Compute dynamic ranking score if status is completed
        if data.get("analysis_status") == "completed" and data.get("relevance_score") is not None:
            scoring_payload = {
                "article_id": data["id"],
                "title": data["title"],
                "source": data["source"],
                "published_at": data["published_at"],
                "relevance_score": data["relevance_score"],
                "novelty": data.get("novelty"),
                "confidence": data.get("confidence", 1.0),
                "event_type": data.get("event_type"),
                "status": "completed",
            }
            ranked = score_article(scoring_payload)
            if ranked:
                data["importance_stars"] = score_to_stars(ranked.final_score)
                data["importance_count"] = score_to_star_count(ranked.final_score)
                data["final_score"] = ranked.final_score
            else:
                data["importance_stars"] = ""
                data["importance_count"] = 0
                data["final_score"] = None
        else:
            data["importance_stars"] = ""
            data["importance_count"] = 0
            data["final_score"] = None

        # Clean human-friendly summary (Step 13)
        if data.get("analysis_status") == "completed" and data.get("analyzed_summary"):
            data["human_summary"] = clean_human_summary(
                data["analyzed_summary"],
                title=data["title"],
                source=data["source"],
                event_type=data.get("event_type"),
            )
        else:
            data["human_summary"] = "Analysis not available yet."

        # Clean key takeaways (Requirement 7)
        data["clean_key_takeaways"] = clean_key_takeaways(
            data.get("key_points"),
            title=data["title"],
            source=data["source"],
        )

        # Remove internal confidence from user view (Requirement 3)
        if "confidence" in data:
            del data["confidence"]

        return data
    finally:
        if should_close:
            conn.close()
