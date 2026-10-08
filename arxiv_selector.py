"""
ArXiv Analysis Selection Layer.

Deterministically scores and prioritizes newly ingested arXiv papers so that
only a manageable, high-value subset enters the expensive Qwen analysis queue
each UTC day, while 100% of papers remain stored in SQLite.
"""

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from config import ARXIV_DAILY_ANALYSIS_BUDGET
from database import get_connection

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. SIGNAL CATEGORIES & VOCABULARY SPECIFICATION
# ------------------------------------------------------------------------------

# Generic terms that must not dominate the score by themselves
GENERIC_TERMS: Set[str] = {
    "agent",
    "reinforcement learning",
    "planning",
    "inference",
    "efficient",
    "robotics",
    "safety",
    "multimodal",
    "attention",
    "language model",
}

# Signal categories and compiled patterns: (term_key, regex_pattern, is_case_sensitive)
CATEGORIES_DEF: Dict[str, List[Tuple[str, str, bool]]] = {
    "foundation_model": [
        ("large language model", r"\blarge language models?\b", False),
        ("LLM", r"\bLLMs?\b", True),
        ("vision-language", r"\bvision[- ]languages?\b", False),
        ("VLM", r"\bVLMs?\b", True),
        ("foundation model", r"\bfoundation models?\b", False),
        ("model architecture", r"\bmodel architectures?\b", False),
        ("transformer", r"\btransformers?\b", False),
        ("attention", r"\battention\b", False),
        ("MoE", r"\b(?:MoE|mixture[- ]of[- ]experts)\b", True),
        ("RAG", r"\b(?:RAG|retrieval[- ]augmented(?: generation)?)\b", True),
        ("language model", r"\blanguage models?\b", False),
        ("multimodal", r"\bmultimodal\b", False),
    ],
    "reasoning_agents": [
        ("agentic", r"\bagentic\b", False),
        ("tool use", r"\b(?:tool[- ]use|using tools|tool utilization)\b", False),
        ("RLHF", r"\bRLHF\b", True),
        ("reasoning", r"\breasoning\b", False),
        ("reinforcement learning", r"\breinforcement learning\b", False),
        ("agent", r"\bagents?\b", False),
        ("planning", r"\bplanning\b", False),
    ],
    "systems_efficiency": [
        ("speculative decoding", r"\bspeculative decoding\b", False),
        ("training efficiency", r"\btraining efficiency\b", False),
        ("quantization", r"\bquantization\b", False),
        ("distillation", r"\bdistillation\b", False),
        ("acceleration", r"\bacceleration\b", False),
        ("inference", r"\binference\b", False),
        ("efficient", r"\befficient(?:cy)?\b", False),
    ],
    "safety_security": [
        ("alignment", r"\balignment\b", False),
        ("jailbreak", r"\bjailbreaks?\b", False),
        ("poisoning", r"\bpoisoning\b", False),
        ("backdoor", r"\bbackdoors?\b", False),
        ("hallucination", r"\bhallucinations?\b", False),
        ("adversarial", r"\badversarial\b", False),
        ("safety", r"\bsafety\b", False),
    ],
    "generative_robotics": [
        ("diffusion", r"\bdiffusion\b", False),
        ("image generation", r"\bimage generation\b", False),
        ("video generation", r"\bvideo generation\b", False),
        ("embodied", r"\bembodied\b", False),
        ("robotics", r"\brobotics\b", False),
    ],
}

# Announce type ranking prior (Requirement 5)
ANNOUNCE_BONUS: Dict[str, float] = {
    "new": 1.0,
    "cross": 0.5,
    "replace": 0.2,
    "replace-cross": 0.0,
}

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def is_arxiv_article(source: str, link: str = "") -> bool:
    """Return True if the source name or link indicates an arXiv article."""
    s_low = (source or "").strip().lower()
    l_low = (link or "").strip().lower()
    return "arxiv" in s_low or "arxiv.org" in l_low


def extract_announce_type(summary: Optional[str]) -> str:
    """Extract announce_type string (new, cross, replace, replace-cross) from summary."""
    if not summary:
        return "new"
    match = re.search(r"Announce Type:\s*([^\s\n]+)", summary)
    if match:
        return match.group(1).lower().strip()
    return "new"


def extract_abstract(summary: Optional[str]) -> str:
    """Extract the substantive abstract body from summary, removing arXiv header prefix."""
    if not summary:
        return ""
    match = re.search(r"Abstract:\s*(.*)", summary, re.DOTALL)
    if match:
        return match.group(1).strip()
    return summary.strip()


def parse_published_at_to_epoch(published_at: Optional[str]) -> float:
    """Convert an ISO-8601 or RFC-like published_at string to UTC epoch timestamp."""
    if not published_at or not isinstance(published_at, str):
        return 0.0
    clean = published_at.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
        return dt.timestamp()
    except (ValueError, TypeError):
        pass
    return 0.0

# ------------------------------------------------------------------------------
# 3. DETERMINISTIC PRIORITY SCORING
# ------------------------------------------------------------------------------

def score_arxiv_article(
    title: str,
    summary: Optional[str],
    announce_type: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a deterministic, explainable priority score for an arXiv article.

    Scoring Formulation:
      1. Specific keyword in Title: +3.0 pts
      2. Specific keyword in Abstract: +1.0 pt
      3. Generic keyword in Title: +1.5 pts
      4. Generic keyword in Abstract: +0.5 pts
      5. Abstract score cap per category: max 2.5 pts (prevents keyword stuffing)
      6. Multi-category breadth synergy bonus:
         - >= 2 distinct categories: +2.0 pts
         - >= 3 distinct categories: +4.0 pts
      7. Generic-only safeguard:
         - If specific_term_count == 0, base score is capped at 2.5 pts
           (prevents classical engineering/traffic RL papers from scoring high)
      8. Announce type prior:
         - new: +1.0, cross: +0.5, replace: +0.2, replace-cross: 0.0

    Returns:
        tuple of (final_score, details_dict)
    """
    clean_title = title or ""
    clean_summary = summary or ""
    abstract = extract_abstract(clean_summary)
    ann_type = (announce_type or extract_announce_type(clean_summary)).lower().strip()

    title_matches: Dict[str, List[str]] = {}
    abstract_matches: Dict[str, List[str]] = {}
    specific_term_count = 0

    for cat_name, term_defs in CATEGORIES_DEF.items():
        matched_in_title: Set[str] = set()
        for term_key, pat_str, case_sens in term_defs:
            flags = 0 if case_sens else re.IGNORECASE
            pat = re.compile(pat_str, flags)
            if pat.search(clean_title):
                # Avoid sub-phrase double counting (e.g. LLM + language model)
                if term_key == "language model" and ("large language model" in matched_in_title or "LLM" in matched_in_title):
                    continue
                matched_in_title.add(term_key)
                if term_key not in GENERIC_TERMS:
                    specific_term_count += 1
        if matched_in_title:
            title_matches[cat_name] = sorted(matched_in_title)

        matched_in_abs: Set[str] = set()
        for term_key, pat_str, case_sens in term_defs:
            flags = 0 if case_sens else re.IGNORECASE
            pat = re.compile(pat_str, flags)
            if pat.search(abstract):
                if term_key == "language model" and ("large language model" in matched_in_abs or "LLM" in matched_in_abs):
                    continue
                matched_in_abs.add(term_key)
                if term_key not in GENERIC_TERMS:
                    specific_term_count += 1
        if matched_in_abs:
            abstract_matches[cat_name] = sorted(matched_in_abs)

    all_matched_cats = sorted(set(title_matches.keys()) | set(abstract_matches.keys()))
    num_cats = len(all_matched_cats)

    category_scores: Dict[str, float] = {}
    base_score = 0.0

    for cat_name in all_matched_cats:
        t_terms = title_matches.get(cat_name, [])
        a_terms = abstract_matches.get(cat_name, [])

        t_pts = sum(1.5 if t in GENERIC_TERMS else 3.0 for t in t_terms)
        a_pts = sum(0.5 if a in GENERIC_TERMS else 1.0 for a in a_terms)
        a_pts = min(a_pts, 2.5)  # Cap abstract per category

        cat_total = t_pts + a_pts
        category_scores[cat_name] = round(cat_total, 2)
        base_score += cat_total

    # Category synergy bonus
    synergy_bonus = 0.0
    if num_cats >= 3:
        synergy_bonus = 4.0
    elif num_cats >= 2:
        synergy_bonus = 2.0
    base_score += synergy_bonus

    # Generic-only safeguard
    generic_cap_applied = False
    if specific_term_count == 0 and base_score > 2.5:
        base_score = 2.5
        generic_cap_applied = True

    # Announce type prior
    announce_bonus = ANNOUNCE_BONUS.get(ann_type, 0.0)
    final_score = round(base_score + announce_bonus, 2)

    details = {
        "final_score": final_score,
        "base_score": round(base_score, 2),
        "announce_type": ann_type,
        "announce_bonus": announce_bonus,
        "synergy_bonus": synergy_bonus,
        "specific_term_count": specific_term_count,
        "generic_cap_applied": generic_cap_applied,
        "matched_categories": all_matched_cats,
        "category_scores": category_scores,
        "title_matches": title_matches,
        "abstract_matches": abstract_matches,
    }

    return final_score, details

# ------------------------------------------------------------------------------
# 4. BACKLOG RECLASSIFICATION & BUDGET TRACKING
# ------------------------------------------------------------------------------

def reclassify_unprocessed_arxiv_articles(conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Safely reclassify existing unprocessed arXiv articles to 'STORED'.
    Articles that are already ANALYZED or NOTIFIED are strictly left untouched.
    Ensures all arXiv articles not in article_pipeline_state are inserted as 'STORED'.
    Returns number of articles updated.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()

        # 1. Ensure any arXiv article missing from article_pipeline_state is inserted as 'STORED'
        cursor.execute(
            """
            INSERT OR IGNORE INTO article_pipeline_state (article_id, stage, created_at, updated_at)
            SELECT id, 'STORED', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            FROM articles
            WHERE (LOWER(source) LIKE '%arxiv%' OR LOWER(link) LIKE '%arxiv.org%')
              AND id NOT IN (SELECT article_id FROM article_pipeline_state)
            """
        )

        # 2. Safely reclassify unprocessed arXiv articles in stage 'NEW' to 'STORED'
        # Crucial safeguard: articles in 'ANALYZED' or 'NOTIFIED' are NOT touched.
        cursor.execute(
            """
            UPDATE article_pipeline_state
            SET stage = 'STORED', updated_at = CURRENT_TIMESTAMP
            WHERE stage = 'NEW'
              AND article_id IN (
                  SELECT id FROM articles
                  WHERE LOWER(source) LIKE '%arxiv%' OR LOWER(link) LIKE '%arxiv.org%'
              )
            """
        )
        reclassified = cursor.rowcount
        conn.commit()

        if reclassified > 0:
            logger.info("Safely reclassified %d unprocessed arXiv articles from 'NEW' to 'STORED'", reclassified)

        return reclassified
    finally:
        if should_close:
            conn.close()


def get_daily_promoted_count(
    target_date_utc: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Return count of arXiv articles promoted on the given UTC date (YYYY-MM-DD)."""
    date_key = target_date_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM arxiv_selection_log WHERE selected_date = ?",
            (date_key,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        if should_close:
            conn.close()

# ------------------------------------------------------------------------------
# 5. PRIORITY PROMOTION PIPELINE
# ------------------------------------------------------------------------------

def promote_top_arxiv_candidates(
    daily_budget: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
    target_date_utc: Optional[str] = None,
) -> List[int]:
    """
    Rank all eligible unprocessed arXiv articles currently stored in SQLite,
    and promote the highest-ranked candidates up to the remaining daily budget.

    Ordering: score DESC, published_at DESC, article_id ASC

    Promoted articles transition from 'STORED' to 'NEW' and are recorded
    in arxiv_selection_log. Unselected articles remain 'STORED'.

    Returns list of promoted article IDs.
    """
    budget = daily_budget if daily_budget is not None else ARXIV_DAILY_ANALYSIS_BUDGET
    date_key = target_date_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        # Check budget consumed today across monitor cycles
        already_promoted = get_daily_promoted_count(date_key, conn=conn)
        remaining_budget = max(0, budget - already_promoted)

        if remaining_budget <= 0:
            logger.info(
                "[ARXIV SELECTION] Daily budget (%d) already exhausted for %s (%d promoted today). Remaining papers stay STORED.",
                budget,
                date_key,
                already_promoted,
            )
            return []

        cursor = conn.cursor()

        # Fetch ALL eligible unprocessed arXiv articles in stage 'STORED'
        cursor.execute(
            """
            SELECT a.id, a.title, a.summary, a.published_at, a.link, p.stage
            FROM articles a
            JOIN article_pipeline_state p ON a.id = p.article_id
            WHERE p.stage = 'STORED'
              AND (LOWER(a.source) LIKE '%arxiv%' OR LOWER(a.link) LIKE '%arxiv.org%')
            """
        )
        candidates = [dict(r) for r in cursor.fetchall()]

        if not candidates:
            logger.debug("[ARXIV SELECTION] No STORED arXiv candidates available for promotion.")
            return []

        # Score every candidate
        scored_candidates = []
        for cand in candidates:
            aid = cand["id"]
            title = cand["title"] or ""
            summary = cand["summary"] or ""
            ann_type = extract_announce_type(summary)
            score, details = score_arxiv_article(title, summary, announce_type=ann_type)
            pub_epoch = parse_published_at_to_epoch(cand.get("published_at"))

            scored_candidates.append({
                "id": aid,
                "title": title,
                "score": score,
                "pub_epoch": pub_epoch,
                "published_at": cand.get("published_at") or "",
                "announce_type": ann_type,
                "details": details,
            })

        # Deterministic sorting: score DESC, published_at DESC, article_id ASC
        scored_candidates.sort(
            key=lambda c: (-c["score"], -c["pub_epoch"], c["id"])
        )

        # Select top candidates up to remaining budget
        to_promote = scored_candidates[:remaining_budget]
        promoted_ids: List[int] = []

        for item in to_promote:
            aid = item["id"]
            sc = item["score"]
            ann = item["announce_type"]

            # Transition stage STORED -> NEW
            cursor.execute(
                """
                UPDATE article_pipeline_state
                SET stage = 'NEW', updated_at = CURRENT_TIMESTAMP
                WHERE article_id = ? AND stage = 'STORED'
                """,
                (aid,),
            )

            # Record in daily selection log
            cursor.execute(
                """
                INSERT OR REPLACE INTO arxiv_selection_log (article_id, selected_date, score, announce_type, promoted_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (aid, date_key, sc, ann),
            )
            promoted_ids.append(aid)

        conn.commit()

        logger.info(
            "[ARXIV SELECTION] Promoted %d arXiv articles to 'NEW' for %s (Budget: %d, Promoted Today: %d/%d)",
            len(promoted_ids),
            date_key,
            budget,
            already_promoted + len(promoted_ids),
            budget,
        )

        return promoted_ids
    finally:
        if should_close:
            conn.close()

# ------------------------------------------------------------------------------
# 6. DIAGNOSTIC REPORTING MECHANISM
# ------------------------------------------------------------------------------

def get_arxiv_selection_report(
    daily_budget: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
    target_date_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate structured diagnostic metrics for the arXiv selection layer:
      - total arXiv articles
      - today's arXiv articles
      - today's budget
      - today's selected count
      - today's remaining budget
      - STORED count
      - NEW count
      - ANALYZED count
      - top 20 current arXiv candidates
    """
    budget = daily_budget if daily_budget is not None else ARXIV_DAILY_ANALYSIS_BUDGET
    date_key = target_date_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()

        # Total arXiv articles
        cursor.execute(
            """
            SELECT COUNT(*) as total FROM articles
            WHERE LOWER(source) LIKE '%arxiv%' OR LOWER(link) LIKE '%arxiv.org%'
            """
        )
        total_arxiv = cursor.fetchone()["total"]

        # Today's arXiv articles (created_at or published_at starting with today's date)
        cursor.execute(
            """
            SELECT COUNT(*) as today_cnt FROM articles
            WHERE (LOWER(source) LIKE '%arxiv%' OR LOWER(link) LIKE '%arxiv.org%')
              AND (created_at LIKE ? OR published_at LIKE ?)
            """,
            (f"{date_key}%", f"{date_key}%"),
        )
        today_arxiv = cursor.fetchone()["today_cnt"]

        # Stage breakdowns for arXiv
        cursor.execute(
            """
            SELECT p.stage, COUNT(*) as cnt
            FROM article_pipeline_state p
            JOIN articles a ON p.article_id = a.id
            WHERE LOWER(a.source) LIKE '%arxiv%' OR LOWER(a.link) LIKE '%arxiv.org%'
            GROUP BY p.stage
            """
        )
        stage_counts = {r["stage"]: r["cnt"] for r in cursor.fetchall()}

        stored_count = stage_counts.get("STORED", 0)
        new_count = stage_counts.get("NEW", 0)
        analyzed_count = stage_counts.get("ANALYZED", 0)

        # Today's selection log counts
        today_selected = get_daily_promoted_count(date_key, conn=conn)
        remaining_budget = max(0, budget - today_selected)

        # Top 20 current arXiv candidates currently in STORED (or overall STORED candidates)
        cursor.execute(
            """
            SELECT a.id, a.title, a.summary, a.published_at, a.link, p.stage
            FROM articles a
            JOIN article_pipeline_state p ON a.id = p.article_id
            WHERE p.stage = 'STORED'
              AND (LOWER(a.source) LIKE '%arxiv%' OR LOWER(a.link) LIKE '%arxiv.org%')
            """
        )
        stored_rows = [dict(r) for r in cursor.fetchall()]

        candidates = []
        for row in stored_rows:
            aid = row["id"]
            title = row["title"] or ""
            summary = row["summary"] or ""
            ann_type = extract_announce_type(summary)
            sc, details = score_arxiv_article(title, summary, announce_type=ann_type)
            pub_epoch = parse_published_at_to_epoch(row.get("published_at"))

            candidates.append({
                "id": aid,
                "title": title,
                "score": sc,
                "announce_type": ann_type,
                "pub_epoch": pub_epoch,
                "published_at": row.get("published_at") or "",
                "matched_categories": details.get("matched_categories", []),
            })

        candidates.sort(key=lambda c: (-c["score"], -c["pub_epoch"], c["id"]))
        top_20 = candidates[:20]

        return {
            "date": date_key,
            "total_arxiv_articles": total_arxiv,
            "today_arxiv_articles": today_arxiv,
            "daily_budget": budget,
            "today_selected_count": today_selected,
            "today_remaining_budget": remaining_budget,
            "stored_count": stored_count,
            "new_count": new_count,
            "analyzed_count": analyzed_count,
            "top_20_candidates": top_20,
        }
    finally:
        if should_close:
            conn.close()


def format_arxiv_selection_report(report_data: Dict[str, Any]) -> str:
    """Format diagnostic report into human-readable text."""
    lines = [
        "============================================================",
        "              ARXIV ANALYSIS SELECTION DIAGNOSTIC           ",
        "============================================================",
        f"UTC Date:               {report_data['date']}",
        f"Total arXiv Articles:   {report_data['total_arxiv_articles']}",
        f"Today's arXiv Ingested: {report_data['today_arxiv_articles']}",
        f"Daily Analysis Budget:  {report_data['daily_budget']}",
        f"Today's Selected Count: {report_data['today_selected_count']}",
        f"Remaining Budget:       {report_data['today_remaining_budget']}",
        "------------------------------------------------------------",
        f"Current arXiv Stages:   STORED: {report_data['stored_count']} | NEW: {report_data['new_count']} | ANALYZED: {report_data['analyzed_count']}",
        "============================================================",
        "TOP 20 CURRENT arXiv CANDIDATES (Ranked by Score, Date, ID):",
        "------------------------------------------------------------",
        f"{'ID':<6} | {'Score':<6} | {'Announce':<12} | {'Title'}",
        "-" * 60,
    ]

    for item in report_data.get("top_20_candidates", []):
        t = item["title"]
        if len(t) > 50:
            t = t[:47] + "..."
        lines.append(f"{item['id']:<6} | {item['score']:<6.2f} | {item['announce_type']:<12} | {t}")

    lines.append("============================================================")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    data = get_arxiv_selection_report()
    print(format_arxiv_selection_report(data))
