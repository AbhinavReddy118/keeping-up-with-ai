"""
Progressive Article Deduplication Analysis Module.
Performs read-only multi-stage deduplication and candidate story overlap analysis.
"""

import hashlib
import logging
import re
import sqlite3
import time
import urllib.parse
import difflib
from collections import defaultdict
from typing import Optional
from database import get_connection

logger = logging.getLogger(__name__)

# Common tracking parameters to strip during URL normalization
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "rss",
    "feed",
}

# Common stopwords to exclude from candidate blocking index
BLOCKING_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "to", "for", "on", "with", "at", "by", "from",
    "is", "it", "as", "that", "this", "are", "was", "be", "ai", "model", "models", "new",
    "using", "via", "our", "learning", "deep", "system", "systems", "based", "about"
}


# ==============================================================================
# STAGE 2: URL NORMALIZATION
# ==============================================================================

def normalize_url(url: str | None) -> str:
    """
    Safely normalize an article URL to identify structural variations.
    - Strips URL fragments
    - Strips common marketing/tracking query parameters
    - Normalizes scheme and host to lowercase
    - Normalizes trailing slashes on path
    """
    if not url or not isinstance(url, str):
        return ""

    stripped = url.strip()
    if not stripped:
        return ""

    try:
        parsed = urllib.parse.urlsplit(stripped)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports if present
        if ":" in netloc:
            host, port = netloc.split(":", 1)
            if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                netloc = host

        # Normalize path
        path = parsed.path
        if path and path != "/" and path.endswith("/"):
            path = path[:-1]

        # Filter out tracking query parameters
        q_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered_q = [(k, v) for k, v in q_pairs if k.lower() not in TRACKING_PARAMS]
        new_query = urllib.parse.urlencode(filtered_q)

        # Rebuild URL without fragment
        return urllib.parse.urlunsplit((scheme, netloc, path, new_query, ""))
    except Exception:
        return stripped


# ==============================================================================
# STAGE 4: TITLE NORMALIZATION
# ==============================================================================

def normalize_title_for_dedup(title: str | None) -> str:
    """
    Normalize title string for comparison without over-normalizing.
    - Lowercase
    - Strip punctuation and symbols
    - Collapse multiple whitespace
    """
    if not title or not isinstance(title, str):
        return ""

    cleaned = re.sub(r"[^\w\s]", " ", title.lower())
    return " ".join(cleaned.split())


# ==============================================================================
# STAGE 3: CONTENT FINGERPRINTING
# ==============================================================================

def content_fingerprint(title: str | None, summary: str | None = None) -> str:
    """
    Generate a deterministic SHA-256 hash from normalized article title and summary.
    """
    nt = normalize_title_for_dedup(title)
    ns = normalize_title_for_dedup(summary)
    payload = f"{nt}|||{ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ==============================================================================
# STAGE 5: SIMILARITY METRICS & CANDIDATE BLOCKING
# ==============================================================================

def extract_significant_tokens(title: str | None) -> set[str]:
    """Extract significant keywords for candidate blocking."""
    nt = normalize_title_for_dedup(title)
    return {w for w in nt.split() if w not in BLOCKING_STOPWORDS and len(w) >= 3}


def calculate_title_similarity(title_a: str, title_b: str) -> dict:
    """
    Calculate title similarity metrics using standard library SequenceMatcher and token Jaccard.
    """
    norm_a = normalize_title_for_dedup(title_a)
    norm_b = normalize_title_for_dedup(title_b)

    if not norm_a or not norm_b:
        return {"ratio": 0.0, "jaccard": 0.0, "combined": 0.0}

    if norm_a == norm_b:
        return {"ratio": 1.0, "jaccard": 1.0, "combined": 1.0}

    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    toks_a = extract_significant_tokens(title_a)
    toks_b = extract_significant_tokens(title_b)
    union = toks_a | toks_b
    jaccard = len(toks_a & toks_b) / len(union) if union else 0.0

    combined = (0.7 * ratio) + (0.3 * jaccard)
    return {
        "ratio": round(ratio, 4),
        "jaccard": round(jaccard, 4),
        "combined": round(combined, 4),
    }


def generate_candidate_pairs(articles: list[dict]) -> set[tuple[int, int]]:
    """
    Inverted-index token blocking to reduce comparison complexity from O(n²) to O(c).
    Only pairs sharing >= 2 significant tokens (or 1 token for very short titles) are generated.
    """
    token_sets = [extract_significant_tokens(a.get("title", "")) for a in articles]

    # Inverted index: token -> list of article indices
    inv_index = defaultdict(list)
    for idx, toks in enumerate(token_sets):
        for tok in toks:
            inv_index[tok].append(idx)

    # Count shared tokens between pairs
    pair_shared_counts = defaultdict(int)
    for tok, doc_ids in inv_index.items():
        # Skip tokens that appear in over 150 documents (generic terms)
        if len(doc_ids) < 150:
            for i in range(len(doc_ids)):
                id1 = doc_ids[i]
                for j in range(i + 1, len(doc_ids)):
                    id2 = doc_ids[j]
                    pair = (id1, id2) if id1 < id2 else (id2, id1)
                    pair_shared_counts[pair] += 1

    candidate_pairs = set()
    for (id1, id2), cnt in pair_shared_counts.items():
        if cnt >= 2:
            candidate_pairs.add((id1, id2))
        elif cnt == 1 and (len(token_sets[id1]) <= 2 or len(token_sets[id2]) <= 2):
            candidate_pairs.add((id1, id2))

    return candidate_pairs


# ==============================================================================
# PROGRESSIVE DEDUPLICATION ANALYSIS
# ==============================================================================


def _deduplication_key(article: dict, kind: str) -> str:
    link = article.get("link", "")
    title = article.get("title", "")
    summary = article.get("summary")
    if kind == "url":
        return link.strip()
    if kind == "normalized_url":
        return normalize_url(link)
    if kind == "fingerprint":
        return content_fingerprint(title, summary)
    if kind == "title":
        return normalize_title_for_dedup(title)
    raise ValueError(f"Unknown deduplication key: {kind}")


def deduplicate_articles(articles: list[dict]) -> dict:
    """Return one representative per deduplication group and its mapping."""
    count = len(articles)
    if count == 0:
        return {
            "articles": [],
            "representative_for": [],
            "duplicates": 0,
            "report": analyze_duplicates([]),
        }

    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for kind in ("url", "normalized_url", "fingerprint", "title"):
        groups = defaultdict(list)
        for index, article in enumerate(articles):
            key = _deduplication_key(article, kind)
            if key:
                groups[key].append(index)
        for indexes in groups.values():
            if len(indexes) > 1:
                first = indexes[0]
                for index in indexes[1:]:
                    union(first, index)

    for left, right in generate_candidate_pairs(articles):
        similarity = calculate_title_similarity(
            articles[left].get("title", ""), articles[right].get("title", "")
        )
        ratio = similarity["ratio"]
        jaccard = similarity["jaccard"]
        if ratio >= 0.85 or (ratio >= 0.78 and jaccard >= 0.65):
            union(left, right)

    representatives = {}
    representative_for = []
    first_original_for_root = {}

    for original_index in range(count):
        root = find(original_index)
        if root not in representatives:
            representatives[root] = len(representatives)
            first_original_for_root[root] = original_index
        representative_for.append(representatives[root])

    unique_articles = [
        articles[first_original_for_root[root]] for root in representatives
    ]
    return {
        "articles": unique_articles,
        "representative_for": representative_for,
        "duplicates": count - len(unique_articles),
        "report": analyze_duplicates(articles),
    }


def analyze_duplicates(articles: list[dict]) -> dict:

    """
    Run multi-stage deduplication analysis across an in-memory list of article dicts.
    """
    t_start = time.time()
    total_articles = len(articles)

    if total_articles == 0:
        return {
            "total_articles": 0,
            "exact_url_duplicates": 0,
            "normalized_url_groups": [],
            "content_fingerprint_groups": [],
            "exact_title_groups": [],
            "candidate_groups": [],
            "cross_source_groups": [],
            "execution_time_seconds": 0.0,
        }

    # Stage 1: Exact URL
    url_groups = defaultdict(list)
    for a in articles:
        url_groups[a.get("link", "")].append(a)
    exact_url_dups = sum(len(items) - 1 for items in url_groups.values() if len(items) > 1)

    # Stage 2: Normalized URL
    norm_url_groups = defaultdict(list)
    for a in articles:
        nu = normalize_url(a.get("link", ""))
        norm_url_groups[nu].append(a)
    normalized_url_groups = [
        {"normalized_url": u, "count": len(items), "articles": items}
        for u, items in norm_url_groups.items() if len(items) > 1
    ]

    # Stage 3: Content Fingerprint
    fp_groups = defaultdict(list)
    for a in articles:
        # Only compute fingerprint if both title and non-empty summary exist
        summary = a.get("summary")
        if summary and summary.strip():
            fp = content_fingerprint(a.get("title"), summary)
            fp_groups[fp].append(a)
    content_fingerprint_groups = [
        {"fingerprint": fp, "count": len(items), "articles": items}
        for fp, items in fp_groups.items() if len(items) > 1
    ]

    # Stage 4: Normalized Title Duplicates
    norm_title_groups = defaultdict(list)
    for a in articles:
        nt = normalize_title_for_dedup(a.get("title", ""))
        if nt:
            norm_title_groups[nt].append(a)
    exact_title_groups = [
        {"normalized_title": nt, "count": len(items), "articles": items}
        for nt, items in norm_title_groups.items() if len(items) > 1
    ]

    # Stage 5: Title Similarity via Candidate Blocking
    candidate_pairs = generate_candidate_pairs(articles)
    evaluated_candidate_pairs = []

    for id1, id2 in candidate_pairs:
        a1 = articles[id1]
        a2 = articles[id2]
        t1 = a1.get("title", "")
        t2 = a2.get("title", "")

        sim = calculate_title_similarity(t1, t2)
        ratio = sim["ratio"]
        jaccard = sim["jaccard"]

        # Stage 4 matches are exact title duplicates
        if normalize_title_for_dedup(t1) == normalize_title_for_dedup(t2):
            evaluated_candidate_pairs.append({
                "article_a": a1,
                "article_b": a2,
                "match_type": "exact_title_duplicate",
                "similarity_ratio": ratio,
                "jaccard_similarity": jaccard,
                "reason": "Exact normalized title match across different URLs",
                "cross_source": a1.get("source") != a2.get("source"),
            })
        elif ratio >= 0.85 or (ratio >= 0.78 and jaccard >= 0.65):
            evaluated_candidate_pairs.append({
                "article_a": a1,
                "article_b": a2,
                "match_type": "probable_duplicate",
                "similarity_ratio": ratio,
                "jaccard_similarity": jaccard,
                "reason": f"High lexical similarity ({ratio:.2f}) and token overlap ({jaccard:.2f})",
                "cross_source": a1.get("source") != a2.get("source"),
            })
        elif ratio >= 0.70 or (ratio >= 0.62 and jaccard >= 0.50):
            evaluated_candidate_pairs.append({
                "article_a": a1,
                "article_b": a2,
                "match_type": "possible_duplicate",
                "similarity_ratio": ratio,
                "jaccard_similarity": jaccard,
                "reason": f"Moderate title similarity ({ratio:.2f}) and keyword sharing ({jaccard:.2f})",
                "cross_source": a1.get("source") != a2.get("source"),
            })

    # Sort candidates by similarity descending
    evaluated_candidate_pairs.sort(
        key=lambda p: (
            {"exact_title_duplicate": 3, "probable_duplicate": 2, "possible_duplicate": 1}.get(p["match_type"], 0),
            p["similarity_ratio"]
        ),
        reverse=True
    )

    cross_source_groups = [p for p in evaluated_candidate_pairs if p["cross_source"]]

    t_end = time.time()
    return {
        "total_articles": total_articles,
        "exact_url_duplicates": exact_url_dups,
        "normalized_url_groups": normalized_url_groups,
        "content_fingerprint_groups": content_fingerprint_groups,
        "exact_title_groups": exact_title_groups,
        "candidate_pairs_evaluated": len(candidate_pairs),
        "candidate_groups": evaluated_candidate_pairs,
        "cross_source_groups": cross_source_groups,
        "execution_time_seconds": round(t_end - t_start, 3),
    }


def format_deduplication_report(results: dict, sample_limit: int = 10) -> str:
    """Format deduplication results into a comprehensive, human-readable report."""
    lines = [
        "=" * 80,
        "AI INTELLIGENCE MONITOR - PROGRESSIVE DEDUPLICATION AUDIT REPORT",
        "=" * 80,
        "",
        "1. ANALYSIS OVERVIEW & PERFORMANCE",
        f"  * Total Articles Analyzed     : {results['total_articles']}",
        f"  * Candidate Pairs Evaluated   : {results.get('candidate_pairs_evaluated', 0)} (via Token Blocking)",
        f"  * Execution Time              : {results['execution_time_seconds']} seconds",
        "",
        "2. PROGRESSIVE DEDUPLICATION SUMMARY BY STAGE",
        f"  * Stage 1 - Exact URL Duplicates        : {results['exact_url_duplicates']}",
        f"  * Stage 2 - Normalized URL Groups       : {len(results['normalized_url_groups'])}",
        f"  * Stage 3 - Content Fingerprint Groups  : {len(results['content_fingerprint_groups'])}",
        f"  * Stage 4 - Exact Normalized Title Grps : {len(results['exact_title_groups'])}",
        f"  * Stage 5 - Total Story Candidates      : {len(results['candidate_groups'])}",
        f"              - Exact Title Matches       : {sum(1 for c in results['candidate_groups'] if c['match_type'] == 'exact_title_duplicate')}",
        f"              - Probable Duplicates       : {sum(1 for c in results['candidate_groups'] if c['match_type'] == 'probable_duplicate')}",
        f"              - Possible Duplicates       : {sum(1 for c in results['candidate_groups'] if c['match_type'] == 'possible_duplicate')}",
        f"  * Cross-Source Overlap Candidates       : {len(results['cross_source_groups'])}",
        "",
        "3. CROSS-SOURCE STORY CANDIDATES (Top Matches Across Differing Feeds)",
    ]

    cross_sources = results["cross_source_groups"]
    if cross_sources:
        for idx, p in enumerate(cross_sources[:sample_limit], 1):
            a1, a2 = p["article_a"], p["article_b"]
            lines.extend([
                f"  GROUP {idx:03d} [{p['match_type'].upper()}] (Similarity: {p['similarity_ratio']:.2f}, Jaccard: {p['jaccard_similarity']:.2f})",
                f"    Article 1 [ID {a1.get('id')}]: \"{a1.get('title')}\"",
                f"      Source: {a1.get('source')} | Published: {a1.get('published_at')}",
                f"      Link: {a1.get('link')}",
                f"    Article 2 [ID {a2.get('id')}]: \"{a2.get('title')}\"",
                f"      Source: {a2.get('source')} | Published: {a2.get('published_at')}",
                f"      Link: {a2.get('link')}",
                f"    Reason: {p['reason']}",
                "",
            ])
    else:
        lines.append("  (No cross-source candidate matches met the threshold in this run)\n")

    lines.extend([
        "4. SAME-SOURCE STORY CANDIDATE SAMPLES (High Lexical Similarity)",
    ])

    same_sources = [p for p in results["candidate_groups"] if not p["cross_source"]]
    if same_sources:
        for idx, p in enumerate(same_sources[:sample_limit], 1):
            a1, a2 = p["article_a"], p["article_b"]
            lines.extend([
                f"  SAMPLE {idx:03d} [{p['match_type'].upper()}] (Similarity: {p['similarity_ratio']:.2f}, Jaccard: {p['jaccard_similarity']:.2f})",
                f"    Article 1 [ID {a1.get('id')}]: \"{a1.get('title')}\"",
                f"      Source: {a1.get('source')} | Published: {a1.get('published_at')}",
                f"    Article 2 [ID {a2.get('id')}]: \"{a2.get('title')}\"",
                f"      Source: {a2.get('source')} | Published: {a2.get('published_at')}",
                f"    Reason: {p['reason']}",
                "",
            ])
    else:
        lines.append("  (No same-source candidate matches found)\n")

    lines.extend(["=" * 80, ""])
    return "\n".join(lines)


def run_deduplication_audit(conn: sqlite3.Connection = None) -> dict:
    """
    Execute progressive deduplication audit against database.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, link, summary, source, published_at FROM articles")
        articles = [dict(r) for r in cursor.fetchall()]
        results = analyze_duplicates(articles)
        print(format_deduplication_report(results))
        return results
    finally:
        if should_close:
            conn.close()


if __name__ == "__main__":
    run_deduplication_audit()
