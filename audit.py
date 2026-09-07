"""
Database Data Quality Audit Module.
Performs a read-only assessment of the SQLite articles table.
"""

import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from database import get_connection


def normalize_title(title: str | None) -> str:
    """Normalize a title string for fuzzy comparison (lowercase, strip punctuation, collapse whitespace)."""
    if not title:
        return ""
    # Strip non-alphanumeric characters except whitespace
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(cleaned.split())


def audit_database(conn: sqlite3.Connection = None) -> dict:
    """
    Perform a comprehensive read-only audit of stored articles.

    Args:
        conn: Optional active sqlite3.Connection. If None, default project database connection is opened.

    Returns:
        dict containing structured metrics across volume, date coverage, duplicates,
        cross-source overlap, archive depth, and data quality issues.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()

        # 1. Total count & Unique URLs
        cursor.execute("SELECT COUNT(*) as total, COUNT(DISTINCT link) as unique_links FROM articles")
        vol_row = cursor.fetchone()
        total_articles = vol_row["total"]
        unique_urls = vol_row["unique_links"]
        duplicate_urls = total_articles - unique_urls

        # 2. Articles grouped by source
        cursor.execute("""
            SELECT source, COUNT(*) as count 
            FROM articles 
            GROUP BY source 
            ORDER BY count DESC
        """)
        source_rows = cursor.fetchall()
        source_breakdown = []
        for r in source_rows:
            pct = (r["count"] / total_articles * 100.0) if total_articles > 0 else 0.0
            source_breakdown.append({
                "source": r["source"],
                "count": r["count"],
                "percentage": round(pct, 2),
            })

        # 3. Date coverage
        cursor.execute("""
            SELECT MIN(published_at) as earliest, 
                   MAX(published_at) as latest, 
                   SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END) as missing_dates 
            FROM articles
        """)
        date_stats = cursor.fetchone()
        earliest_date = date_stats["earliest"]
        latest_date = date_stats["latest"]
        missing_dates = date_stats["missing_dates"] or 0

        # Articles grouped by month (YYYY-MM)
        cursor.execute("""
            SELECT SUBSTR(published_at, 1, 7) as month, COUNT(*) as count
            FROM articles
            WHERE published_at IS NOT NULL
            GROUP BY month
            ORDER BY month ASC
        """)
        month_rows = cursor.fetchall()
        articles_by_month = [{"month": r["month"], "count": r["count"]} for r in month_rows]

        # 4. Duplicate & Cross-source overlap analysis
        cursor.execute("SELECT id, title, link, source, published_at, summary FROM articles")
        all_articles = cursor.fetchall()

        exact_title_groups = defaultdict(list)
        norm_title_groups = defaultdict(list)

        for art in all_articles:
            t = art["title"].strip() if art["title"] else ""
            exact_title_groups[t].append(art)
            nt = normalize_title(t)
            if nt:
                norm_title_groups[nt].append(art)

        # Exact title duplicates
        exact_title_dups = [
            {"title": t, "count": len(items), "sources": list(set(i["source"] for i in items))}
            for t, items in exact_title_groups.items() if len(items) > 1
        ]

        # Normalized title duplicates
        norm_title_dups = [
            {
                "normalized_title": nt,
                "count": len(items),
                "titles": list(set(i["title"] for i in items)),
                "sources": list(set(i["source"] for i in items)),
            }
            for nt, items in norm_title_groups.items() if len(items) > 1
        ]

        # Cross-source overlap (articles sharing normalized title across >= 2 different sources)
        cross_source_overlap = []
        for nt, items in norm_title_groups.items():
            sources = set(i["source"] for i in items)
            if len(sources) > 1:
                cross_source_overlap.append({
                    "normalized_title": nt,
                    "sample_title": items[0]["title"],
                    "sources": sorted(list(sources)),
                    "count": len(items),
                    "links": [i["link"] for i in items],
                })

        # 5. Feed / archive coverage per source
        cursor.execute("""
            SELECT source, 
                   COUNT(*) as count, 
                   MIN(published_at) as oldest, 
                   MAX(published_at) as newest
            FROM articles
            GROUP BY source
            ORDER BY count DESC
        """)
        archive_rows = cursor.fetchall()
        archive_coverage = []
        for r in archive_rows:
            oldest = r["oldest"]
            newest = r["newest"]
            days_span = None
            if oldest and newest:
                try:
                    d_old = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
                    d_new = datetime.fromisoformat(newest.replace("Z", "+00:00"))
                    days_span = (d_new - d_old).days
                except Exception:
                    pass

            archive_coverage.append({
                "source": r["source"],
                "count": r["count"],
                "oldest_date": oldest,
                "newest_date": newest,
                "span_days": days_span,
            })

        # 6. Data quality problems
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE title IS NULL OR TRIM(title) = ''")
        empty_titles = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE link IS NULL OR (link NOT LIKE 'http://%' AND link NOT LIKE 'https://%')")
        invalid_links = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE source IS NULL OR TRIM(source) = ''")
        missing_sources = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE published_at > ?", (now_utc,))
        future_dates = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE summary IS NULL OR LENGTH(TRIM(summary)) = 0")
        empty_summaries = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) as c FROM articles WHERE summary IS NOT NULL AND LENGTH(TRIM(summary)) > 0 AND LENGTH(TRIM(summary)) < 20")
        short_summaries = cursor.fetchone()["c"]

        # 7. Recent 10 articles
        cursor.execute("""
            SELECT id, source, title, published_at, link 
            FROM articles 
            ORDER BY published_at DESC, id DESC 
            LIMIT 10
        """)
        newest_articles = [dict(r) for r in cursor.fetchall()]

        return {
            "total_articles": total_articles,
            "unique_urls": unique_urls,
            "duplicate_urls": duplicate_urls,
            "source_breakdown": source_breakdown,
            "date_coverage": {
                "earliest": earliest_date,
                "latest": latest_date,
                "missing_dates": missing_dates,
                "months_count": len(articles_by_month),
                "by_month": articles_by_month,
            },
            "duplicate_analysis": {
                "exact_url_duplicates": duplicate_urls,
                "exact_title_duplicates": exact_title_dups,
                "normalized_title_duplicates": norm_title_dups,
            },
            "cross_source_overlap": cross_source_overlap,
            "archive_coverage": archive_coverage,
            "data_quality": {
                "empty_titles": empty_titles,
                "invalid_links": invalid_links,
                "missing_sources": missing_sources,
                "missing_dates": missing_dates,
                "future_dates": future_dates,
                "empty_summaries": empty_summaries,
                "short_summaries": short_summaries,
            },
            "newest_articles": newest_articles,
        }
    finally:
        if should_close:
            conn.close()


def format_audit_report(results: dict) -> str:
    """Format the audit findings dictionary into a clean markdown report."""
    lines = [
        "=" * 80,
        "AI INTELLIGENCE MONITOR - DATABASE DATA QUALITY AUDIT REPORT",
        "=" * 80,
        "",
        "1. DATABASE VOLUME & URL INTEGRITY",
        f"  * Total Articles Stored : {results['total_articles']}",
        f"  * Unique URLs           : {results['unique_urls']}",
        f"  * Duplicate URLs Found  : {results['duplicate_urls']}",
        "",
        "2. SOURCE DISTRIBUTION",
        f"  {'Source Name':<26} | {'Count':<6} | {'Share':<7}",
        "  " + "-" * 45,
    ]

    for s in results["source_breakdown"]:
        lines.append(f"  {s['source']:<26} | {s['count']:<6} | {s['percentage']:>5.1f}%")

    lines.extend([
        "",
        "3. ARCHIVE & TIME SPAN COVERAGE PER SOURCE",
        f"  {'Source Name':<24} | {'Count':<5} | {'Oldest':<10} | {'Newest':<10} | {'Span':<10}",
        "  " + "-" * 68,
    ])

    for a in results["archive_coverage"]:
        old_str = a["oldest_date"][:10] if a["oldest_date"] else "N/A"
        new_str = a["newest_date"][:10] if a["newest_date"] else "N/A"
        if a["span_days"] is not None:
            span_str = f"{a['span_days']}d (~{a['span_days']//365}y)" if a["span_days"] > 365 else f"{a['span_days']} days"
        else:
            span_str = "N/A"
        lines.append(f"  {a['source']:<24} | {a['count']:<5} | {old_str:<10} | {new_str:<10} | {span_str:<10}")

    dc = results["date_coverage"]
    lines.extend([
        "",
        "4. OVERALL DATE RANGE & MONTHLY DISTRIBUTION",
        f"  * Earliest Article Date : {dc['earliest']}",
        f"  * Latest Article Date   : {dc['latest']}",
        f"  * Articles Missing Date : {dc['missing_dates']}",
        f"  * Total Active Months   : {dc['months_count']}",
        "  * Recent Monthly Counts (Latest 6 Months):",
    ])

    for m in dc["by_month"][-6:]:
        lines.append(f"    - {m['month']}: {m['count']} articles")

    da = results["duplicate_analysis"]
    lines.extend([
        "",
        "5. DUPLICATE & OVERLAP ANALYSIS",
        f"  * Exact URL Duplicates in DB        : {da['exact_url_duplicates']}",
        f"  * Exact Title Duplicate Groups      : {len(da['exact_title_duplicates'])}",
        f"  * Normalized Title Duplicate Groups : {len(da['normalized_title_duplicates'])}",
        f"  * Cross-Source Title Overlaps       : {len(results['cross_source_overlap'])}",
    ])

    if da["exact_title_duplicates"]:
        lines.append("  * Exact Title Duplicates Detected (Same Source):")
        for d in da["exact_title_duplicates"][:5]:
            lines.append(f"    - \"{d['title']}\" (count={d['count']}, sources={d['sources']})")

    if results["cross_source_overlap"]:
        lines.append("  * Cross-Source Overlaps Detected:")
        for o in results["cross_source_overlap"][:5]:
            lines.append(f"    - \"{o['sample_title']}\" (Sources: {', '.join(o['sources'])})")
    else:
        lines.append("  * Cross-Source Overlaps: 0 (No identical headlines across differing feeds in current snapshot).")

    dq = results["data_quality"]
    lines.extend([
        "",
        "6. DATA QUALITY ANOMALIES & HYGIENE",
        f"  * Missing/Empty Titles     : {dq['empty_titles']}",
        f"  * Malformed/Invalid Links  : {dq['invalid_links']}",
        f"  * Missing Sources          : {dq['missing_sources']}",
        f"  * Missing Published Dates  : {dq['missing_dates']}",
        f"  * Suspicious Future Dates  : {dq['future_dates']}",
        f"  * Empty Summaries          : {dq['empty_summaries']} ({round(dq['empty_summaries']/results['total_articles']*100, 1) if results['total_articles'] else 0}%)",
        f"  * Short Summaries (<20 ch) : {dq['short_summaries']}",
        "",
        "7. NEWEST 10 ARTICLES IN DATABASE",
    ])

    for idx, art in enumerate(results["newest_articles"], 1):
        lines.append(f"  [{idx}] {art['published_at']} | {art['source']} | {art['title'][:65]}")
        lines.append(f"      Link: {art['link']}")

    lines.extend(["=" * 80, ""])
    return "\n".join(lines)


def run_audit(conn: sqlite3.Connection = None) -> dict:
    """Execute audit and print formatted report."""
    results = audit_database(conn=conn)
    print(format_audit_report(results))
    return results


if __name__ == "__main__":
    run_audit()
