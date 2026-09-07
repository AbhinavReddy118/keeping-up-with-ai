"""
Ingestion Reliability & Source Health Module.
Evaluates the health and data quality of configured RSS/Atom sources during pipeline execution.
"""

from datetime import datetime, timezone
from typing import Optional

# Maximum days since the newest article before a feed is flagged as stale
STALE_THRESHOLD_DAYS = 90


def evaluate_source_health(
    name: str,
    url: str,
    fetch_success: bool,
    fetched: int,
    normalized: int,
    inserted: int,
    duplicates: int,
    invalid: int,
    missing_dates: int = 0,
    newest_published_at: Optional[str] = None,
    oldest_published_at: Optional[str] = None,
    error: Optional[str] = None,
    current_time: Optional[datetime] = None,
) -> dict:
    """
    Evaluate deterministic health status and diagnostic issues for an ingested source.

    Returns:
        dict containing:
        - name: str
        - url: str
        - status: 'HEALTHY' | 'WARNING' | 'FAILED'
        - fetch_success: bool
        - fetched: int
        - normalized: int
        - inserted: int
        - duplicates: int
        - invalid: int
        - missing_dates: int
        - newest_published_at: str | None
        - oldest_published_at: str | None
        - issues: list[str]
        - error: str | None
    """
    now = current_time if current_time is not None else datetime.now(timezone.utc)
    issues = []

    # 1. Check FAILED conditions
    if not fetch_success:
        issues.append(f"Fetch failed: {error or 'Unknown connection error'}")
        return {
            "name": name,
            "url": url,
            "status": "FAILED",
            "fetch_success": False,
            "fetched": fetched,
            "normalized": normalized,
            "inserted": inserted,
            "duplicates": duplicates,
            "invalid": invalid,
            "missing_dates": missing_dates,
            "newest_published_at": newest_published_at,
            "oldest_published_at": oldest_published_at,
            "issues": issues,
            "error": error,
        }

    if fetched > 0 and normalized == 0:
        issues.append("Feed returned entries, but 0 valid articles could be normalized.")
        return {
            "name": name,
            "url": url,
            "status": "FAILED",
            "fetch_success": True,
            "fetched": fetched,
            "normalized": 0,
            "inserted": inserted,
            "duplicates": duplicates,
            "invalid": invalid,
            "missing_dates": missing_dates,
            "newest_published_at": newest_published_at,
            "oldest_published_at": oldest_published_at,
            "issues": issues,
            "error": error,
        }

    # 2. Check WARNING conditions
    if fetched == 0:
        issues.append("Feed returned 0 entries (empty feed).")

    if invalid > 0:
        issues.append(f"{invalid} of {fetched} entries failed normalization.")

    if missing_dates > 0:
        issues.append(f"{missing_dates} entries missing valid publication dates.")

    if newest_published_at:
        try:
            newest_dt = datetime.fromisoformat(newest_published_at.replace("Z", "+00:00"))
            if newest_dt > now:
                issues.append(f"Suspicious future timestamp detected: {newest_published_at}")
            else:
                days_old = (now - newest_dt).days
                if days_old > STALE_THRESHOLD_DAYS:
                    issues.append(f"Feed content appears stale (newest entry is {days_old} days old).")
        except Exception:
            issues.append(f"Unparseable newest timestamp: {newest_published_at}")

    status = "WARNING" if issues else "HEALTHY"

    return {
        "name": name,
        "url": url,
        "status": status,
        "fetch_success": True,
        "fetched": fetched,
        "normalized": normalized,
        "inserted": inserted,
        "duplicates": duplicates,
        "invalid": invalid,
        "missing_dates": missing_dates,
        "newest_published_at": newest_published_at,
        "oldest_published_at": oldest_published_at,
        "issues": issues,
        "error": None,
    }


def format_health_report(summary: dict) -> str:
    """Format the complete ingestion run health report into a clear human-readable table."""
    lines = [
        "\n" + "=" * 80,
        "INGESTION HEALTH REPORT",
        "=" * 80,
        f"{'Source Name':<26} | {'Fetched':<7} | {'Norm':<5} | {'Insert':<6} | {'Dupl':<5} | {'Inval':<5} | {'Status':<7}",
        "-" * 80,
    ]

    warning_or_failed = []

    for s in summary.get("sources", []):
        status_display = s["status"].upper()
        lines.append(
            f"{s['name'][:26]:<26} | {s['fetched']:<7} | {s['normalized']:<5} | "
            f"{s['inserted']:<6} | {s['duplicates']:<5} | {s['invalid']:<5} | {status_display:<7}"
        )
        if s["status"] in {"WARNING", "FAILED"}:
            warning_or_failed.append(s)

    lines.extend([
        "-" * 80,
        f"TOTALS: Sources: {summary['total_sources']} (Healthy: {summary.get('healthy_sources', 0)}, "
        f"Warning: {summary.get('warning_sources', 0)}, Failed: {summary.get('failed_sources', 0)})",
        f"        Fetched: {summary['total_fetched']} | Normalized: {summary['total_normalized']} | "
        f"Inserted: {summary['total_inserted']} | Duplicates: {summary['total_duplicates']} | Invalid: {summary['total_invalid']}",
        "=" * 80,
    ])

    # Show diagnostic section for WARNING or FAILED feeds
    if warning_or_failed:
        lines.extend([
            "",
            "DIAGNOSTIC DETAILS FOR WARNING / FAILED SOURCES:",
        ])
        for s in warning_or_failed:
            lines.append(f"  * {s['name']} [{s['status']}]:")
            if s.get("error"):
                lines.append(f"      Error: {s['error']}")
            for issue in s.get("issues", []):
                lines.append(f"      Issue: {issue}")
            if s.get("newest_published_at"):
                lines.append(f"      Newest entry: {s['newest_published_at']}")
            if s.get("oldest_published_at"):
                lines.append(f"      Oldest entry: {s['oldest_published_at']}")

    lines.append("")
    return "\n".join(lines)
