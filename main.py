"""
AI News Digest - End-to-end pipeline entry point.

Runs the full pipeline: ingestion, deduplication, LLM analysis,
deterministic ranking, and Telegram delivery.
"""

import argparse
import json
import logging
import sys
import time

from config import (
    LLM_BATCH_SIZE,
    LLM_MAX_RETRIES,
)
from orchestration import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AI News Digest intelligence pipeline."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Run continuously every N seconds (0 = single run).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLM_BATCH_SIZE,
        help="Maximum articles to analyze per run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=LLM_MAX_RETRIES,
        help="Maximum retries for transient LLM failures.",
    )
    parser.add_argument(
        "--rank-limit",
        type=int,
        default=10,
        help="Maximum articles to rank and display.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable progressive deduplication.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip LLM analysis stage.",
    )
    parser.add_argument(
        "--no-delivery",
        action="store_true",
        help="Skip Telegram delivery stage.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for RSS fetch requests.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format for the pipeline summary.",
    )
    return parser


def format_summary(result: dict) -> str:
    lines = []
    ingestion = result.get("ingestion", {})
    lines.append("INGESTION")
    lines.append(f"  Sources: {ingestion.get('total_sources', 0)}")
    lines.append(f"  Fetched: {ingestion.get('total_fetched', 0)}")
    lines.append(f"  Inserted: {ingestion.get('total_inserted', 0)}")
    lines.append(f"  Duplicates: {ingestion.get('total_duplicates', 0)}")

    analysis = result.get("analysis")
    if analysis is not None:
        lines.append("")
        lines.append("ANALYSIS")
        lines.append(f"  Attempted: {analysis.get('total_attempted', 0)}")
        lines.append(f"  Completed: {analysis.get('completed', 0)}")
        lines.append(f"  Failed: {analysis.get('failed', 0)}")

    ranking = result.get("ranking")
    if ranking is not None:
        lines.append("")
        lines.append("RANKING")
        lines.append(f"  Ranked articles: {ranking.get('count', 0)}")

    delivery = result.get("delivery", {})
    if delivery:
        lines.append("")
        lines.append("DELIVERY")
        if delivery.get("enabled"):
            lines.append(f"  Sent: {delivery.get('sent', 0)}")
            lines.append(f"  Failed: {delivery.get('failed', 0)}")
        else:
            lines.append("  Status: Skipped (not configured)")

    return "\n".join(lines)


def run_once(args) -> dict:
    return run_full_pipeline(
        timeout=args.timeout,
        deduplicate=not args.no_dedup,
        analyze=not args.no_analysis,
        analysis_limit=args.batch_size,
        max_retries=args.max_retries,
        rank=True,
        rank_limit=args.rank_limit,
        deliver=not args.no_delivery,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interval > 0:
        logger.info(
            "Starting continuous pipeline (interval=%ds). Ctrl+C to stop.",
            args.interval,
        )
        try:
            while True:
                result = run_once(args)
                if args.output == "json":
                    print(json.dumps(result, indent=2, default=str))
                else:
                    print(format_summary(result))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            return 0
    else:
        result = run_once(args)
        if args.output == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_summary(result))
        return 0


if __name__ == "__main__":
    sys.exit(main())
