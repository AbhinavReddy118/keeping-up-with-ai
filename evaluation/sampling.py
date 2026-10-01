"""
Deterministic dataset sampling for Phase 0.2 human evaluation.

Implements stratified sampling across all 7 sources and publication eras
in data/news_digest.db to construct a representative 150-article dataset.
"""

import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default random seed for strict mathematical reproducibility
DEFAULT_SAMPLING_SEED = 42
DEFAULT_TARGET_COUNT = 150
DEFAULT_RUBRIC_VERSION = "1.0"

# Explicit stratification specification across available database sources and eras
DEFAULT_STRATA: List[Tuple[str, str, int]] = [
    # (Stratum Label, SQL WHERE Condition, Target Sample Size)
    (
        "OpenAI News (Historical 2015-2023)",
        "source = 'OpenAI News' AND substr(published_at, 1, 4) <= '2023'",
        14,
    ),
    (
        "OpenAI News (Recent 2024-2026)",
        "source = 'OpenAI News' AND substr(published_at, 1, 4) >= '2024'",
        28,
    ),
    (
        "Hugging Face Blog (Historical 2020-2023)",
        "source = 'Hugging Face Blog' AND substr(published_at, 1, 4) <= '2023'",
        12,
    ),
    (
        "Hugging Face Blog (Recent 2024-2026)",
        "source = 'Hugging Face Blog' AND substr(published_at, 1, 4) >= '2024'",
        23,
    ),
    (
        "Google DeepMind Blog (2025)",
        "source = 'Google DeepMind Blog' AND substr(published_at, 1, 4) = '2025'",
        7,
    ),
    (
        "Google DeepMind Blog (2026)",
        "source = 'Google DeepMind Blog' AND substr(published_at, 1, 4) = '2026'",
        18,
    ),
    (
        "arXiv CS.AI Papers (Preprints)",
        "source = 'arXiv CS.AI Papers'",
        26,
    ),
    (
        "Hacker News AI (Community)",
        "source = 'Hacker News AI'",
        10,
    ),
    (
        "MIT Tech Review AI (Journalism/Op-Ed)",
        "source = 'MIT Tech Review AI'",
        6,
    ),
    (
        "Microsoft Research (Industry Research)",
        "source = 'Microsoft Research'",
        6,
    ),
]


def sample_evaluation_dataset(
    db_path: str = "data/news_digest.db",
    seed: int = DEFAULT_SAMPLING_SEED,
    strata: Optional[List[Tuple[str, str, int]]] = None,
    rubric_version: str = DEFAULT_RUBRIC_VERSION,
) -> List[Dict[str, Any]]:
    """
    Deterministically sample articles from SQLite database according to stratification plan.

    Args:
        db_path: Path to SQLite database.
        seed: Random seed for pseudo-random number generator.
        strata: List of (name, sql_filter, count) tuples. Defaults to DEFAULT_STRATA.
        rubric_version: Canonical rubric version string to record in each record.

    Returns:
        List of un-labeled article dictionaries prepared for human annotation.
    """
    active_strata = strata if strata is not None else DEFAULT_STRATA
    rng = random.Random(seed)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sampled_articles: List[Dict[str, Any]] = []
    selected_ids_set = set()

    try:
        for stratum_name, sql_cond, count in active_strata:
            query = f"""
                SELECT id, title, source, published_at, link, summary, content, enrichment_status
                FROM articles
                WHERE {sql_cond}
                ORDER BY id ASC
            """
            cursor.execute(query)
            pool = [dict(row) for row in cursor.fetchall()]

            if len(pool) < count:
                logger.warning(
                    f"Stratum '{stratum_name}' requested {count} articles, but pool only has {len(pool)}. Taking all available."
                )
                chosen = pool
            else:
                chosen = rng.sample(pool, count)

            for art in chosen:
                art_id = art["id"]
                if art_id in selected_ids_set:
                    continue
                selected_ids_set.add(art_id)

                # Initialize unpopulated human evaluation schema
                sampled_articles.append(
                    {
                        "article_id": art_id,
                        "title": art["title"],
                        "source": art["source"],
                        "published_at": art["published_at"],
                        "link": art["link"],
                        "summary": art["summary"],
                        "content": art["content"],
                        "enrichment_status": art["enrichment_status"],
                        "stratum": stratum_name,
                        "human_label": None,
                        "label_reason": None,
                        "labeled_at": None,
                        "rubric_version": rubric_version,
                    }
                )

        # Sort final list by article_id ascending for deterministic presentation
        sampled_articles.sort(key=lambda x: x["article_id"])
        return sampled_articles

    finally:
        conn.close()


def write_evaluation_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """
    Write dataset records to a JSON Lines file.

    Args:
        records: List of article record dictionaries.
        filepath: Target output file path.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "data/news_digest.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "evaluation/phase_0_2_labels.jsonl"

    print(f"Sampling evaluation dataset from {db} using seed {DEFAULT_SAMPLING_SEED}...")
    dataset = sample_evaluation_dataset(db_path=db, seed=DEFAULT_SAMPLING_SEED)
    write_evaluation_jsonl(dataset, out)
    print(f"Successfully generated {len(dataset)} evaluation records in {out}.")
