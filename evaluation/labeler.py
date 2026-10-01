"""
Interactive CLI workflow for Phase 0.2 human evaluation labeling.

Provides utilities to load, inspect, label, and validate articles
according to the canonical Importance Rubric (docs/importance_rubric.md).
"""

import json
import logging
import os
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CANONICAL_RUBRIC_VERSION = "1.0"

VALID_LABELS = ["MUST_KNOW", "INTERESTING", "NOISE"]

LABEL_SHORTCUTS = {
    "1": "MUST_KNOW",
    "2": "INTERESTING",
    "3": "NOISE",
    "m": "MUST_KNOW",
    "i": "INTERESTING",
    "n": "NOISE",
    "must_know": "MUST_KNOW",
    "interesting": "INTERESTING",
    "noise": "NOISE",
}


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from a JSON Lines file.

    Args:
        filepath: Path to JSONL file.

    Returns:
        List of article record dictionaries.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {filepath}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} in {filepath}: {e}")

    return records


def save_dataset(filepath: str, dataset: List[Dict[str, Any]]) -> None:
    """
    Atomically save the evaluation dataset to a JSON Lines file.

    Args:
        filepath: Destination filepath.
        dataset: List of article records.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via temporary file
    temp_dir = target.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        for rec in dataset:
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    os.replace(temp_name, target)


def normalize_label(user_input: str) -> Optional[str]:
    """Normalize user input string to canonical label name."""
    clean = user_input.strip().lower()
    return LABEL_SHORTCUTS.get(clean)


def apply_label(
    record: Dict[str, Any],
    label: str,
    reason: str,
    rubric_version: str = CANONICAL_RUBRIC_VERSION,
) -> Dict[str, Any]:
    """
    Validate and apply a human ground-truth label to an evaluation record.

    Args:
        record: Target article dictionary.
        label: Raw label string or shortcut (e.g. '1', 'MUST_KNOW').
        reason: Mandatory short explanation of the labeling decision.
        rubric_version: Canonically applied rubric version.

    Returns:
        Updated article dictionary.
    """
    norm_label = normalize_label(label)
    if norm_label is None or norm_label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label}'. Must be one of {VALID_LABELS} or shortcuts 1, 2, 3."
        )

    clean_reason = reason.strip() if reason else ""
    if not clean_reason:
        raise ValueError("A non-empty human reasoning string is required for every label.")

    record["human_label"] = norm_label
    record["label_reason"] = clean_reason
    record["labeled_at"] = datetime.now(timezone.utc).isoformat()
    record["rubric_version"] = rubric_version
    return record


def get_labeling_progress(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary metrics on the labeling progress of the dataset.

    Args:
        dataset: List of evaluation records.

    Returns:
        Dictionary with counts and percentage complete.
    """
    total = len(dataset)
    labeled = sum(1 for r in dataset if r.get("human_label") is not None)
    unlabeled = total - labeled
    pct = round((labeled / total * 100.0), 1) if total > 0 else 0.0

    must_know = sum(1 for r in dataset if r.get("human_label") == "MUST_KNOW")
    interesting = sum(1 for r in dataset if r.get("human_label") == "INTERESTING")
    noise = sum(1 for r in dataset if r.get("human_label") == "NOISE")

    return {
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "percent_complete": pct,
        "must_know": must_know,
        "interesting": interesting,
        "noise": noise,
    }


def format_article_card(record: Dict[str, Any], index: int, total: int) -> str:
    """Render a terminal card representing the article for human review."""
    divider = "─" * 72
    double_div = "═" * 72
    art_id = record.get("article_id")
    title = record.get("title", "Untitled")
    source = record.get("source", "Unknown Source")
    published = record.get("published_at", "Unknown Date")
    link = record.get("link", "")
    current_label = record.get("human_label")
    current_reason = record.get("label_reason")

    status_tag = f" [Current Label: {current_label}]" if current_label else " [UNLABELED]"

    content = record.get("content") or ""
    summary = record.get("summary") or ""

    if content.strip():
        preview = content[:600].strip() + ("..." if len(content) > 600 else "")
        body_type = "Full Enriched Content (Preview)"
    elif summary.strip():
        preview = summary.strip()
        body_type = "Feed Summary"
    else:
        preview = "(No text body available - evaluate headline and source)"
        body_type = "Headline Only"

    lines = [
        "",
        double_div,
        f" ARTICLE [{index + 1}/{total}] — ID: {art_id}{status_tag}",
        double_div,
        f" Source:    {source}",
        f" Published: {published}",
        f" Title:     {title}",
        f" Link:      {link}",
        divider,
        f" [{body_type}]:",
        f" {preview}",
    ]

    if current_label and current_reason:
        lines.extend([
            divider,
            f" Existing Reason: {current_reason}",
        ])

    lines.append(divider)
    return "\n".join(lines)


def run_labeler_cli(dataset_path: str = "evaluation/phase_0_2_labels.jsonl") -> None:
    """
    Launch interactive terminal labeling CLI for human annotators.

    Args:
        dataset_path: Path to phase_0_2_labels.jsonl dataset.
    """
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    if total == 0:
        print("Dataset is empty. Run evaluation/sampling.py first.")
        return

    # Find first unlabeled article index
    current_idx = 0
    for idx, r in enumerate(dataset):
        if r.get("human_label") is None:
            current_idx = idx
            break

    print("\n" + "=" * 72)
    print(" AI NEWS INTELLIGENCE MONITOR — HUMAN EVALUATION LABELER")
    print(f" Rubric: docs/importance_rubric.md (v{CANONICAL_RUBRIC_VERSION})")
    print("=" * 72)
    print(" Choices: [1] MUST_KNOW  |  [2] INTERESTING  |  [3] NOISE")
    print(" Navigation: [s] Skip  |  [b] Back  |  [c] Full Content  |  [o] Open Link")
    print("             [p] Progress  |  [q] Save & Quit")
    print("=" * 72)

    while 0 <= current_idx < total:
        record = dataset[current_idx]
        progress = get_labeling_progress(dataset)

        print(format_article_card(record, current_idx, total))
        print(f" Progress: {progress['labeled']}/{progress['total']} labeled ({progress['percent_complete']}%)"
              f" | MK: {progress['must_know']} | INT: {progress['interesting']} | NOI: {progress['noise']}")

        prompt = "\n Select [1=MUST_KNOW, 2=INTERESTING, 3=NOISE, s, b, c, o, p, q]: "
        try:
            choice = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting and saving progress...")
            save_dataset(dataset_path, dataset)
            print("Progress saved. Goodbye!")
            return

        if not choice:
            continue

        cmd = choice.lower()

        if cmd == "q":
            print("\nSaving progress and exiting...")
            save_dataset(dataset_path, dataset)
            print(f"Saved {progress['labeled']}/{progress['total']} labels to {dataset_path}.")
            return

        elif cmd == "p":
            print("\n" + "─" * 40)
            print(f" Total Articles:    {progress['total']}")
            print(f" Labeled:           {progress['labeled']} ({progress['percent_complete']}%)")
            print(f" Remaining:         {progress['unlabeled']}")
            print(f"   - MUST_KNOW:     {progress['must_know']}")
            print(f"   - INTERESTING:   {progress['interesting']}")
            print(f"   - NOISE:         {progress['noise']}")
            print("─" * 40)
            input("Press Enter to continue...")
            continue

        elif cmd == "b":
            if current_idx > 0:
                current_idx -= 1
            else:
                print("Already at the first article.")
            continue

        elif cmd == "s":
            current_idx += 1
            continue

        elif cmd == "c":
            text = record.get("content") or record.get("summary") or "No full text available."
            print("\n" + "─" * 72)
            print(" FULL AVAILABLE TEXT:")
            print(text)
            print("─" * 72)
            input("\nPress Enter to return to prompt...")
            continue

        elif cmd == "o":
            link = record.get("link")
            if link:
                print(f"Opening in browser: {link}")
                try:
                    webbrowser.open(link)
                except Exception as e:
                    print(f"Could not open browser: {e}")
            else:
                print("No link recorded for this article.")
            continue

        norm_label = normalize_label(choice)
        if norm_label is not None:
            # Prompt for reason
            while True:
                try:
                    reason = input(f" Reason for [{norm_label}] (required): ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nAction cancelled.")
                    reason = ""
                    break

                if reason:
                    break
                print(" A non-empty reason is mandatory per docs/importance_rubric.md. Please explain briefly.")

            if reason:
                apply_label(record, norm_label, reason, rubric_version=CANONICAL_RUBRIC_VERSION)
                save_dataset(dataset_path, dataset)
                print(f" ✓ Saved Article {record['article_id']} as {norm_label}.")
                current_idx += 1
        else:
            print(f" Unrecognized command '{choice}'. Enter 1, 2, 3, s, b, c, o, p, or q.")

    print("\n" + "=" * 72)
    print(" ALL 150 ARTICLES HAVE BEEN REVIEWED!")
    final_prog = get_labeling_progress(dataset)
    print(f" Final Breakdown: MUST_KNOW={final_prog['must_know']} | "
          f"INTERESTING={final_prog['interesting']} | NOISE={final_prog['noise']}")
    print("=" * 72)
    save_dataset(dataset_path, dataset)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/phase_0_2_labels.jsonl"
    run_labeler_cli(path)
