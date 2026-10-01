#!/usr/bin/env python3
"""
CLI entry point for Phase 0.2 human article labeling.

Usage:
    .venv/bin/python scripts/label_dataset.py
    .venv/bin/python scripts/label_dataset.py [path_to_jsonl]
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.labeler import run_labeler_cli

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/phase_0_2_labels.jsonl"
    run_labeler_cli(target_path)
