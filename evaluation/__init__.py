"""
Evaluation package for AI News Intelligence Monitor.
Provides dataset sampling, human labeling schemas, and benchmark verification tools.
"""

from evaluation.sampling import sample_evaluation_dataset, write_evaluation_jsonl
from evaluation.labeler import load_dataset, save_dataset, apply_label, get_labeling_progress

__all__ = [
    "sample_evaluation_dataset",
    "write_evaluation_jsonl",
    "load_dataset",
    "save_dataset",
    "apply_label",
    "get_labeling_progress",
]
