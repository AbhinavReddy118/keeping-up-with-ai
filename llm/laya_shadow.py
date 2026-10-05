"""
Laya Shadow Triage Module (Step 7).

Executes Laya local inference in shadow/triage mode:
- Runs only when LAYA_ENABLED is True.
- Never filters, gates, or discards articles.
- Every article remains fully eligible for Qwen analysis.
- Stores Laya classification metadata in the database for auditing.
- Gracefully handles missing laya library, model load errors, or inference failures.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from config import LAYA_ENABLED, LAYA_MODEL_ID

logger = logging.getLogger(__name__)

# Questions matching canonical Phase 0.3 / 0.4 evaluation setup
LAYA_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "ai_relevance": {
        "type": "choice",
        "instructions": (
            "Determine whether this article has useful relationship to AI models, "
            "research, tools, engineering, companies, or AI industry developments."
        ),
        "criteria": {
            "not_relevant": "The article has little or no useful relationship to AI.",
            "weakly_relevant": "The article has some AI connection but limited value.",
            "moderately_relevant": "The article contains meaningful AI-related information.",
            "highly_relevant": "The article directly concerns important AI models, research, or tools.",
        },
    },
    "event_type": {
        "type": "choice",
        "instructions": "Classify the primary event type of this article.",
        "criteria": {
            "model_release": "New AI model release or weights announcement",
            "research": "AI research paper, methodology, or academic finding",
            "tool_release": "New software tool, library, framework, or developer platform",
            "company_news": "Corporate announcements, business deals, funding, or executive changes",
            "industry_news": "Broader AI ecosystem developments, policy, or market trends",
            "opinion_editorial": "Commentary, think piece, perspective, or analysis",
            "other": "Any other topic or uncategorized event",
        },
    },
    "importance": {
        "type": "choice",
        "instructions": "Rate the overall importance of this article for an AI practitioner.",
        "criteria": {
            "high": "Major breakthrough, frontier model, or critical industry event.",
            "medium": "Valuable incremental finding, practical tool, or notable news.",
            "low": "Minor incremental news or niche technical detail.",
            "ignore": "Off-topic, non-technical commentary, spam, or promotional noise.",
        },
    },
}

_LAYA_AGENT: Any = None
_LAYA_LOAD_ATTEMPTED: bool = False
_LAYA_LOAD_FAILED: bool = False


def is_laya_active() -> bool:
    """Check if Laya shadow mode is enabled via config or environment."""
    return os.getenv("LAYA_ENABLED", str(LAYA_ENABLED)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_laya_agent() -> Any:
    """Retrieve or lazily initialize the singleton Laya inference agent."""
    global _LAYA_AGENT, _LAYA_LOAD_ATTEMPTED, _LAYA_LOAD_FAILED

    if not is_laya_active():
        return None

    if _LAYA_AGENT is not None:
        return _LAYA_AGENT

    if _LAYA_LOAD_FAILED:
        return None

    _LAYA_LOAD_ATTEMPTED = True
    try:
        import laya  # type: ignore

        model_id = os.getenv("LAYA_MODEL_ID", LAYA_MODEL_ID)
        logger.info("[LAYA] Loading Laya model '%s' for shadow triage...", model_id)
        _LAYA_AGENT = laya.load(model_id)
        logger.info("[LAYA] Shadow triage model loaded successfully.")
        return _LAYA_AGENT
    except ImportError:
        _LAYA_LOAD_FAILED = True
        logger.warning(
            "[LAYA] 'laya' package not installed. Laya shadow triage is disabled."
        )
        return None
    except Exception as exc:
        _LAYA_LOAD_FAILED = True
        logger.warning(
            "[LAYA] Failed to load Laya model: %s. Continuing pipeline without Laya.",
            exc,
        )
        return None


def map_laya_probabilities(
    raw_probabilities: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Map Laya 4-way choices to canonical 3-class probabilities:
    MUST_KNOW   = P(high)
    INTERESTING = P(medium)
    NOISE       = P(low) + P(ignore)
    """
    p_mk = round(float(raw_probabilities.get("high", 0.0)), 4)
    p_int = round(float(raw_probabilities.get("medium", 0.0)), 4)
    p_noi = round(
        float(raw_probabilities.get("low", 0.0))
        + float(raw_probabilities.get("ignore", 0.0)),
        4,
    )
    canonical = {
        "MUST_KNOW": p_mk,
        "INTERESTING": p_int,
        "NOISE": p_noi,
    }
    pred = max(canonical.items(), key=lambda x: x[1])[0]
    return pred, canonical


def run_laya_shadow_triage(article: dict) -> Optional[Dict[str, Any]]:
    """
    Execute Laya shadow triage on an article without gating or filtering.

    Returns:
        Dict with triage metadata, or None if Laya is disabled.
        Never raises exceptions; errors are captured in the return dict.
    """
    if not is_laya_active():
        return None

    agent = get_laya_agent()
    if agent is None:
        return {
            "status": "skipped",
            "reason": "Laya agent unavailable",
            "prediction": None,
            "probabilities": None,
        }

    title = str(article.get("title") or "").strip()
    source = str(article.get("source") or "").strip()
    summary = str(article.get("summary") or article.get("content") or "").strip()
    if len(summary) > 1000:
        summary = summary[:1000] + "..."

    input_text = f"Source: {source}\nTitle: {title}\nSummary: {summary}"

    try:
        output = agent.predict(input_text, LAYA_QUESTIONS)
        answers = output.get("answers", {})
        ans_imp = answers.get("importance", {})
        raw_choice = ans_imp.get("choice", "medium")
        raw_probs = ans_imp.get("probabilities", {})

        if raw_probs and isinstance(raw_probs, dict):
            mapped_label, canonical_probs = map_laya_probabilities(raw_probs)
        else:
            # Fallback if model did not return probabilities
            choice_map = {
                "high": "MUST_KNOW",
                "medium": "INTERESTING",
                "low": "NOISE",
                "ignore": "NOISE",
            }
            mapped_label = choice_map.get(raw_choice.lower(), "INTERESTING")
            canonical_probs = {mapped_label: 1.0}

        return {
            "status": "completed",
            "prediction": mapped_label,
            "probabilities": canonical_probs,
            "raw_importance": raw_choice,
            "event_type": answers.get("event_type", {}).get("choice"),
            "ai_relevance": answers.get("ai_relevance", {}).get("choice"),
        }
    except Exception as exc:
        logger.warning(
            "[LAYA] Shadow triage failed for article %s: %s. Continuing.",
            article.get("id"),
            exc,
        )
        return {
            "status": "failed",
            "error": str(exc),
            "prediction": None,
            "probabilities": None,
        }
