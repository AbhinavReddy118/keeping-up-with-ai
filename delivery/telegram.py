"""Telegram delivery module for ranked AI intelligence updates."""

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence


class TelegramDeliveryError(RuntimeError):
    """Raised when a Telegram delivery request cannot be completed."""


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    min_score: float = 0.0
    limit: int = 5
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        import os

        enabled = os.getenv("TELEGRAM_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            min_score = float(os.getenv("TELEGRAM_MIN_SCORE", "0.0"))
        except (ValueError, TypeError):
            min_score = 0.0
        try:
            limit = max(1, int(os.getenv("TELEGRAM_LIMIT", "5")))
        except (ValueError, TypeError):
            limit = 5
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            min_score=min_score,
            limit=limit,
            enabled=enabled,
        )


def format_telegram_update(
    article: dict,
    position: int,
    total: int,
) -> str:
    """Create a compact Telegram message for one ranked article."""
    score = article.get("final_score")
    score_text = f"{float(score):.3f}" if score is not None else "n/a"
    title = str(article.get("title") or "Untitled article").replace("\n", " ")
    source = str(article.get("source") or "Unknown source")
    published = str(article.get("published_at") or "unknown date")
    link = str(article.get("link") or "")
    link_text = f"\n{link}" if link else ""
    text = (
        f"{position}/{total}. {title}\n"
        f"Source: {source} | Published: {published} | Score: {score_text}"
        f"{link_text}"
    )
    return text[:3900]


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    timeout: float = 15.0,
) -> dict:
    """Send one message to Telegram and return the parsed API response."""
    if not bot_token or not chat_id:
        raise TelegramDeliveryError("Telegram bot token and chat id are required.")

    endpoint = (
        f"https://api.telegram.org/bot{urllib.parse.quote(bot_token, safe='')}"
        "/sendMessage"
    )
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AI-News-Digest/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
            parsed = json.loads(response_text)
    except urllib.error.HTTPError as exc:
        raise TelegramDeliveryError(
            f"Telegram returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TelegramDeliveryError(
            f"Telegram delivery failed: {type(exc).__name__}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TelegramDeliveryError(
            "Telegram returned a non-JSON response"
        ) from exc

    if not isinstance(parsed, dict) or not parsed.get("ok"):
        description = (
            parsed.get("description", "Unknown error")
            if isinstance(parsed, dict)
            else "Unknown error"
        )
        raise TelegramDeliveryError(f"Telegram API error: {description}")

    return parsed


def deliver_ranked_articles(
    ranked_articles: Sequence[dict],
    config: TelegramConfig,
) -> dict:
    """Deliver the highest-scoring articles and return per-message results."""
    if not config.enabled or not config.bot_token or not config.chat_id:
        return {
            "enabled": False,
            "sent": 0,
            "failed": 0,
            "skipped": max(0, len(ranked_articles) - config.limit),
            "details": [],
        }

    candidates = [
        article
        for article in ranked_articles
        if float(article.get("final_score", 0.0)) >= config.min_score
    ]
    selected = candidates[: config.limit]
    details = []
    sent = 0
    failed = 0

    for index, article in enumerate(selected, start=1):
        message = format_telegram_update(
            article,
            position=index,
            total=len(selected),
        )
        try:
            response = send_telegram_message(
                config.bot_token,
                config.chat_id,
                message,
            )
            details.append({"ok": True, "article_id": article.get("article_id")})
            sent += 1
        except TelegramDeliveryError as exc:
            details.append(
                {
                    "ok": False,
                    "error": str(exc),
                    "article_id": article.get("article_id"),
                }
            )
            failed += 1

    return {
        "enabled": True,
        "sent": sent,
        "failed": failed,
        "skipped": max(0, len(ranked_articles) - len(selected)),
        "details": details,
    }
