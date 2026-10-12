"""Telegram delivery module for ranked AI intelligence updates."""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


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
        ) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TelegramDeliveryError(
            f"Telegram delivery failed: {type(exc).__name__}"
        ) from None
    except json.JSONDecodeError as exc:
        raise TelegramDeliveryError(
            "Telegram returned a non-JSON response"
        ) from None

    if not isinstance(parsed, dict) or not parsed.get("ok"):
        description = (
            parsed.get("description", "Unknown error")
            if isinstance(parsed, dict)
            else "Unknown error"
        )
        raise TelegramDeliveryError(f"Telegram API error: {description}")

    return parsed


def is_article_delivered(
    article_id: int,
    channel: str = "telegram",
    conn=None,
    db_path=None,
) -> bool:
    """Check if an article has already been successfully delivered."""
    from database import get_connection

    should_close = False
    if conn is None:
        try:
            conn = get_connection(db_path)
            should_close = True
        except Exception:
            return False

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM article_deliveries WHERE article_id = ? AND channel = ? AND status = 'sent'",
            (article_id, channel),
        )
        return cursor.fetchone() is not None
    except Exception:
        return False
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass


def record_article_delivery(
    article_id: int,
    channel: str = "telegram",
    score: float | None = None,
    status: str = "sent",
    conn=None,
    db_path=None,
) -> None:
    """Record an article delivery attempt in the database."""
    from database import get_connection

    should_close = False
    if conn is None:
        try:
            conn = get_connection(db_path)
            should_close = True
        except Exception:
            return

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO article_deliveries (article_id, channel, score, status, sent_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(article_id) DO UPDATE SET
                status = excluded.status,
                score = excluded.score,
                sent_at = CURRENT_TIMESTAMP
            """,
            (article_id, channel, score, status),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass


def deliver_ranked_articles(
    ranked_articles: Sequence[dict],
    config: TelegramConfig,
    conn=None,
    db_path=None,
    deduplicate: bool = False,
) -> dict:
    """Deliver the highest-scoring articles and return per-message results."""
    if not config.enabled or not config.bot_token or not config.chat_id:
        if not config.enabled:
            reason = "Telegram disabled"
        elif not config.bot_token:
            reason = "missing TELEGRAM_BOT_TOKEN"
        else:
            reason = "missing TELEGRAM_CHAT_ID."
        return {
            "enabled": False,
            "reason": f"Telegram not configured: {reason}",
            "sent": 0,
            "failed": 0,
            "skipped": max(0, len(ranked_articles) - config.limit),
            "details": [],
        }

    candidates = []
    for article in ranked_articles:
        if float(article.get("final_score", 0.0)) < config.min_score:
            continue
        art_id = article.get("article_id")
        if deduplicate and art_id is not None:
            if is_article_delivered(art_id, channel="telegram", conn=conn, db_path=db_path):
                continue
        candidates.append(article)

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
        art_id = article.get("article_id")
        score_val = float(article.get("final_score", 0.0))
        try:
            response = send_telegram_message(
                config.bot_token,
                config.chat_id,
                message,
            )
            details.append({"ok": True, "article_id": art_id})
            sent += 1
            if art_id is not None:
                record_article_delivery(
                    art_id,
                    channel="telegram",
                    score=score_val,
                    status="sent",
                    conn=conn,
                    db_path=db_path,
                )
        except TelegramDeliveryError as exc:
            details.append(
                {
                    "ok": False,
                    "error": str(exc),
                    "article_id": art_id,
                }
            )
            failed += 1
            if art_id is not None:
                record_article_delivery(
                    art_id,
                    channel="telegram",
                    score=score_val,
                    status="failed",
                    conn=conn,
                    db_path=db_path,
                )

    return {
        "enabled": True,
        "sent": sent,
        "failed": failed,
        "skipped": max(0, len(ranked_articles) - len(selected)),
        "details": details,
    }


def send_notification_for_article(
    article_id: int,
    config: Optional[TelegramConfig] = None,
    conn: Any = None,
    db_path: Any = None,
    mark_skipped_if_disabled: bool = False,
) -> Dict[str, Any]:
    """
    Deliver a single analyzed article to Telegram with strict qualification checks.

    Validates:
    - Article ID is a positive integer
    - Article exists in database
    - Pipeline stage == 'ANALYZED'
    - Analysis status == 'completed'
    - Article has not already been successfully delivered ('sent')
    - TELEGRAM_ENABLED is True and credentials are present
    - Final score meets or exceeds TELEGRAM_MIN_SCORE

    Updates pipeline state and article_deliveries consistently with production semantics.
    Never mutates or processes any other article.
    """
    from database import get_connection
    from ranking.scorer import score_article

    if not isinstance(article_id, int) or article_id <= 0:
        return {
            "ok": False,
            "status": "rejected",
            "reason": f"Invalid article_id: {article_id}. Must be a positive integer.",
            "article_id": article_id,
            "delivered": False,
        }

    if config is None:
        config = TelegramConfig.from_env()

    should_close = False
    if conn is None:
        conn = get_connection(db_path)
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id, a.title, a.source, a.published_at, a.link,
                   p.stage, p.notification_attempts,
                   an.status as analysis_status, an.relevance_score, an.novelty,
                   an.confidence, an.event_type, an.summary
            FROM articles a
            LEFT JOIN article_pipeline_state p ON a.id = p.article_id
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE a.id = ?
            """,
            (article_id,),
        )
        row = cursor.fetchone()

        # 1. Verify article exists
        if not row:
            return {
                "ok": False,
                "status": "rejected",
                "reason": f"Article {article_id} does not exist",
                "article_id": article_id,
                "delivered": False,
            }

        data = dict(row)

        # 2. Verify pipeline stage == 'ANALYZED'
        current_stage = data.get("stage")
        if current_stage != "ANALYZED":
            return {
                "ok": False,
                "status": "rejected",
                "reason": f"Article {article_id} in stage '{current_stage}', expected 'ANALYZED'",
                "article_id": article_id,
                "stage": current_stage,
                "delivered": False,
            }

        # 3. Verify analysis status == 'completed'
        analysis_status = data.get("analysis_status")
        if analysis_status != "completed":
            return {
                "ok": False,
                "status": "rejected",
                "reason": f"Article {article_id} analysis status is '{analysis_status}', expected 'completed'",
                "article_id": article_id,
                "analysis_status": analysis_status,
                "delivered": False,
            }

        # 4. Verify not already successfully delivered
        if is_article_delivered(article_id, channel="telegram", conn=conn, db_path=db_path):
            return {
                "ok": False,
                "status": "rejected",
                "reason": f"Article {article_id} has already been delivered via telegram",
                "article_id": article_id,
                "delivered": False,
            }

        # 5. Compute deterministic score
        scoring_payload = {
            "article_id": article_id,
            "title": data.get("title") or "",
            "source": data.get("source") or "",
            "published_at": data.get("published_at"),
            "relevance_score": data.get("relevance_score"),
            "novelty": data.get("novelty"),
            "confidence": data.get("confidence") if data.get("confidence") is not None else 1.0,
            "event_type": data.get("event_type"),
            "status": "completed",
        }
        ranked = score_article(scoring_payload)
        final_score = ranked.final_score if ranked else 0.0

        # 6. Check Telegram configuration
        if not config.enabled or not config.bot_token or not config.chat_id:
            reason = "Telegram disabled" if not config.enabled else "Telegram credentials missing"
            if mark_skipped_if_disabled:
                cursor.execute(
                    "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                    (article_id,),
                )
                record_article_delivery(
                    article_id,
                    channel="telegram",
                    score=final_score,
                    status="skipped",
                    conn=conn,
                    db_path=db_path,
                )
                conn.commit()
            return {
                "ok": False,
                "status": "disabled",
                "reason": f"Telegram delivery is disabled or not configured: {reason}",
                "article_id": article_id,
                "score": final_score,
                "delivered": False,
            }

        # 7. Respect TELEGRAM_MIN_SCORE
        if final_score < config.min_score:
            logger.debug(
                "[TELEGRAM] Article %d score (%.3f) below threshold (%.2f); skipping",
                article_id,
                final_score,
                config.min_score,
            )
            cursor.execute(
                "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                (article_id,),
            )
            record_article_delivery(
                article_id,
                channel="telegram",
                score=final_score,
                status="skipped",
                conn=conn,
                db_path=db_path,
            )
            conn.commit()
            return {
                "ok": False,
                "status": "skipped",
                "reason": f"Article {article_id} score ({final_score:.3f}) below threshold ({config.min_score:.2f})",
                "article_id": article_id,
                "score": final_score,
                "delivered": False,
            }

        # 8. Format and send message
        message = format_telegram_update(
            {**data, "final_score": final_score},
            position=1,
            total=1,
        )
        try:
            send_telegram_message(config.bot_token, config.chat_id, message)
            logger.info("[ARTICLE %d] notification sent", article_id)
            cursor.execute(
                "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                (article_id,),
            )
            record_article_delivery(
                article_id,
                channel="telegram",
                score=final_score,
                status="sent",
                conn=conn,
                db_path=db_path,
            )
            conn.commit()
            return {
                "ok": True,
                "status": "sent",
                "article_id": article_id,
                "score": final_score,
                "delivered": True,
            }
        except TelegramDeliveryError as tele_err:
            logger.error("[TELEGRAM] failed for article %d: %s", article_id, str(tele_err))
            cursor.execute(
                """
                UPDATE article_pipeline_state SET
                    notification_attempts = notification_attempts + 1,
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE article_id = ?
                """,
                (str(tele_err), article_id),
            )
            record_article_delivery(
                article_id,
                channel="telegram",
                score=final_score,
                status="failed",
                conn=conn,
                db_path=db_path,
            )
            conn.commit()
            return {
                "ok": False,
                "status": "failed",
                "error": str(tele_err),
                "article_id": article_id,
                "score": final_score,
                "delivered": False,
            }
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass
