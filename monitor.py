"""
Continuous Intelligence Monitor (Step 2 - 13).

Single long-running unattended process:
SOURCE -> POLL -> INGEST -> NORMALIZE -> DEDUPLICATE -> STORE ->
ENRICH -> OPTIONAL LAYA SHADOW TRIAGE -> QWEN ANALYSIS -> RANK ->
WEB UI -> TELEGRAM -> REPEAT

Guarantees:
- Restart-safe and idempotent
- Bounded batch analysis (no historical backfill)
- Isolated failures (source, enrichment, LLM, or Telegram errors never crash the monitor)
- Observable structured logging with periodic heartbeats
- Clean signal handling (SIGINT, SIGTERM)
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ARXIV_DAILY_ANALYSIS_BUDGET,
    DATABASE_PATH,
    LLM_BATCH_SIZE,
    LLM_MAX_RETRIES,
    MONITOR_BATCH_SIZE,
    MONITOR_HEARTBEAT_SECONDS,
    MONITOR_POLL_INTERVAL_TIER1_MINUTES,
    MONITOR_POLL_INTERVAL_TIER2_MINUTES,
    MONITOR_POLL_INTERVAL_TIER3_HOURS,
    TELEGRAM_MIN_SCORE,
)
from arxiv_selector import (
    is_arxiv_article,
    promote_top_arxiv_candidates,
    reclassify_unprocessed_arxiv_articles,
)
from database import get_connection, init_db
from deduplication import deduplicate_articles
from delivery.telegram import (
    TelegramConfig,
    format_telegram_update,
    is_article_delivered,
    record_article_delivery,
    send_telegram_message,
    TelegramDeliveryError,
)
from ingestion.enricher import enrich_article
from ingestion.normalizer import normalize_article
from ingestion.rss_reader import fetch_rss_feed_conditional
from llm.batch import BatchProcessor
from llm.laya_shadow import is_laya_active, run_laya_shadow_triage
from ranking.scorer import score_article
from sources import DEFAULT_SOURCES
from storage import save_article

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("monitor")

# Source Tier Definitions matching Step 3
SOURCE_TIERS = {
    "OpenAI News": ("tier1", MONITOR_POLL_INTERVAL_TIER1_MINUTES * 60),
    "Google DeepMind Blog": ("tier1", MONITOR_POLL_INTERVAL_TIER1_MINUTES * 60),
    "Microsoft Research": ("tier1", MONITOR_POLL_INTERVAL_TIER1_MINUTES * 60),
    "Hugging Face Blog": ("tier1", MONITOR_POLL_INTERVAL_TIER1_MINUTES * 60),
    "MIT Tech Review AI": ("tier2", MONITOR_POLL_INTERVAL_TIER2_MINUTES * 60),
    "Hacker News AI": ("tier2", MONITOR_POLL_INTERVAL_TIER2_MINUTES * 60),
    "arXiv CS.AI Papers": ("tier3", MONITOR_POLL_INTERVAL_TIER3_HOURS * 3600),
}


class ContinuousMonitor:
    """Orchestrates unattended continuous monitoring, ingestion, analysis, and notification."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = MONITOR_BATCH_SIZE,
        max_retries: int = LLM_MAX_RETRIES,
        heartbeat_interval: int = MONITOR_HEARTBEAT_SECONDS,
        provider: Any = None,
        arxiv_daily_budget: int = ARXIV_DAILY_ANALYSIS_BUDGET,
    ):
        self.db_path = db_path or str(DATABASE_PATH)
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.heartbeat_interval = max(5, heartbeat_interval)
        self.provider = provider
        self.arxiv_daily_budget = max(0, arxiv_daily_budget)
        self.running = False
        self.stop_event = threading.Event()
        self._last_heartbeat_time = 0.0

        init_db(self.db_path)
        self.init_source_states()
        with self._open_conn() as conn:
            reclassify_unprocessed_arxiv_articles(conn=conn)
        self.batch_processor = BatchProcessor(
            provider=self.provider,
            batch_size=self.batch_size,
            max_retries=self.max_retries,
            db_path=self.db_path,
        )

    def _open_conn(self):
        conn = get_connection(self.db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def init_source_states(self) -> None:
        """Seed source_state table with configured sources if missing."""
        with self._open_conn() as conn:
            cursor = conn.cursor()
            for src in DEFAULT_SOURCES:
                if not src.get("enabled", True):
                    continue
                name = src["name"]
                url = src["url"]
                tier, interval = SOURCE_TIERS.get(name, ("tier2", 1800))
                cursor.execute(
                    """
                    INSERT INTO source_state (
                        source_name, url, tier, poll_interval_seconds, next_due_at
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_name) DO UPDATE SET
                        url = excluded.url,
                        tier = excluded.tier,
                        poll_interval_seconds = excluded.poll_interval_seconds
                    """,
                    (name, url, tier, interval),
                )
            conn.commit()

    def get_due_sources(self, force_all: bool = False) -> List[Dict[str, Any]]:
        """Return sources that are currently due for polling."""
        with self._open_conn() as conn:
            cursor = conn.cursor()
            if force_all:
                cursor.execute("SELECT * FROM source_state ORDER BY source_name ASC")
            else:
                cursor.execute(
                    """
                    SELECT * FROM source_state 
                    WHERE next_due_at <= CURRENT_TIMESTAMP 
                    ORDER BY next_due_at ASC
                    """
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def poll_single_source(self, source_record: Dict[str, Any]) -> int:
        """
        Poll a single source conditionally.
        Isolates errors so one source failure never crashes the monitor.
        Returns the count of newly inserted articles.
        """
        name = source_record["source_name"]
        url = source_record["url"]
        interval = source_record.get("poll_interval_seconds", 1800)
        etag = source_record.get("etag")
        last_modified = source_record.get("last_modified")
        failures = source_record.get("consecutive_failures", 0)

        logger.info("[%s] polling", name.upper())
        new_inserted = 0

        try:
            entries, new_etag, new_last_mod, not_modified = fetch_rss_feed_conditional(
                url, timeout=15, etag=etag, last_modified=last_modified
            )

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            next_due_dt = datetime.now(timezone.utc) + timedelta(seconds=interval)
            next_due_iso = next_due_dt.strftime("%Y-%m-%d %H:%M:%S")

            if not_modified:
                logger.info("[%s] 0 new articles (feed unchanged / 304)", name.upper())
                with self._open_conn() as conn:
                    conn.execute(
                        """
                        UPDATE source_state SET
                            last_polled_at = ?,
                            last_success_at = ?,
                            consecutive_failures = 0,
                            next_due_at = ?,
                            last_error = NULL
                        WHERE source_name = ?
                        """,
                        (now_iso, now_iso, next_due_iso, name),
                    )
                    conn.commit()
                return 0

            # Normalize entries
            source_cfg = {"name": name, "url": url, "type": "rss"}
            normalized = []
            for entry in entries:
                try:
                    norm = normalize_article(entry, source_cfg)
                    normalized.append(norm)
                except Exception as norm_err:
                    logger.debug("[%s] normalization skipped entry: %s", name.upper(), norm_err)

            # Deduplicate against current batch
            if normalized:
                dedup_result = deduplicate_articles(normalized)
                surviving = dedup_result["articles"]
            else:
                surviving = []

            # Persist newly discovered articles into DB and pipeline queue
            with self._open_conn() as conn:
                for article in surviving:
                    try:
                        is_new = save_article(article, conn=conn)
                        if is_new:
                            # Retrieve assigned article id
                            c = conn.cursor()
                            c.execute("SELECT id FROM articles WHERE link = ?", (article["link"],))
                            row = c.fetchone()
                            if row:
                                aid = row["id"]
                                is_arxiv = is_arxiv_article(name, article.get("link", ""))
                                initial_stage = "STORED" if is_arxiv else "NEW"
                                conn.execute(
                                    """
                                    INSERT OR IGNORE INTO article_pipeline_state (
                                        article_id, stage, created_at, updated_at
                                    ) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                    """,
                                    (aid, initial_stage),
                                )
                                logger.info("[ARTICLE %d] stored (stage: %s)", aid, initial_stage)
                                new_inserted += 1
                    except Exception as save_err:
                        logger.error("[%s] storage error for '%s': %s", name.upper(), article.get("link"), save_err)

                # Update source state on success
                conn.execute(
                    """
                    UPDATE source_state SET
                        last_polled_at = ?,
                        last_success_at = ?,
                        consecutive_failures = 0,
                        next_due_at = ?,
                        etag = ?,
                        last_modified = ?,
                        last_error = NULL
                    WHERE source_name = ?
                    """,
                    (now_iso, now_iso, next_due_iso, new_etag, new_last_mod, name),
                )
                conn.commit()

            logger.info("[%s] %d new articles", name.upper(), new_inserted)
            return new_inserted

        except Exception as exc:
            failures += 1
            # Exponential backoff capped at 24 hours
            backoff_sec = min(interval * (2 ** min(failures, 4)), 86400)
            next_due_dt = datetime.now(timezone.utc) + timedelta(seconds=backoff_sec)
            next_due_iso = next_due_dt.strftime("%Y-%m-%d %H:%M:%S")
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            logger.error("[%s] failed: %s (retry in %ds)", name.upper(), exc, backoff_sec)
            with self._open_conn() as conn:
                conn.execute(
                    """
                    UPDATE source_state SET
                        last_polled_at = ?,
                        consecutive_failures = ?,
                        next_due_at = ?,
                        last_error = ?
                    WHERE source_name = ?
                    """,
                    (now_iso, failures, next_due_iso, str(exc), name),
                )
                conn.commit()
            return 0

    def poll_due_sources(self, force_all: bool = False) -> int:
        """Poll all currently due sources and return total newly inserted articles."""
        due = self.get_due_sources(force_all=force_all)
        total_inserted = 0
        for src in due:
            if self.stop_event.is_set():
                break
            inserted = self.poll_single_source(src)
            total_inserted += inserted
        return total_inserted

    def select_and_promote_arxiv_articles(self) -> int:
        """
        Promote top-ranked stored arXiv articles up to the daily analysis budget.
        Returns count of newly promoted articles.
        """
        with self._open_conn() as conn:
            promoted_ids = promote_top_arxiv_candidates(
                daily_budget=self.arxiv_daily_budget,
                conn=conn,
            )
            return len(promoted_ids)

    def process_pending_articles(self) -> int:
        """
        Process unanalyzed articles currently in the pipeline queue.
        Stages:
          1. Webpage Content Enrichment (with fallback)
          2. Optional Laya Shadow Triage
          3. Qwen Structured Analysis (with retry)
        Returns count of successfully analyzed articles.
        """
        with self._open_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.id, a.title, a.link, a.summary, a.source, a.published_at, a.content,
                       p.stage, p.analysis_attempts
                FROM articles a
                JOIN article_pipeline_state p ON a.id = p.article_id
                WHERE p.stage IN ('NEW', 'ENRICHED', 'ANALYSIS_PENDING')
                ORDER BY
                    CASE WHEN a.published_at IS NULL OR trim(a.published_at) = ''
                         THEN 1 ELSE 0 END,
                    a.published_at DESC,
                    a.id DESC
                LIMIT ?
                """,
                (self.batch_size,),
            )
            pending_articles = [dict(r) for r in cursor.fetchall()]

        if not pending_articles:
            return 0

        completed_count = 0
        for art in pending_articles:
            if self.stop_event.is_set():
                break

            aid = art["id"]
            stage = art["stage"]

            # --- STAGE 1: ENRICHMENT ---
            if stage == "NEW":
                try:
                    with self._open_conn() as conn:
                        enriched_rec = enrich_article(aid, conn=conn)
                        st = enriched_rec.get("enrichment_status")
                        if st == "enriched":
                            logger.info("[ARTICLE %d] enrichment complete", aid)
                        elif st == "not_needed":
                            logger.info("[ARTICLE %d] enrichment not needed (RSS summary sufficient)", aid)
                        else:
                            logger.warning(
                                "[ARTICLE %d] enrichment failed: %s (falling back to RSS summary)",
                                aid,
                                enriched_rec.get("enrichment_error"),
                            )

                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = 'ENRICHED',
                                enrichment_attempts = enrichment_attempts + 1,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (aid,),
                        )
                        conn.commit()
                except Exception as enrich_exc:
                    logger.error("[ARTICLE %d] enrichment error: %s. Proceeding to analysis.", aid, enrich_exc)
                    with self._open_conn() as conn:
                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = 'ENRICHED',
                                enrichment_attempts = enrichment_attempts + 1,
                                last_error = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (str(enrich_exc), aid),
                        )
                        conn.commit()

            # --- STAGE 2: OPTIONAL LAYA SHADOW TRIAGE ---
            if is_laya_active():
                try:
                    laya_result = run_laya_shadow_triage(art)
                    if laya_result:
                        pred = laya_result.get("prediction")
                        meta_json = json.dumps(laya_result)
                        logger.info("[ARTICLE %d] Laya shadow triage: %s", aid, pred)
                        with self._open_conn() as conn:
                            conn.execute(
                                """
                                UPDATE article_pipeline_state SET
                                    laya_status = ?,
                                    laya_prediction = ?,
                                    laya_metadata = ?,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE article_id = ?
                                """,
                                (laya_result.get("status", "completed"), pred, meta_json, aid),
                            )
                            conn.commit()
                except Exception as laya_exc:
                    logger.warning("[ARTICLE %d] Laya shadow triage failed: %s. Continuing.", aid, laya_exc)

            # --- STAGE 3: QWEN ANALYSIS ---
            logger.info("[ARTICLE %d] analysis started", aid)
            # Fetch latest article data including any enriched content
            with self._open_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM articles WHERE id = ?", (aid,))
                fresh_art = dict(cursor.fetchone())

            try:
                with self._open_conn() as conn:
                    analysis_summary = self.batch_processor.process_single_article(fresh_art, conn=conn)

                    if analysis_summary["status"] == "completed":
                        score = analysis_summary.get("relevance_score")
                        logger.info("[ARTICLE %d] analysis complete (relevance: %s)", aid, score)
                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = 'ANALYZED',
                                analysis_attempts = analysis_attempts + ?,
                                last_error = NULL,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (analysis_summary.get("attempts", 1), aid),
                        )
                        conn.commit()
                        completed_count += 1
                    else:
                        err = analysis_summary.get("error", "Unknown error")
                        logger.error("[ARTICLE %d] analysis failed: %s", aid, err)
                        conn.execute(
                            """
                            UPDATE article_pipeline_state SET
                                stage = CASE WHEN analysis_attempts >= ? THEN 'FAILED' ELSE 'ANALYSIS_PENDING' END,
                                analysis_attempts = analysis_attempts + ?,
                                last_error = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE article_id = ?
                            """,
                            (self.max_retries, analysis_summary.get("attempts", 1), err, aid),
                        )
                        conn.commit()
            except Exception as batch_exc:
                logger.error("[ARTICLE %d] analysis exception: %s", aid, batch_exc)
                with self._open_conn() as conn:
                    conn.execute(
                        """
                        UPDATE article_pipeline_state SET
                            stage = CASE WHEN analysis_attempts >= ? THEN 'FAILED' ELSE 'ANALYSIS_PENDING' END,
                            analysis_attempts = analysis_attempts + 1,
                            last_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE article_id = ?
                        """,
                        (self.max_retries, str(batch_exc), aid),
                    )
                    conn.commit()

        return completed_count

    def send_notifications(self) -> int:
        """
        Deliver analyzed articles to Telegram with article-level deduplication.
        Returns count of sent notifications.
        """
        with self._open_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.id as article_id, a.title, a.source, a.published_at, a.link,
                       an.relevance_score, an.novelty, an.confidence, an.event_type, an.summary
                FROM articles a
                JOIN article_analysis an ON a.id = an.article_id
                JOIN article_pipeline_state p ON a.id = p.article_id
                WHERE p.stage = 'ANALYZED' AND an.status = 'completed'
                ORDER BY a.id ASC
                """
            )
            analyzed = [dict(r) for r in cursor.fetchall()]

        if not analyzed:
            return 0

        config = TelegramConfig.from_env()
        sent_count = 0

        for art in analyzed:
            aid = art["article_id"]

            # Compute score deterministically
            scoring_payload = {
                "article_id": aid,
                "title": art["title"],
                "source": art["source"],
                "published_at": art.get("published_at"),
                "relevance_score": art.get("relevance_score"),
                "novelty": art.get("novelty"),
                "confidence": art.get("confidence", 1.0),
                "event_type": art.get("event_type"),
                "status": "completed",
            }
            ranked = score_article(scoring_payload)
            final_score = ranked.final_score if ranked else 0.0

            # Check if Telegram is configured
            if not config.enabled or not config.bot_token or not config.chat_id:
                # Mark as processed without sending
                with self._open_conn() as conn:
                    conn.execute(
                        "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                        (aid,),
                    )
                    record_article_delivery(aid, channel="telegram", score=final_score, status="skipped", conn=conn)
                    conn.commit()
                continue

            # Check importance threshold
            if final_score < config.min_score:
                logger.debug(
                    "[TELEGRAM] Article %d score (%.3f) below threshold (%.2f); skipping",
                    aid,
                    final_score,
                    config.min_score,
                )
                with self._open_conn() as conn:
                    conn.execute(
                        "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                        (aid,),
                    )
                    record_article_delivery(aid, channel="telegram", score=final_score, status="skipped", conn=conn)
                    conn.commit()
                continue

            # Check deduplication
            with self._open_conn() as conn:
                already_delivered = is_article_delivered(aid, channel="telegram", conn=conn)

            if already_delivered:
                with self._open_conn() as conn:
                    conn.execute(
                        "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                        (aid,),
                    )
                    conn.commit()
                continue

            # Deliver update
            message = format_telegram_update(
                {**art, "final_score": final_score},
                position=1,
                total=1,
            )
            try:
                send_telegram_message(config.bot_token, config.chat_id, message)
                logger.info("[ARTICLE %d] notification sent", aid)
                sent_count += 1
                with self._open_conn() as conn:
                    conn.execute(
                        "UPDATE article_pipeline_state SET stage = 'NOTIFIED', updated_at = CURRENT_TIMESTAMP WHERE article_id = ?",
                        (aid,),
                    )
                    record_article_delivery(aid, channel="telegram", score=final_score, status="sent", conn=conn)
                    conn.commit()
            except TelegramDeliveryError as tele_err:
                logger.error("[TELEGRAM] failed for article %d: %s", aid, tele_err)
                with self._open_conn() as conn:
                    conn.execute(
                        """
                        UPDATE article_pipeline_state SET
                            notification_attempts = notification_attempts + 1,
                            last_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE article_id = ?
                        """,
                        (str(tele_err), aid),
                    )
                    record_article_delivery(aid, channel="telegram", score=final_score, status="failed", conn=conn)
                    conn.commit()

        return sent_count

    def update_heartbeat(self, status: str = "running") -> None:
        """Write monitor heartbeat and process telemetry into the database."""
        pid = os.getpid()
        with self._open_conn() as conn:
            conn.execute(
                """
                INSERT INTO monitor_heartbeat (id, last_heartbeat, status, pid, started_at)
                VALUES (1, CURRENT_TIMESTAMP, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    last_heartbeat = CURRENT_TIMESTAMP,
                    status = excluded.status,
                    pid = excluded.pid
                """,
                (status, pid),
            )
            conn.commit()

    def log_heartbeat_summary(self) -> None:
        """Log structured heartbeat metrics."""
        with self._open_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as total, SUM(CASE WHEN consecutive_failures == 0 THEN 1 ELSE 0 END) as healthy FROM source_state")
            src_row = c.fetchone()
            total_src = src_row["total"] or 0
            healthy_src = src_row["healthy"] or 0

            c.execute("SELECT COUNT(*) as pending FROM article_pipeline_state WHERE stage IN ('NEW', 'ENRICHED', 'ANALYSIS_PENDING')")
            pending_count = c.fetchone()["pending"] or 0

            c.execute("SELECT COUNT(*) as stored FROM article_pipeline_state WHERE stage = 'STORED'")
            stored_count = c.fetchone()["stored"] or 0

            c.execute("SELECT COUNT(*) as completed FROM article_analysis WHERE status = 'completed'")
            completed_analyses = c.fetchone()["completed"] or 0

            c.execute("SELECT MIN(strftime('%s', next_due_at) - strftime('%s', 'now')) as next_sec FROM source_state")
            next_row = c.fetchone()
            next_due_sec = max(0, int(next_row["next_sec"])) if next_row and next_row["next_sec"] is not None else 0

        logger.info(
            "[HEARTBEAT] Monitor alive | Sources: %d/%d healthy | Pending queue: %d (Stored arXiv: %d) | Analyzed: %d | Next poll in %ds",
            healthy_src,
            total_src,
            pending_count,
            stored_count,
            completed_analyses,
            next_due_sec,
        )

    def sleep_until_next_due_source(self) -> None:
        """Sleep until the next scheduled source is due, waking periodically for heartbeats."""
        with self._open_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT MIN(strftime('%s', next_due_at) - strftime('%s', 'now')) as next_sec FROM source_state")
            row = c.fetchone()
            seconds_to_wait = max(1, int(row["next_sec"])) if row and row["next_sec"] is not None else 60

        # Bound single sleep period by heartbeat interval
        sleep_chunk = min(seconds_to_wait, self.heartbeat_interval)
        logger.debug("[MONITOR] Sleeping for up to %ds (next source in %ds)...", sleep_chunk, seconds_to_wait)

        # Sleep in 1s increments so stop_event is responsive
        start = time.time()
        while time.time() - start < sleep_chunk:
            if self.stop_event.is_set():
                break
            time.sleep(1.0)

    def step(self, force_poll: bool = False) -> Dict[str, int]:
        """Execute one complete monitoring cycle."""
        inserted = self.poll_due_sources(force_all=force_poll)
        promoted = self.select_and_promote_arxiv_articles()
        analyzed = self.process_pending_articles()
        notified = self.send_notifications()

        now = time.time()
        if now - self._last_heartbeat_time >= self.heartbeat_interval:
            self.update_heartbeat("running")
            self.log_heartbeat_summary()
            self._last_heartbeat_time = now

        return {
            "inserted": inserted,
            "promoted": promoted,
            "analyzed": analyzed,
            "notified": notified,
        }

    def run_continuous(self) -> None:
        """Main monitoring loop. Runs until interrupted."""
        self.running = True
        logger.info("[MONITOR] Starting continuous intelligence monitor (PID: %d)", os.getpid())
        self.update_heartbeat("running")
        self.log_heartbeat_summary()
        self._last_heartbeat_time = time.time()

        try:
            while not self.stop_event.is_set():
                self.step()
                if not self.stop_event.is_set():
                    self.sleep_until_next_due_source()
        finally:
            logger.info("[MONITOR] Shutdown signal received. Exiting cleanly...")
            self.update_heartbeat("stopped")
            self.running = False

    def stop(self) -> None:
        """Signal the monitor to shut down gracefully."""
        self.stop_event.set()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous AI News Intelligence Monitor")
    parser.add_argument("--once", action="store_true", help="Run a single monitor cycle and exit.")
    parser.add_argument("--poll-all", action="store_true", help="Force polling of all sources immediately.")
    parser.add_argument("--batch-size", type=int, default=MONITOR_BATCH_SIZE, help="Batch size for pending analysis.")
    parser.add_argument("--max-retries", type=int, default=LLM_MAX_RETRIES, help="Max LLM retry attempts.")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path override.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    monitor = ContinuousMonitor(
        db_path=args.db_path,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )

    def handle_signal(sig, frame):
        logger.info("[MONITOR] Caught signal %s. Initiating graceful shutdown...", sig)
        monitor.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.once:
        logger.info("[MONITOR] Executing single cycle (--once)...")
        results = monitor.step(force_poll=args.poll_all)
        logger.info("[MONITOR] Cycle complete: %s", results)
        monitor.update_heartbeat("stopped")
        return 0

    monitor.run_continuous()
    return 0


if __name__ == "__main__":
    sys.exit(main())
