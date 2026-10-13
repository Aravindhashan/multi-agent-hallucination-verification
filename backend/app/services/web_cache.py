"""Thread-safe SQLite persistent query cache for web retrieval."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from app.config import settings

logger = logging.getLogger("hallucination_verifier.web_cache")


class WebCache:
    """Persistent, thread-safe SQLite key-value cache for web search queries.

    Stores structured evidence candidate payloads indexed by query hash and search
    engine type to avoid redundant network scraping, respect rate limits, and
    ensure sub-millisecond retrieval on repeated debate turns.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        default_ttl_seconds: float | None = None,
    ) -> None:
        """Initialize SQLite database, table schema, and thread lock."""
        resolved_path = (
            Path(db_path) if db_path is not None else Path(settings.WEB_CACHE_DB_PATH)
        )
        self.db_path = resolved_path
        self.default_ttl = (
            float(default_ttl_seconds)
            if default_ttl_seconds is not None
            else float(settings.WEB_CACHE_TTL_SECONDS)
        )
        self._lock = threading.Lock()

        # Ensure directory exists if not an in-memory DB
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection configured with WAL and busy timeout."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # WAL mode offers higher concurrency for parallel readers/writers
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create table and indexes if they do not exist."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS web_cache (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        engine TEXT NOT NULL,
                        results_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        ttl_seconds REAL NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_web_cache_created
                    ON web_cache(created_at);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_web_cache_engine
                    ON web_cache(engine);
                    """
                )
                conn.commit()

    @staticmethod
    def compute_query_hash(query: str, engine: str) -> str:
        """Generate deterministic SHA256 key from normalized query text and engine."""
        normalized_q = " ".join(query.strip().lower().split())
        normalized_eng = engine.strip().lower()
        payload = f"{normalized_eng}::{normalized_q}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, engine: str = "free_web") -> list[dict[str, Any]] | None:
        """Retrieve unexpired cached results for a query and engine, or None."""
        if not query or not query.strip():
            return None

        q_hash = self.compute_query_hash(query, engine)
        now = time.time()

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT results_json, created_at, ttl_seconds
                    FROM web_cache
                    WHERE query_hash = ?;
                    """,
                    (q_hash,),
                ).fetchone()

                if not row:
                    return None

                results_json = row["results_json"]
                created_at = row["created_at"]
                ttl = row["ttl_seconds"]

                # Expiration check
                if now - created_at > ttl:
                    # Clean up expired entry asynchronously/lazily
                    conn.execute(
                        "DELETE FROM web_cache WHERE query_hash = ?;",
                        (q_hash,),
                    )
                    conn.commit()
                    logger.debug("Cache expired for query: '%s' (engine=%s)", query, engine)
                    return None

                try:
                    data = json.loads(results_json)
                    logger.debug("Cache hit for query: '%s' (engine=%s)", query, engine)
                    return data
                except json.JSONDecodeError as exc:
                    logger.warning("Corrupted cache payload for %s: %s", q_hash, exc)
                    return None

    def set(
        self,
        query: str,
        engine: str,
        results: list[dict[str, Any]],
        ttl_seconds: float | None = None,
    ) -> None:
        """Store results for a query and engine with specified or default TTL."""
        if not query or not query.strip():
            return

        q_hash = self.compute_query_hash(query, engine)
        now = time.time()
        ttl = float(ttl_seconds) if ttl_seconds is not None else self.default_ttl
        results_json = json.dumps(results, ensure_ascii=False)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO web_cache (query_hash, query, engine, results_json, created_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_hash) DO UPDATE SET
                        results_json=excluded.results_json,
                        created_at=excluded.created_at,
                        ttl_seconds=excluded.ttl_seconds;
                    """,
                    (q_hash, query.strip(), engine.strip().lower(), results_json, now, ttl),
                )
                conn.commit()
                logger.debug("Cached %d results for query: '%s' (engine=%s)", len(results), query, engine)

    def prune_expired(self) -> int:
        """Delete all expired records from cache and return count of deleted items."""
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM web_cache WHERE (? - created_at) > ttl_seconds;",
                    (now,),
                )
                deleted_count = cursor.rowcount
                conn.commit()
                if deleted_count > 0:
                    logger.info("Pruned %d expired entries from web cache.", deleted_count)
                return deleted_count

    def clear(self) -> None:
        """Delete all records from web cache."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM web_cache;")
                conn.commit()
                logger.info("Cleared all web cache records.")

    def count(self, engine: str | None = None) -> int:
        """Return count of unexpired cached queries."""
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                if engine is not None:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) as cnt FROM web_cache
                        WHERE engine = ? AND (? - created_at) <= ttl_seconds;
                        """,
                        (engine.strip().lower(), now),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) as cnt FROM web_cache
                        WHERE (? - created_at) <= ttl_seconds;
                        """,
                        (now,),
                    ).fetchone()
                return int(row["cnt"]) if row else 0
