"""Thread-safe SQLite-backed session and state management for the verification pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from uuid import UUID

from app.config import settings


@dataclass
class PipelineSession:
    """Store the state and progression of a verification pipeline run."""

    query_id: UUID
    query: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_text: str | None = None
    model_name: str | None = None
    generated_at: datetime | None = None
    claims: list[Any] = field(default_factory=list)
    extracted_at: datetime | None = None
    complexity_score: float | None = None
    verification_plan: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineSessionStore:
    """Thread-safe SQLite-backed repository for pipeline sessions."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        resolved_path = db_path if db_path is not None else settings.SESSION_DB_PATH

        if str(resolved_path) != ":memory:":
            path_obj = Path(resolved_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(path_obj.resolve())
        else:
            self._db_path = ":memory:"

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create the sessions table and indexes if they do not exist."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        query_id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        response_text TEXT,
                        model_name TEXT,
                        generated_at TEXT,
                        claims_json TEXT,
                        extracted_at TEXT,
                        complexity_score REAL,
                        verification_plan_json TEXT,
                        metadata_json TEXT
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                    ON sessions(created_at DESC)
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_claims (
                        query_id TEXT NOT NULL,
                        claim_id TEXT NOT NULL,
                        claim_text TEXT NOT NULL,
                        complexity_score REAL,
                        verification_plan_json TEXT,
                        requires_deep_verification INTEGER,
                        ablation_log_json TEXT,
                        supporting_evidence_json TEXT,
                        contradicting_evidence_json TEXT,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (query_id, claim_id),
                        FOREIGN KEY (query_id) REFERENCES sessions(query_id) ON DELETE CASCADE
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_claims_query
                    ON session_claims(query_id)
                    """
                )

    def _row_to_session(self, row: sqlite3.Row) -> PipelineSession:
        """Convert a database row into a PipelineSession object."""
        created_at_dt = datetime.fromisoformat(row["created_at"])
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

        gen_at_dt = None
        if row["generated_at"]:
            gen_at_dt = datetime.fromisoformat(row["generated_at"])
            if gen_at_dt.tzinfo is None:
                gen_at_dt = gen_at_dt.replace(tzinfo=timezone.utc)

        ext_at_dt = None
        if row["extracted_at"]:
            ext_at_dt = datetime.fromisoformat(row["extracted_at"])
            if ext_at_dt.tzinfo is None:
                ext_at_dt = ext_at_dt.replace(tzinfo=timezone.utc)

        claims = json.loads(row["claims_json"]) if row["claims_json"] else []
        plan = json.loads(row["verification_plan_json"]) if row["verification_plan_json"] else []
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return PipelineSession(
            query_id=UUID(row["query_id"]),
            query=row["query"],
            created_at=created_at_dt,
            response_text=row["response_text"],
            model_name=row["model_name"],
            generated_at=gen_at_dt,
            claims=claims,
            extracted_at=ext_at_dt,
            complexity_score=row["complexity_score"],
            verification_plan=plan,
            metadata=metadata,
        )

    def get_session(self, query_id: UUID) -> PipelineSession | None:
        """Retrieve a session by its query_id."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions WHERE query_id = ?",
                (str(query_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    def create_session(
        self,
        query_id: UUID,
        query: str,
        created_at: datetime | None = None,
    ) -> PipelineSession:
        """Create and store a new pipeline session."""
        now = created_at if created_at is not None else datetime.now(timezone.utc)
        created_at_str = now.isoformat()

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions (query_id, query, created_at, claims_json, verification_plan_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_id) DO UPDATE SET
                        query = excluded.query,
                        created_at = excluded.created_at
                    """,
                    (str(query_id), query, created_at_str, json.dumps([]), json.dumps([]), json.dumps({})),
                )
            return self.get_session(query_id)  # type: ignore[return-value]

    def record_generation(
        self,
        query_id: UUID,
        query: str,
        response_text: str,
        model_name: str,
        generated_at: datetime | None = None,
    ) -> PipelineSession:
        """Record or update the generated LLM response for a query session."""
        gen_dt = generated_at if generated_at is not None else datetime.now(timezone.utc)
        gen_str = gen_dt.isoformat()

        with self._lock:
            with self._conn:
                # Upsert query and generation
                self._conn.execute(
                    """
                    INSERT INTO sessions (query_id, query, created_at, response_text, model_name, generated_at, claims_json, verification_plan_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_id) DO UPDATE SET
                        response_text = excluded.response_text,
                        model_name = excluded.model_name,
                        generated_at = excluded.generated_at
                    """,
                    (str(query_id), query, gen_str, response_text, model_name, gen_str, json.dumps([]), json.dumps([]), json.dumps({})),
                )
            return self.get_session(query_id)  # type: ignore[return-value]

    def record_claims(
        self,
        query_id: UUID,
        claims: list[Any],
        extracted_at: datetime | None = None,
    ) -> PipelineSession | None:
        """Attach extracted claims to an existing session and populate session_claims rows."""
        ext_dt = extracted_at if extracted_at is not None else datetime.now(timezone.utc)
        ext_str = ext_dt.isoformat()
        claims_json = json.dumps(claims)

        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE sessions
                    SET claims_json = ?, extracted_at = ?
                    WHERE query_id = ?
                    """,
                    (claims_json, ext_str, str(query_id)),
                )
                if cur.rowcount == 0:
                    return None

                # Also insert/upsert into session_claims
                now_str = datetime.now(timezone.utc).isoformat()
                for c in claims:
                    if isinstance(c, dict):
                        cid = str(c.get("claim_id") or c.get("id") or "")
                        ctext = str(c.get("claim_text") or c.get("text") or "")
                    else:
                        cid = str(getattr(c, "claim_id", getattr(c, "id", "")))
                        ctext = str(getattr(c, "claim_text", getattr(c, "text", "")))
                    if cid and ctext:
                        self._conn.execute(
                            """
                            INSERT INTO session_claims (
                                query_id, claim_id, claim_text, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(query_id, claim_id) DO UPDATE SET
                                claim_text = excluded.claim_text,
                                updated_at = excluded.updated_at
                            """,
                            (str(query_id), cid, ctext, now_str, now_str),
                        )
            return self.get_session(query_id)

    def record_analysis(
        self,
        query_id: UUID,
        complexity_score: float,
        verification_plan: list[str],
    ) -> PipelineSession | None:
        """Attach complexity score and verification plan to an existing session."""
        plan_json = json.dumps(verification_plan)

        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE sessions
                    SET complexity_score = ?, verification_plan_json = ?
                    WHERE query_id = ?
                    """,
                    (complexity_score, plan_json, str(query_id)),
                )
                if cur.rowcount == 0:
                    return None
            return self.get_session(query_id)

    def record_claim_analysis(
        self,
        query_id: UUID,
        claim_id: str | UUID,
        claim_text: str,
        complexity_score: float,
        verification_plan: list[str],
        requires_deep_verification: bool | None = None,
        ablation_log: dict[str, Any] | None = None,
    ) -> None:
        """Record or update the analysis and verification plan for a specific claim in the session."""
        now_str = datetime.now(timezone.utc).isoformat()
        plan_json = json.dumps(verification_plan)
        ablation_json = json.dumps(ablation_log) if ablation_log else None
        deep_int = int(requires_deep_verification) if requires_deep_verification is not None else None

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO session_claims (
                        query_id, claim_id, claim_text, complexity_score,
                        verification_plan_json, requires_deep_verification,
                        ablation_log_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_id, claim_id) DO UPDATE SET
                        claim_text = excluded.claim_text,
                        complexity_score = excluded.complexity_score,
                        verification_plan_json = excluded.verification_plan_json,
                        requires_deep_verification = excluded.requires_deep_verification,
                        ablation_log_json = excluded.ablation_log_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(query_id),
                        str(claim_id),
                        claim_text,
                        complexity_score,
                        plan_json,
                        deep_int,
                        ablation_json,
                        now_str,
                        now_str,
                    ),
                )

    def record_claim_evidence(
        self,
        query_id: UUID,
        claim_id: str | UUID,
        evidence_type: str,
        evidence: list[dict[str, Any]],
    ) -> None:
        """Record supporting or contradicting evidence for a specific claim."""
        now_str = datetime.now(timezone.utc).isoformat()
        ev_json = json.dumps(evidence)
        col = (
            "supporting_evidence_json"
            if evidence_type == "supporting"
            else "contradicting_evidence_json"
        )

        with self._lock:
            with self._conn:
                # Ensure row exists
                self._conn.execute(
                    """
                    INSERT INTO session_claims (query_id, claim_id, claim_text, created_at, updated_at)
                    VALUES (?, ?, '', ?, ?)
                    ON CONFLICT(query_id, claim_id) DO NOTHING
                    """,
                    (str(query_id), str(claim_id), now_str, now_str),
                )
                self._conn.execute(
                    f"""
                    UPDATE session_claims
                    SET {col} = ?, updated_at = ?
                    WHERE query_id = ? AND claim_id = ?
                    """,
                    (ev_json, now_str, str(query_id), str(claim_id)),
                )

    def get_claim_records(self, query_id: UUID) -> list[dict[str, Any]]:
        """Retrieve all individual claim records for a given session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM session_claims WHERE query_id = ? ORDER BY created_at ASC",
                (str(query_id),),
            )
            results = []
            for row in cur.fetchall():
                results.append({
                    "query_id": row["query_id"],
                    "claim_id": row["claim_id"],
                    "claim_text": row["claim_text"],
                    "complexity_score": row["complexity_score"],
                    "verification_plan": json.loads(row["verification_plan_json"]) if row["verification_plan_json"] else [],
                    "requires_deep_verification": bool(row["requires_deep_verification"]) if row["requires_deep_verification"] is not None else None,
                    "ablation_log": json.loads(row["ablation_log_json"]) if row["ablation_log_json"] else None,
                    "supporting_evidence": json.loads(row["supporting_evidence_json"]) if row["supporting_evidence_json"] else [],
                    "contradicting_evidence": json.loads(row["contradicting_evidence_json"]) if row["contradicting_evidence_json"] else [],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                })
            return results

    def get_claim_record(self, query_id: UUID, claim_id: str | UUID) -> dict[str, Any] | None:
        """Retrieve a specific claim record within a session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM session_claims WHERE query_id = ? AND claim_id = ?",
                (str(query_id), str(claim_id)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "query_id": row["query_id"],
                "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "complexity_score": row["complexity_score"],
                "verification_plan": json.loads(row["verification_plan_json"]) if row["verification_plan_json"] else [],
                "requires_deep_verification": bool(row["requires_deep_verification"]) if row["requires_deep_verification"] is not None else None,
                "ablation_log": json.loads(row["ablation_log_json"]) if row["ablation_log_json"] else None,
                "supporting_evidence": json.loads(row["supporting_evidence_json"]) if row["supporting_evidence_json"] else [],
                "contradicting_evidence": json.loads(row["contradicting_evidence_json"]) if row["contradicting_evidence_json"] else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def update_metadata(
        self,
        query_id: UUID,
        key: str,
        value: Any,
    ) -> PipelineSession | None:
        """Attach or update a key-value pair in a session's metadata dict."""
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "SELECT metadata_json FROM sessions WHERE query_id = ?",
                    (str(query_id),),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                try:
                    meta = json.loads(row[0]) if row[0] else {}
                except (ValueError, TypeError):
                    meta = {}
                meta[key] = value
                self._conn.execute(
                    "UPDATE sessions SET metadata_json = ? WHERE query_id = ?",
                    (json.dumps(meta), str(query_id)),
                )
            return self.get_session(query_id)

    def list_sessions(self, limit: int = 50) -> list[PipelineSession]:
        """Return a list of recent sessions up to limit."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_session(row) for row in cur.fetchall()]

    def clear(self) -> None:
        """Remove all stored sessions and claim records from the SQLite database."""
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM session_claims")
                self._conn.execute("DELETE FROM sessions")

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM sessions")
            return int(cur.fetchone()[0])


_global_session_store: PipelineSessionStore | None = None


def get_session_store() -> PipelineSessionStore:
    """Return the application singleton session store."""
    global _global_session_store
    if _global_session_store is None:
        _global_session_store = PipelineSessionStore()
    return _global_session_store


def reset_session_store() -> None:
    """Reset the singleton session store (useful for test isolation)."""
    global _global_session_store
    if _global_session_store is not None:
        _global_session_store.clear()
    else:
        _global_session_store = PipelineSessionStore()
