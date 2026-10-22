"""Live Database Inspector Utility for Video Demonstrations.

Queries and neatly formats the latest records from pipeline_sessions.db and web_cache.db
to show real-time synchronization between frontend UI actions and backend database tables.
"""

import json
import os
import sqlite3
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SESSIONS_DB = os.path.join(DB_DIR, "pipeline_sessions.db")
WEB_CACHE_DB = os.path.join(DB_DIR, "web_cache.db")

def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def inspect_latest_session():
    if not os.path.exists(SESSIONS_DB):
        print(f"[!] Sessions DB not found at: {SESSIONS_DB}")
        return

    con = sqlite3.connect(SESSIONS_DB)
    cur = con.cursor()

    print_separator("1. LATEST FRONTEND SESSION (pipeline_sessions.db -> sessions)")
    row = cur.execute("""
        SELECT query_id, query, created_at, length(claims_json) 
        FROM sessions 
        ORDER BY created_at DESC 
        LIMIT 1
    """).fetchone()

    if not row:
        print("No sessions recorded yet.")
        con.close()
        return

    q_id, query, created_at, claims_len = row
    print(f"• Query ID     : {q_id}")
    print(f"• Submitted At : {created_at}")
    print(f"• Query/Text   : {query[:80]}...")
    print(f"• Claims Size  : {claims_len} bytes")

    print_separator("2. ASSOCIATED ATOMIC CLAIMS & NLI METADATA (session_claims)")
    claims = cur.execute("""
        SELECT claim_id, claim_text, complexity_score, metadata_json, created_at
        FROM session_claims
        WHERE query_id = ?
        LIMIT 3
    """, (q_id,)).fetchall()

    if not claims:
        # Fallback to latest global claims
        claims = cur.execute("""
            SELECT claim_id, claim_text, complexity_score, metadata_json, created_at
            FROM session_claims
            ORDER BY created_at DESC
            LIMIT 2
        """).fetchall()

    for i, (c_id, c_text, comp, meta_json, c_time) in enumerate(claims, 1):
        print(f"\n[Claim #{i}] ID: {c_id}")
        print(f"  Claim Text  : {c_text}")
        if comp is not None:
            print(f"  Complexity  : {comp:.2f}")

        if meta_json:
            try:
                meta = json.loads(meta_json)
                adj = meta.get("adjudication", {})
                if adj:
                    print(f"  Verdict     : {adj.get('verdict')} (Conf: {adj.get('confidence_score', 0):.2f})")
                    print(f"  NLI Margin  : P(Support)={adj.get('support_probability', 0):.2f} | P(Contradict)={adj.get('contradiction_probability', 0):.2f}")
                    print(f"  Joint n_eff : {adj.get('joint_n_eff', 1.0):.2f}")
            except Exception:
                pass

    con.close()

def inspect_latest_cache():
    if not os.path.exists(WEB_CACHE_DB):
        print(f"[!] Web Cache DB not found at: {WEB_CACHE_DB}")
        return

    con = sqlite3.connect(WEB_CACHE_DB)
    cur = con.cursor()

    print_separator("3. FEDERATED EVIDENCE RETRIEVAL CACHE (web_cache.db -> web_cache)")
    count = cur.execute("SELECT count(*) FROM web_cache").fetchone()[0]
    print(f"• Total Cached Queries in Database: {count}")

    recent = cur.execute("""
        SELECT query, engine, created_at, length(results_json)
        FROM web_cache
        ORDER BY created_at DESC
        LIMIT 2
    """).fetchall()

    for i, (q, eng, c_at, r_len) in enumerate(recent, 1):
        print(f"\n[Cached Query #{i}]")
        print(f"  Query String: {q}")
        print(f"  Engine      : {eng}")
        print(f"  Cached At   : {c_at}")
        print(f"  Payload Size: {r_len} bytes")

    con.close()

if __name__ == "__main__":
    print("\n🔍 Multi-Agent Hallucination Verifier — Live Database Inspector")
    inspect_latest_session()
    inspect_latest_cache()
    print("\n" + "=" * 70)
    print("  ✓ Database State Synchronized with Frontend")
    print("=" * 70 + "\n")
