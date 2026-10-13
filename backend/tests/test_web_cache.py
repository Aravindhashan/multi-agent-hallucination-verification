"""Unit tests for the SQLite WebCache service."""

import concurrent.futures
import time
from pathlib import Path
import pytest

from app.services.web_cache import WebCache


def test_web_cache_basic_set_get(tmp_path: Path):
    """Verify storing and retrieving query results from SQLite cache."""
    db_file = tmp_path / "test_cache.db"
    cache = WebCache(db_path=db_file, default_ttl_seconds=3600)

    query = "James Webb Space Telescope exoplanet"
    results = [
        {"title": "JWST WASP-39b", "url": "https://nasa.gov/jwst", "snippet": "CO2 detected"}
    ]

    # Initial get should be a miss
    assert cache.get(query, engine="free_web") is None
    assert cache.count() == 0

    # Store
    cache.set(query, engine="free_web", results=results)
    assert cache.count() == 1

    # Retrieve
    cached = cache.get(query, engine="free_web")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "JWST WASP-39b"
    assert cached[0]["snippet"] == "CO2 detected"


def test_web_cache_query_normalization(tmp_path: Path):
    """Verify that case and redundant whitespace do not cause cache misses."""
    db_file = tmp_path / "norm_cache.db"
    cache = WebCache(db_path=db_file)

    cache.set("  Quantum  Computing  Qubits  ", engine="duckduckgo", results=[{"id": 1}])

    # Same query with different casing and spacing should hit
    hit = cache.get("quantum computing qubits", engine="duckduckgo")
    assert hit == [{"id": 1}]

    # Different engine should miss
    assert cache.get("quantum computing qubits", engine="wikipedia") is None


def test_web_cache_ttl_expiration(tmp_path: Path):
    """Verify that expired items are treated as misses and cleaned up."""
    db_file = tmp_path / "ttl_cache.db"
    # Set short TTL of 0.2 seconds
    cache = WebCache(db_path=db_file, default_ttl_seconds=0.2)

    cache.set("temporary claim", engine="test", results=[{"status": "fresh"}], ttl_seconds=0.15)
    assert cache.get("temporary claim", engine="test") is not None

    # Wait for expiration
    time.sleep(0.25)

    # Retrieval should detect expiration, prune it, and return None
    assert cache.get("temporary claim", engine="test") is None
    assert cache.count(engine="test") == 0


def test_web_cache_prune_expired(tmp_path: Path):
    """Verify explicit pruning of multiple expired records."""
    db_file = tmp_path / "prune_cache.db"
    cache = WebCache(db_path=db_file)

    # Add 1 active entry (1 hour TTL) and 2 entries with past expiration
    cache.set("active query", engine="test", results=[{"val": 1}], ttl_seconds=3600)
    cache.set("expired query 1", engine="test", results=[{"val": 2}], ttl_seconds=0.05)
    cache.set("expired query 2", engine="test", results=[{"val": 3}], ttl_seconds=0.05)

    time.sleep(0.1)
    deleted = cache.prune_expired()
    assert deleted == 2
    assert cache.count() == 1
    assert cache.get("active query", engine="test") is not None


def test_web_cache_update_overwrite(tmp_path: Path):
    """Verify updating an existing cached query overwrites with newer data."""
    db_file = tmp_path / "overwrite_cache.db"
    cache = WebCache(db_path=db_file)

    cache.set("claim 1", engine="test", results=[{"version": 1}])
    assert cache.get("claim 1", engine="test")[0]["version"] == 1

    cache.set("claim 1", engine="test", results=[{"version": 2}])
    assert cache.get("claim 1", engine="test")[0]["version"] == 2
    assert cache.count() == 1


def test_web_cache_clear(tmp_path: Path):
    """Verify clearing entire cache."""
    db_file = tmp_path / "clear_cache.db"
    cache = WebCache(db_path=db_file)

    cache.set("q1", engine="e1", results=[{"a": 1}])
    cache.set("q2", engine="e2", results=[{"b": 2}])
    assert cache.count() == 2

    cache.clear()
    assert cache.count() == 0
    assert cache.get("q1", engine="e1") is None


def test_web_cache_thread_concurrency(tmp_path: Path):
    """Verify thread-safety across concurrent workers reading and writing."""
    db_file = tmp_path / "concurrent_cache.db"
    cache = WebCache(db_path=db_file)

    def worker(worker_id: int):
        q = f"query number {worker_id % 5}"
        cache.set(q, engine="test", results=[{"worker": worker_id}])
        res = cache.get(q, engine="test")
        return res is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(40)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results)
    assert cache.count() <= 5
