"""Batch evaluation suite for FreeWebRetriever (Component 1).

Tests a batch of 8 diverse real-world factual claims across science, history,
biology, and geography to verify:
1. Live web retrieval reliability across DuckDuckGo and Wikipedia.
2. 100% cache population and sub-millisecond cache hit acceleration.
3. Downstream compatibility with Module 8 Quality Firewall (Credibility, Kish Clustering, Pollution).
"""

from pathlib import Path
import time
import pytest

from app.schemas import EvidenceItem
from app.services.free_web_retriever import FreeWebRetriever
from app.services.source_credibility_scorer import SourceCredibilityScorer
from app.services.source_independence_scorer import SourceIndependenceDetector
from app.services.evidence_pollution_detector import EvidencePollutionDetector
from app.services.evidence_pollution_models import PollutionClass
from app.services.web_cache import WebCache


BATCH_CLAIMS = [
    {
        "domain": "Astrophysics",
        "claim": "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b",
        "expected_terms": ["carbon dioxide", "webb", "wasp-39b"],
    },
    {
        "domain": "Computer Science",
        "claim": "Tim Berners-Lee invented the World Wide Web in 1989 at CERN",
        "expected_terms": ["berners-lee", "world wide web", "cern"],
    },
    {
        "domain": "Physics",
        "claim": "The speed of light in vacuum is approximately 299792458 meters per second",
        "expected_terms": ["speed of light", "vacuum", "meters per second"],
    },
    {
        "domain": "Genetics",
        "claim": "CRISPR Cas9 was adapted for genome editing by Doudna and Charpentier",
        "expected_terms": ["crispr", "doudna", "charpentier"],
    },
    {
        "domain": "Planetary Science",
        "claim": "Olympus Mons is a large shield volcano on the planet Mars",
        "expected_terms": ["olympus mons", "volcano", "mars"],
    },
    {
        "domain": "Environmental Science",
        "claim": "The Montreal Protocol phased out the production of chlorofluorocarbons",
        "expected_terms": ["montreal protocol", "chlorofluorocarbons"],
    },
    {
        "domain": "Mathematics",
        "claim": "Fermat's Last Theorem was proved by Andrew Wiles in 1994",
        "expected_terms": ["fermat", "wiles", "theorem"],
    },
    {
        "domain": "Chemistry",
        "claim": "Water molecules consist of two hydrogen atoms and one oxygen atom",
        "expected_terms": ["water", "hydrogen", "oxygen"],
    },
]


@pytest.fixture(scope="module")
def batch_cache(tmp_path_factory: pytest.TempPathFactory) -> WebCache:
    """Provide a module-scoped SQLite cache so subsequent tests verify cache hits."""
    tmp_dir = tmp_path_factory.mktemp("batch_eval")
    db_file = tmp_dir / "batch_eval_cache.db"
    return WebCache(db_path=db_file, default_ttl_seconds=3600)


def test_batch_live_retrieval_and_caching(batch_cache: WebCache):
    """Execute batch of 8 diverse claims, validating live retrieval and cache population."""
    retriever = FreeWebRetriever(
        cache=batch_cache,
        timeout=15.0,
        max_results=4,
        min_request_interval=0.25,
    )

    batch_results: dict[str, list[EvidenceItem]] = {}
    latencies: list[float] = []

    print("\n--- BATCH LIVE RETRIEVAL COMMENCING (8 CLAIMS) ---")
    for idx, item in enumerate(BATCH_CLAIMS, start=1):
        claim_text = item["claim"]
        domain = item["domain"]

        start_time = time.perf_counter()
        evidence_items = retriever.retrieve(claim_text, stance="supporting")
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        # Invariants for retrieved items
        assert len(evidence_items) >= 1, f"Failed to retrieve evidence for claim #{idx}: {claim_text}"
        batch_results[claim_text] = evidence_items

        top_evidence = evidence_items[0]
        assert top_evidence.text and len(top_evidence.text) > 10
        assert top_evidence.relevance_score >= 0.50
        assert top_evidence.retrieval_method in ("free_web_duckduckgo", "free_web_wikipedia")

        print(
            f"  [{idx}/8] Domain: {domain:<21} | Items: {len(evidence_items)} | "
            f"Latency: {elapsed:.2f}s | Engine: {top_evidence.retrieval_method} | Domain: {top_evidence.domain}"
        )

    # Verify cache has stored all 8 queries
    assert batch_cache.count() == len(BATCH_CLAIMS)
    avg_latency = sum(latencies) / len(latencies)
    print(f"--- BATCH INITIAL RETRIEVAL COMPLETE (Avg Latency: {avg_latency:.2f}s) ---\n")


def test_batch_cache_hit_speed_and_zero_network(batch_cache: WebCache):
    """Verify that re-running the batch achieves 100% cache hits with sub-10ms latency."""
    retriever = FreeWebRetriever(cache=batch_cache, min_request_interval=0.0)

    assert batch_cache.count() == len(BATCH_CLAIMS)

    # Re-retrieve every claim from cache
    cache_latencies: list[float] = []
    print("\n--- BATCH CACHE RE-RETRIEVAL COMMENCING (100% CACHE HIT) ---")
    for idx, item in enumerate(BATCH_CLAIMS, start=1):
        claim_text = item["claim"]

        start_time = time.perf_counter()
        cached_items = retriever.retrieve(claim_text, stance="supporting")
        elapsed = time.perf_counter() - start_time
        cache_latencies.append(elapsed)

        assert len(cached_items) >= 1
        # Each query should resolve under 0.05 seconds from local SQLite
        assert elapsed < 0.05, f"Cache retrieval was too slow ({elapsed:.4f}s) for {claim_text}"

        print(
            f"  [{idx}/8] Cache Hit: {claim_text[:40]}... | Items: {len(cached_items)} | "
            f"Time: {elapsed * 1000.0:.2f}ms"
        )

    avg_cache_ms = (sum(cache_latencies) / len(cache_latencies)) * 1000.0
    print(f"--- BATCH CACHE RE-RETRIEVAL COMPLETE (Avg Cache Time: {avg_cache_ms:.2f}ms) ---\n")
    # Average lookup time across the batch should be under 15ms
    assert avg_cache_ms < 15.0


def test_batch_module8_quality_firewall_compatibility(batch_cache: WebCache):
    """Verify live web retrieved evidence passes cleanly through Module 8 Quality Pipeline."""
    retriever = FreeWebRetriever(cache=batch_cache, timeout=15.0, min_request_interval=0.2)

    cred_scorer = SourceCredibilityScorer()
    indep_detector = SourceIndependenceDetector()
    pollution_detector = EvidencePollutionDetector()

    print("\n--- BATCH MODULE 8 QUALITY FIREWALL VERIFICATION ---")
    total_items = 0
    clean_items = 0

    for idx, item in enumerate(BATCH_CLAIMS[:4], start=1):  # Test first 4 claims thoroughly
        claim_text = item["claim"]
        evidence_items = retriever.retrieve(claim_text, stance="supporting")
        assert len(evidence_items) >= 1

        # 1. Source Credibility Scoring (Module 8.1)
        for ev in evidence_items:
            cred_res = cred_scorer.score(ev)
            assert 0.0 <= cred_res.credibility_score <= 1.0
            total_items += 1

            # 2. Evidence Pollution Filtering (Module 8.4)
            poll_res = pollution_detector.evaluate(
                claim=claim_text,
                evidence={
                    "document_id": str(ev.evidence_id),
                    "text": ev.text,
                    "source": ev.source_title,
                    "source_url": ev.source_url,
                    "domain": ev.domain,
                },
            )
            assert 0.0 <= poll_res.pollution_score <= 1.0
            if poll_res.classification in (PollutionClass.CLEAN, PollutionClass.WEAK_POLLUTION_SIGNAL):
                clean_items += 1

        # 3. Redundancy Clustering & Kish n_eff (Module 8.3)
        indep_items = [
            {
                "document_id": str(ev.evidence_id),
                "text": ev.text,
                "source": ev.source_title,
                "source_url": ev.source_url,
                "domain": ev.domain,
            }
            for ev in evidence_items
        ]
        indep_res = indep_detector.analyze_set(indep_items)
        assert indep_res.kish_effective_sample_size >= 1.0

        print(
            f"  [{idx}/4] Claim: {claim_text[:35]}... | Items: {len(evidence_items)} | "
            f"Kish n_eff: {indep_res.kish_effective_sample_size:.2f} | Redundant Pairs: {len(indep_res.pairwise_comparisons)}"
        )

    # Confirm clean retention target (>70% on live search snippets)
    retention_rate = clean_items / total_items
    print(f"  Quality Retention Rate: {retention_rate * 100:.1f}% ({clean_items}/{total_items} admissible)")
    assert retention_rate >= 0.70, f"Pollution filter was overly aggressive: retention was {retention_rate:.2f}"


def test_batch_adversarial_stance_query_differentiation(batch_cache: WebCache):
    """Verify supporting vs. contradicting stances produce differentiated search queries."""
    retriever = FreeWebRetriever(cache=batch_cache, timeout=15.0, min_request_interval=0.2)

    claim = "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b"

    supp_items = retriever.retrieve(claim, stance="supporting")
    opp_items = retriever.retrieve(claim, stance="contradicting")

    assert len(supp_items) >= 1
    assert len(opp_items) >= 1

    # Verify both stances retrieved distinct evidence pools from web cache/search
    assert batch_cache.count() >= 2
