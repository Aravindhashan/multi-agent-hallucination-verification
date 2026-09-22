"""Comprehensive tests for Source Credibility Scorer: properties, determinism, snapshots, performance."""

from datetime import date
import json
import random
import time
import pytest

from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult
from app.services.source_credibility_config import (
    DISCLAIMER_PHRASE,
    SourceCredibilityInputError,
)
from app.services.source_credibility_scorer import (
    SourceCredibilityScorer,
    get_source_credibility_scorer,
    reset_source_credibility_scorer,
)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Ensure singleton instance is cleared before and after each test."""
    reset_source_credibility_scorer()
    yield
    reset_source_credibility_scorer()


@pytest.fixture
def scorer() -> SourceCredibilityScorer:
    return SourceCredibilityScorer()


# ---------------------------------------------------------------------------
# Section 13 Edge Cases Decision Table Tests
# ---------------------------------------------------------------------------

def test_edge_case_missing_url(scorer: SourceCredibilityScorer):
    doc = {"document_id": "e1", "title": "Test", "source": "Org"}
    res = scorer.score(doc)
    assert res.normalized_domain == "unknown"
    assert res.domain_tier == 4
    assert 0.0 <= res.credibility_score <= 1.0


def test_edge_case_malformed_url(scorer: SourceCredibilityScorer):
    doc = {"document_id": "e2", "source_url": "http://"}
    res = scorer.score(doc)
    assert res.normalized_domain == "unknown"
    assert res.domain_tier == 4


def test_edge_case_empty_metadata(scorer: SourceCredibilityScorer):
    res = scorer.score({})
    assert res.normalized_domain == "unknown"
    assert res.domain_tier == 4
    assert res.provenance_score == 0.0
    assert 0.0 <= res.credibility_score <= 1.0


def test_edge_case_example_domain(scorer: SourceCredibilityScorer):
    doc = {"document_id": "e4", "source_url": "https://data.example/resource"}
    res = scorer.score(doc)
    assert res.normalized_domain == "data.example"
    assert res.domain_tier == 4


def test_edge_case_test_and_invalid_domains(scorer: SourceCredibilityScorer):
    res1 = scorer.score({"document_id": "e5a", "source_url": "https://gov.test"})
    res2 = scorer.score({"document_id": "e5b", "source_url": "https://agency.invalid"})
    assert res1.domain_tier == 4
    assert res2.domain_tier == 4


def test_edge_case_localhost(scorer: SourceCredibilityScorer):
    res = scorer.score({"document_id": "e6", "source_url": "http://localhost:8080/doc"})
    assert res.normalized_domain == "localhost"
    assert res.domain_tier == 4


def test_edge_case_ipv4_and_ipv6(scorer: SourceCredibilityScorer):
    res_v4 = scorer.score({"document_id": "e7a", "source_url": "http://192.168.1.1/info"})
    res_v6 = scorer.score({"document_id": "e7b", "source_url": "http://[2001:db8::1]/info"})
    assert res_v4.domain_tier == 4
    assert res_v6.domain_tier == 4


def test_edge_case_port_present(scorer: SourceCredibilityScorer):
    res = scorer.score({"document_id": "e8", "source_url": "https://nih.gov:443/data"})
    assert res.normalized_domain == "nih.gov"
    assert res.domain_tier == 1


def test_edge_case_www_prefix(scorer: SourceCredibilityScorer):
    res = scorer.score({"document_id": "e9", "source_url": "https://www.nature.com/articles"})
    assert res.normalized_domain == "nature.com"
    assert res.domain_tier == 2


def test_edge_case_uppercase_host(scorer: SourceCredibilityScorer):
    res = scorer.score({"document_id": "e10", "source_url": "HTTPS://WWW.NASA.GOV/PRESS"})
    assert res.normalized_domain == "nasa.gov"
    assert res.domain_tier == 1


# ---------------------------------------------------------------------------
# CorpusDocument & RetrievalResult Input Compatibility
# ---------------------------------------------------------------------------

def test_score_corpus_document_object(scorer: SourceCredibilityScorer):
    corpus_doc = CorpusDocument(
        id="corpus_001",
        title="Speed of Light",
        text="The speed of light in vacuum is 299,792,458 m/s.",
        source="Physics Standards",
        source_url="https://physics.nist.gov/constants",
        domain="physics.nist.gov",
        published_date=date(2022, 5, 10),
    )
    res = scorer.score(corpus_doc)
    assert res.document_id == "corpus_001"
    assert res.normalized_domain == "physics.nist.gov"
    assert res.domain_tier == 1
    assert res.domain_score == 1.0
    assert res.provenance_score == 1.0
    assert res.credibility_score == 1.0


def test_score_retrieval_result_object(scorer: SourceCredibilityScorer):
    retrieval_res = RetrievalResult(
        document_id="ret_001",
        title="General Article",
        text="Sample retrieved passage text.",
        source="The Guardian",
        source_url="https://www.theguardian.com/world/article",
        domain="theguardian.com",
        published_date=date(2023, 8, 20),
        score=0.85,
        retrieval_method="HYBRID",
        rank=1,
    )
    res = scorer.score(retrieval_res)
    assert res.document_id == "ret_001"
    assert res.normalized_domain == "theguardian.com"
    assert res.domain_tier == 3
    assert res.domain_score == 0.50
    assert res.provenance_score == 1.0


def test_score_unsupported_type_raises(scorer: SourceCredibilityScorer):
    with pytest.raises(SourceCredibilityInputError):
        scorer.score("not a valid document object or mapping")


# ---------------------------------------------------------------------------
# Batch Scoring & Order Preservation
# ---------------------------------------------------------------------------

def test_score_batch_preserves_order(scorer: SourceCredibilityScorer):
    docs = [
        {"document_id": "b1", "source_url": "https://nih.gov"},
        {"document_id": "b2", "source_url": "https://nature.com"},
        {"document_id": "b3", "source_url": "https://forbes.com"},
        {"document_id": "b4", "source_url": "https://example.com"},
    ]
    results = scorer.score_batch(docs)
    assert len(results) == 4
    assert [r.document_id for r in results] == ["b1", "b2", "b3", "b4"]
    assert [r.domain_tier for r in results] == [1, 2, 3, 4]


def test_score_batch_empty(scorer: SourceCredibilityScorer):
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# Property-Based Bounds & Invariants (100 Random Combos)
# ---------------------------------------------------------------------------

def test_property_score_bounds_and_finite(scorer: SourceCredibilityScorer):
    """Property test: 100 random document metadata combinations strictly satisfy all invariants."""
    random.seed(42)
    sample_urls = [
        "https://nih.gov/a",
        "https://nature.com/b",
        "https://nytimes.com/c",
        "http://unknown-domain.xyz",
        "https://test.example",
        None,
        "",
        "http://10.0.0.1",
    ]
    sample_types = ["academic", "news", "commercial", "unknown", None, ""]
    titles = ["Valid Title", None, "", "   "]
    sources = ["Valid Source", None, "", "   "]

    for i in range(100):
        doc = {
            "document_id": f"rand_{i}",
            "source_url": random.choice(sample_urls),
            "source_type": random.choice(sample_types),
            "title": random.choice(titles),
            "source": random.choice(sources),
            "published_date": date(2020, 1, 1) if random.random() > 0.5 else None,
        }
        res = scorer.score(doc)
        assert 0.0 <= res.credibility_score <= 1.0
        assert 0.0 <= res.domain_score <= 1.0
        assert 0.0 <= res.provenance_score <= 1.0
        assert res.domain_tier in {1, 2, 3, 4}
        assert abs(sum(res.weights.values()) - 1.0) < 1e-4
        if res.source_type_score is not None:
            assert 0.0 <= res.source_type_score <= 1.0


# ---------------------------------------------------------------------------
# Determinism & Byte-Exact Reproducibility
# ---------------------------------------------------------------------------

def test_scoring_determinism():
    doc = {
        "document_id": "det_01",
        "title": "Deterministic Test",
        "source": "Reuters News",
        "source_url": "https://www.reuters.com/world/article",
        "domain": "reuters.com",
        "published_date": date(2023, 10, 1),
    }
    scorer1 = SourceCredibilityScorer()
    scorer2 = SourceCredibilityScorer()

    res1 = scorer1.score(doc)
    res2 = scorer1.score(doc)
    res3 = scorer2.score(doc)

    assert res1.credibility_score == res2.credibility_score == res3.credibility_score
    assert res1.explanation == res2.explanation == res3.explanation
    assert res1.model_dump() == res2.model_dump() == res3.model_dump()


# ---------------------------------------------------------------------------
# Structured Output & JSON Serialization Round-Trip
# ---------------------------------------------------------------------------

def test_json_round_trip(scorer: SourceCredibilityScorer):
    doc = {
        "document_id": "json_01",
        "title": "JSON Roundtrip Doc",
        "source": "Wikipedia",
        "source_url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "domain": "wikipedia.org",
        "published_date": date(2024, 1, 1),
    }
    res = scorer.score(doc)
    serialized = json.dumps(res.model_dump(mode="json"))
    deserialized = json.loads(serialized)

    assert deserialized["document_id"] == "json_01"
    assert deserialized["domain_tier"] == 2
    assert "explanation" in deserialized
    assert len(deserialized["factors"]) >= 2


def test_explanation_contains_disclaimer_verbatim(scorer: SourceCredibilityScorer):
    res = scorer.score({"document_id": "disc_01", "source_url": "https://nih.gov"})
    assert DISCLAIMER_PHRASE in res.explanation
    assert "Domain tier is a provenance heuristic, not proof of factual correctness." in res.explanation


# ---------------------------------------------------------------------------
# Snapshot Testing for 5 Representative Documents
# ---------------------------------------------------------------------------

SNAPSHOTS = [
    {
        "doc": {
            "document_id": "snap_tier1",
            "title": "NIH Clinical Guide",
            "source": "NIH",
            "source_url": "https://www.nih.gov/health-information",
            "domain": "nih.gov",
            "published_date": date(2023, 6, 1),
        },
        "expected_domain": "nih.gov",
        "expected_tier": 1,
        "expected_cred": 1.0,
    },
    {
        "doc": {
            "document_id": "snap_tier2",
            "title": "Nature Physics Article",
            "source": "Nature",
            "source_url": "https://nature.com/articles/s41567-023-02100-x",
            "domain": "nature.com",
            "published_date": date(2023, 7, 15),
        },
        "expected_domain": "nature.com",
        "expected_tier": 2,
        "expected_cred": 0.8438,
    },
    {
        "doc": {
            "document_id": "snap_tier3",
            "title": "TechCrunch Review",
            "source": "TechCrunch",
            "source_url": "https://techcrunch.com/2023/11/01/ai-breakthrough",
            "domain": "techcrunch.com",
            "published_date": date(2023, 11, 1),
        },
        "expected_domain": "techcrunch.com",
        "expected_tier": 3,
        "expected_cred": 0.6875,
    },
    {
        "doc": {
            "document_id": "snap_tier4_example",
            "title": "Synthetic Corpus Doc",
            "source": "Local Org",
            "source_url": "https://data.example/article",
            "domain": "data.example",
            "published_date": date(2022, 1, 1),
        },
        "expected_domain": "data.example",
        "expected_tier": 4,
        "expected_cred": 0.5312,
    },
    {
        "doc": {
            "document_id": "snap_empty",
        },
        "expected_domain": "unknown",
        "expected_tier": 4,
        "expected_cred": 0.2188,
    },
]


def test_snapshots_stability(scorer: SourceCredibilityScorer):
    """Verify that scores and tier classifications remain rock-solid across runs."""
    for snap in SNAPSHOTS:
        res = scorer.score(snap["doc"])
        assert res.normalized_domain == snap["expected_domain"]
        assert res.domain_tier == snap["expected_tier"]
        assert abs(res.credibility_score - snap["expected_cred"]) < 1e-3


# ---------------------------------------------------------------------------
# Performance Benchmark: score() < 1 ms warm
# ---------------------------------------------------------------------------

def test_performance_warm_score_under_one_ms(scorer: SourceCredibilityScorer):
    doc = {
        "document_id": "perf_01",
        "title": "High Speed Evaluation",
        "source": "Fast Source",
        "source_url": "https://nature.com/article",
        "domain": "nature.com",
        "published_date": date(2023, 5, 20),
    }
    # Warmup
    for _ in range(50):
        _ = scorer.score(doc)

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scorer.score(doc)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) * 1000.0) / iterations
    assert avg_ms < 1.0, f"Average score() latency ({avg_ms:.3f} ms) exceeds 1.0 ms threshold."


# ---------------------------------------------------------------------------
# Singleton Lifecycle
# ---------------------------------------------------------------------------

def test_singleton_accessor():
    s1 = get_source_credibility_scorer()
    s2 = get_source_credibility_scorer()
    assert s1 is s2

    reset_source_credibility_scorer()
    s3 = get_source_credibility_scorer()
    assert s1 is not s3
