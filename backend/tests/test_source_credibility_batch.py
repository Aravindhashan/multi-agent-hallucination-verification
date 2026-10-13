"""Dedicated batch testing and domain authority tier audit for Source Credibility (Submodule 8.1).

Validates:
1. WHO Tier 1 audit (.int suffix inclusion and explicit removal from Tier 2 to prevent downgrade).
2. Domain authority tier separation (Tier 1: 1.00, Tier 2: 0.75, Tier 3: 0.50, Tier 4: 0.25).
3. Memory-safe chunked batch processing with batch_size parameter and input order preservation.
4. Heterogeneous input type support (EvidenceItem, CorpusDocument, RetrievalResult, dict, custom classes).
5. Corrupt, missing, and adversarial metadata resilience.
6. Provenance completeness monotonicity.
7. Dynamic source-type weight normalization across mixed batches.
8. Mutation defenses (invalid batch sizes, non-normalized weights, corrupted tier lookups).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import uuid4
import pytest

from app.schemas import EvidenceItem
from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult
from app.services.source_credibility_config import (
    DISCLAIMER_PHRASE,
    SOURCE_CRED_TIER1_SUFFIXES,
    SOURCE_CRED_TIER2_DOMAINS,
    SOURCE_CRED_TIER3_DOMAINS,
    SOURCE_CRED_TIER_SCORES,
    SourceCredibilityConfigError,
    SourceCredibilityInputError,
    SourceCredibilitySettings,
    SourceCredibilityWeights,
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
    """Fixture providing a clean default SourceCredibilityScorer."""
    return SourceCredibilityScorer()


# ---------------------------------------------------------------------------
# 1. WHO Tier 1 Audit & No Tier 2 Duplicate Test
# ---------------------------------------------------------------------------

def test_who_int_strictly_tier1_audit_and_no_tier2_duplicate(scorer: SourceCredibilityScorer):
    """Verify who.int is deterministically Tier 1, .int is in Tier 1 suffixes, and who.int is not in Tier 2."""
    # Suffix check
    assert ".int" in SOURCE_CRED_TIER1_SUFFIXES, "'.int' must be configured in SOURCE_CRED_TIER1_SUFFIXES."

    # Audit check: who.int must NOT be in Tier 2
    assert "who.int" not in SOURCE_CRED_TIER2_DOMAINS, (
        "who.int must NOT appear in SOURCE_CRED_TIER2_DOMAINS to prevent duplicate downgrade."
    )

    # Architectural docstring audit
    import app.services.source_credibility_config as config_mod
    config_source = open(config_mod.__file__, "r", encoding="utf-8").read()
    assert "who.int" in config_source and "duplicate downgrade" in config_source, (
        "Architectural docstring must document the WHO removal adjustment and duplicate downgrade avoidance."
    )

    # Empirical scoring of WHO URLs and domains
    who_cases = [
        {"document_id": "who_1", "source_url": "https://www.who.int/news/item/01-01-2024"},
        {"document_id": "who_2", "source_url": "https://who.int/publications"},
        {"document_id": "who_3", "domain": "who.int"},
        {"document_id": "who_4", "source_url": "http://data.who.int:80/repo"},
    ]
    results = scorer.score_batch(who_cases)
    for res in results:
        assert res.domain_tier == 1, f"Expected Tier 1 for {res.normalized_domain}, got {res.domain_tier}"
        assert res.domain_score == 1.00
        assert res.domain_tier_label == "High authority"
        assert any("Tier 1 (High authority)" in f.note for f in res.factors if f.name == "domain")
        assert "Domain tier: High authority" in res.explanation

    # Other .int international treaty organizations
    other_int_cases = [
        {"document_id": "int_un", "source_url": "https://un.int/treaties"},
        {"document_id": "int_wipo", "source_url": "https://wipo.int/patents"},
        {"document_id": "int_itu", "source_url": "https://itu.int/standards"},
    ]
    int_results = scorer.score_batch(other_int_cases)
    for res in int_results:
        assert res.domain_tier == 1
        assert res.domain_score == 1.00


# ---------------------------------------------------------------------------
# 2. Tier Hierarchy Separation & Strict Monotonicity
# ---------------------------------------------------------------------------

def test_tier_hierarchy_separation(scorer: SourceCredibilityScorer):
    """Verify distinct score separation and strict ranking across all 4 tiers."""
    tier1_doc = {"document_id": "t1", "source_url": "https://cdc.gov/report", "source_type": "government"}
    tier2_doc = {"document_id": "t2", "source_url": "https://nature.com/articles", "source_type": "academic"}
    tier3_doc = {"document_id": "t3", "source_url": "https://forbes.com/business", "source_type": "news"}
    tier4_doc = {"document_id": "t4", "source_url": "https://unverified-blog.xyz", "source_type": "blog"}

    res1 = scorer.score(tier1_doc)
    res2 = scorer.score(tier2_doc)
    res3 = scorer.score(tier3_doc)
    res4 = scorer.score(tier4_doc)

    assert res1.domain_tier == 1 and res1.domain_score == 1.00
    assert res2.domain_tier == 2 and res2.domain_score == 0.75
    assert res3.domain_tier == 3 and res3.domain_score == 0.50
    assert res4.domain_tier == 4 and res4.domain_score == 0.25

    # Strict monotonicity under identical provenance completeness
    base_prov = {"title": "Title", "source": "Source", "published_date": "2024-01-01"}
    r1 = scorer.score({"document_id": "a1", "source_url": "https://nih.gov", **base_prov})
    r2 = scorer.score({"document_id": "a2", "source_url": "https://bbc.com", **base_prov})
    r3 = scorer.score({"document_id": "a3", "source_url": "https://medium.com", **base_prov})
    r4 = scorer.score({"document_id": "a4", "source_url": "https://random-forum.xyz", **base_prov})

    assert r1.credibility_score > r2.credibility_score > r3.credibility_score > r4.credibility_score


# ---------------------------------------------------------------------------
# 3. Batch Scoring Preserves Order and Counts
# ---------------------------------------------------------------------------

def test_score_batch_preserves_order_and_counts(scorer: SourceCredibilityScorer):
    """Verify batch scoring produces exact 1-to-1 ordered results matching inputs."""
    docs = [
        {"document_id": f"doc_{i:03d}", "source_url": f"https://source{i}.gov"}
        for i in range(15)
    ]
    results = scorer.score_batch(docs)
    assert len(results) == 15
    for i, res in enumerate(results):
        assert res.document_id == f"doc_{i:03d}"
        assert res.domain_tier == 1
        assert res.domain_score == 1.00


# ---------------------------------------------------------------------------
# 4. Boundary Cases: Empty Sequence and Single Item
# ---------------------------------------------------------------------------

def test_score_batch_empty_and_single_item(scorer: SourceCredibilityScorer):
    """Verify boundary conditions: empty list, empty tuple, and single document."""
    assert scorer.score_batch([]) == []
    assert scorer.score_batch(()) == []

    single = [{"document_id": "solo_1", "source_url": "https://nasa.gov"}]
    res_list = scorer.score_batch(single)
    assert len(res_list) == 1
    assert res_list[0].document_id == "solo_1"
    assert res_list[0].domain_tier == 1


# ---------------------------------------------------------------------------
# 5. Chunking Large Payloads with Invariant Parity
# ---------------------------------------------------------------------------

def test_score_batch_chunking_large_payload(scorer: SourceCredibilityScorer):
    """Verify chunked execution (batch_size=1, 5, 10, 25, 50) preserves parity on N=25."""
    docs = [
        {
            "document_id": f"chunk_test_{i}",
            "source_url": f"https://site{i}.edu" if i % 2 == 0 else f"https://site{i}.org",
            "source": f"Source {i}",
        }
        for i in range(25)
    ]

    baseline = scorer.score_batch(docs, batch_size=25)
    chunked_1 = scorer.score_batch(docs, batch_size=1)
    chunked_5 = scorer.score_batch(docs, batch_size=5)
    chunked_10 = scorer.score_batch(docs, batch_size=10)
    chunked_50 = scorer.score_batch(docs, batch_size=50)

    assert len(baseline) == 25
    assert len(chunked_1) == 25
    assert len(chunked_5) == 25
    assert len(chunked_10) == 25
    assert len(chunked_50) == 25

    # Exact parity across all chunk configurations
    for i in range(25):
        assert baseline[i].document_id == chunked_1[i].document_id == chunked_10[i].document_id
        assert baseline[i].credibility_score == chunked_1[i].credibility_score == chunked_10[i].credibility_score
        assert baseline[i].domain_tier == chunked_5[i].domain_tier == chunked_50[i].domain_tier


# ---------------------------------------------------------------------------
# 6. Heterogeneous Input Types in Single Batch
# ---------------------------------------------------------------------------

@dataclass
class CustomAttributedDoc:
    document_id: str
    source_url: str
    source: str
    published_date: str


def test_score_batch_heterogeneous_input_types(scorer: SourceCredibilityScorer):
    """Verify mixed batch containing EvidenceItem, CorpusDocument, RetrievalResult, dict, and dataclass."""
    ev_item = EvidenceItem(
        evidence_id="ev_pydantic_01",
        text="A retrieved snippet from an academic source.",
        source_url="https://arxiv.org/abs/2101.00001",
        domain="arxiv.org",
    )
    corpus_doc = CorpusDocument(
        id="corpus_pydantic_02",
        title="Corpus Title",
        text="Corpus text snippet.",
        source="CDC Official",
        source_url="https://www.cdc.gov/data",
        domain="cdc.gov",
        published_date=date(2023, 1, 1),
    )
    retrieval_res = RetrievalResult(
        document_id="ret_pydantic_03",
        title="Retrieved Passage",
        text="Text snippet.",
        source="Nature Research",
        source_url="https://nature.com/articles/sample",
        domain="nature.com",
        published_date=date(2023, 4, 12),
        score=0.92,
        retrieval_method="HYBRID",
        rank=1,
    )
    dict_item = {
        "document_id": "dict_04",
        "source_url": "https://theguardian.com/article",
        "source": "The Guardian",
    }
    custom_obj = CustomAttributedDoc(
        document_id="custom_05",
        source_url="https://mit.edu/lab",
        source="MIT Laboratory",
        published_date="2023-11-05",
    )

    mixed_batch = [ev_item, corpus_doc, retrieval_res, dict_item, custom_obj]
    results = scorer.score_batch(mixed_batch, batch_size=2)

    assert len(results) == 5
    assert results[0].document_id == "ev_pydantic_01"
    assert results[0].domain_tier == 2  # arxiv.org
    assert results[1].document_id == "corpus_pydantic_02"
    assert results[1].domain_tier == 1  # cdc.gov
    assert results[2].document_id == "ret_pydantic_03"
    assert results[2].domain_tier == 2  # nature.com
    assert results[3].document_id == "dict_04"
    assert results[3].domain_tier == 3  # theguardian.com
    assert results[4].document_id == "custom_05"
    assert results[4].domain_tier == 1  # mit.edu


# ---------------------------------------------------------------------------
# 7. Corrupt and Missing Metadata Resilience
# ---------------------------------------------------------------------------

def test_score_batch_missing_and_corrupt_metadata(scorer: SourceCredibilityScorer):
    """Verify batch handling of corrupt URLs, bare IPs, invalid schemes, and empty payloads."""
    corrupt_cases = [
        {"document_id": "c1", "source_url": "http://"},
        {"document_id": "c2", "source_url": "not-a-valid-scheme://foo"},
        {"document_id": "c3", "source_url": ""},
        {"document_id": "c4", "source_url": None},
        {"document_id": "c5", "source_url": "http://localhost:8000/api"},
        {"document_id": "c6", "source_url": "http://127.0.0.1/test"},
        {"document_id": "c7", "source_url": "https://test.invalid/doc"},
        {"document_id": "c8", "source_url": "https://service.example/index"},
        {"document_id": "c9"},  # completely empty metadata
    ]
    results = scorer.score_batch(corrupt_cases)
    assert len(results) == 9
    for res in results:
        assert res.domain_tier == 4
        assert res.domain_score == 0.25
        assert 0.0 <= res.credibility_score <= 1.0


# ---------------------------------------------------------------------------
# 8. Provenance Completeness Monotonicity
# ---------------------------------------------------------------------------

def test_score_batch_provenance_completeness(scorer: SourceCredibilityScorer):
    """Verify provenance scores scale monotonically with the count of valid schema fields."""
    docs = [
        # 6 fields: document_id, title, source, source_url, domain, published_date
        {
            "document_id": "prov_6",
            "title": "Full Metadata",
            "source": "Official Agency",
            "source_url": "https://agency.gov/page",
            "domain": "agency.gov",
            "published_date": "2024-01-01",
        },
        # 4 fields
        {
            "document_id": "prov_4",
            "title": "Partial Metadata",
            "source": "Official Agency",
            "source_url": "https://agency.gov/page",
        },
        # 2 fields
        {
            "document_id": "prov_2",
            "source_url": "https://agency.gov/page",
        },
        # 0 considered fields present
        {},
    ]
    results = scorer.score_batch(docs)
    assert len(results) == 4
    assert results[0].provenance_score == 1.0
    assert results[1].provenance_score == pytest.approx(4 / 6, abs=1e-4)
    assert results[2].provenance_score == pytest.approx(2 / 6, abs=1e-4)
    assert results[3].provenance_score == 0.0

    assert results[0].provenance_score > results[1].provenance_score > results[2].provenance_score > results[3].provenance_score


# ---------------------------------------------------------------------------
# 9. Dynamic Source-Type Weight Normalization Across Batch
# ---------------------------------------------------------------------------

def test_score_batch_source_type_weight_adjustment(scorer: SourceCredibilityScorer):
    """Verify weight normalization dynamically adapts when source_type is present vs absent in same batch."""
    batch = [
        # With source_type -> full 3-way weights (0.50, 0.20, 0.30)
        {"document_id": "st_present", "source_url": "https://nih.gov", "source_type": "government"},
        # Without source_type -> renormalized 2-way weights (0.5/0.8, 0.3/0.8) = (0.625, 0.375)
        {"document_id": "st_absent", "source_url": "https://nih.gov"},
    ]
    results = scorer.score_batch(batch)
    assert len(results) == 2

    # Present case
    res_pres = results[0]
    assert res_pres.source_type_score is not None
    assert res_pres.weights["w_domain"] == 0.50
    assert res_pres.weights["w_type"] == 0.20
    assert res_pres.weights["w_provenance"] == 0.30
    assert sum(res_pres.weights.values()) == pytest.approx(1.0)

    # Absent case
    res_abs = results[1]
    assert res_abs.source_type_score is None
    assert res_abs.weights["w_domain"] == pytest.approx(0.625, abs=1e-4)
    assert res_abs.weights.get("w_type", 0.0) == 0.0
    assert res_abs.weights["w_provenance"] == pytest.approx(0.375, abs=1e-4)
    assert sum(res_abs.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. Mutation Testing & Defensive Invariants
# ---------------------------------------------------------------------------

def test_source_credibility_mutation_pass(scorer: SourceCredibilityScorer):
    """Mutation testing: verify invalid inputs, corrupted weights, and altered lookups fail as expected."""
    sample_docs = [{"document_id": "m1", "source_url": "https://nih.gov"}]

    # Mutation 1: Invalid batch sizes
    for bad_size in (0, -1, -100, "10", True, False, None, 3.14):
        with pytest.raises(SourceCredibilityInputError):
            scorer.score_batch(sample_docs, batch_size=bad_size)  # type: ignore

    # Mutation 2: Weights violating sum to 1.0 constraint
    with pytest.raises(SourceCredibilityConfigError):
        SourceCredibilityWeights(w_domain=0.50, w_type=0.20, w_provenance=0.50)

    # Mutation 3: Negative or infinite weight
    with pytest.raises(SourceCredibilityConfigError):
        SourceCredibilityWeights(w_domain=-0.10, w_type=0.60, w_provenance=0.50)
    with pytest.raises(SourceCredibilityConfigError):
        SourceCredibilityWeights(w_domain=float("nan"), w_type=0.50, w_provenance=0.50)

    # Mutation 4: Custom settings missing .int would fail who.int classification
    mutated_settings = SourceCredibilitySettings(
        tier1_suffixes=(".gov", ".edu", ".mil"),  # intentionally missing .int
    )
    mutated_scorer = SourceCredibilityScorer(settings=mutated_settings)
    mutated_res = mutated_scorer.score({"document_id": "mut_who", "source_url": "https://who.int"})
    # Without .int suffix, who.int is not in tier 1, tier 2, or tier 3 -> falls to tier 4
    assert mutated_res.domain_tier == 4, (
        "Without .int suffix in settings, who.int must fall to Tier 4, proving .int is the required selector."
    )
