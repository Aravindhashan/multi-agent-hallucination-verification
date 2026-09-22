"""Tests for Step 4.3 Source Independence & Redundancy Detector.

Covers exact duplicates, near duplicates, Jaccard math correctness, metadata signals,
anti-double-counting hierarchy, edge cases (empty, short, self-comparison), clock isolation,
complete-linkage clustering, Jaccard non-transitivity, performance, and snapshots.
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Any
import pytest

from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult
from app.services.source_independence_config import (
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DEFAULT_WEAK_RELATION_THRESHOLD,
    DISCLAIMER_PHRASE,
    SourceIndependenceConfigError,
    SourceIndependenceInputError,
    SourceIndependenceSettings,
)
from app.services.source_independence_models import (
    IndependenceCluster,
    IndependencePairResult,
    IndependenceSetResult,
    RedundancyClass,
)
from app.services.source_independence_scorer import (
    SourceIndependenceDetector,
    compute_char_ngram_jaccard,
    compute_jaccard_similarity,
    extract_token_set,
    get_source_independence_detector,
    normalize_domain,
    normalize_source,
    normalize_text,
    normalize_url,
    reset_source_independence_detector,
    tokenize,
)


@pytest.fixture(autouse=True)
def cleanup_detector() -> None:
    """Reset singleton detector before and after each test."""
    reset_source_independence_detector()
    yield
    reset_source_independence_detector()


# =====================================================================
# 1. Exact Duplicate & Text Normalization Tests (§10, §11)
# =====================================================================


def test_exact_duplicate_identical_raw_text() -> None:
    """Identical raw text with sufficient tokens produces EXACT_DUPLICATE and independence 0.0."""
    detector = SourceIndependenceDetector()
    text = "The James Webb Space Telescope was launched in December 2021 by Ariane 5."
    doc_a = {"document_id": "doc_1", "text": text}
    doc_b = {"document_id": "doc_2", "text": text}

    res = detector.compare(doc_a, doc_b)
    assert res.is_exact_duplicate is True
    assert res.is_near_duplicate is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.text_similarity == 1.0
    assert res.independence_score == 0.0
    assert DISCLAIMER_PHRASE in res.explanation


def test_exact_duplicate_whitespace_and_newline_variation() -> None:
    """Whitespace runs, tabs, and newlines collapse deterministically to yield exact duplicates."""
    detector = SourceIndependenceDetector()
    text_a = "The  Apollo 11 mission landed on the Moon.\n\nNeil Armstrong walked."
    text_b = "The Apollo 11\tmission   landed on the Moon. Neil Armstrong walked."
    doc_a = {"document_id": "doc_a", "text": text_a}
    doc_b = {"document_id": "doc_b", "text": text_b}

    res = detector.compare(doc_a, doc_b)
    assert res.is_exact_duplicate is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.text_similarity == 1.0
    assert res.independence_score == 0.0


def test_exact_duplicate_case_variation() -> None:
    """Case variation collapses under normalized comparison."""
    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "doc_a", "text": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG."}
    doc_b = {"document_id": "doc_b", "text": "the quick brown fox jumps over the lazy dog."}

    res = detector.compare(doc_a, doc_b)
    assert res.is_exact_duplicate is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.text_similarity == 1.0
    assert res.independence_score == 0.0


def test_exact_duplicate_unicode_nfkc_variation() -> None:
    """Unicode NFKC decomposition and ligature compatibility (e.g. fi ligature \ufb01 vs fi)."""
    detector = SourceIndependenceDetector()
    # \ufb01 is the 'fi' ligature character
    doc_a = {"document_id": "doc_a", "text": "Scientific \ufb01ndings on solar physics."}
    doc_b = {"document_id": "doc_b", "text": "Scientific findings on solar physics."}

    res = detector.compare(doc_a, doc_b)
    assert res.is_exact_duplicate is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.text_similarity == 1.0


def test_raw_text_not_mutated() -> None:
    """Verify that text normalization is a pure function and does not mutate raw inputs."""
    detector = SourceIndependenceDetector()
    raw_a = "  Original  Text   With   Spaces  \n"
    raw_b = "Original text with spaces"
    doc_a = {"document_id": "doc_a", "text": raw_a}
    doc_b = {"document_id": "doc_b", "text": raw_b}

    _ = detector.compare(doc_a, doc_b)
    assert doc_a["text"] == raw_a
    assert doc_b["text"] == raw_b


# =====================================================================
# 2. Mathematical Correctness & Jaccard Calculation (§13)
# =====================================================================


def test_jaccard_identical_non_empty_sets() -> None:
    """Identical non-empty token sets produce J = 1.0."""
    tokens = frozenset(["quantum", "computing", "superposition"])
    assert compute_jaccard_similarity(tokens, tokens) == 1.0


def test_jaccard_disjoint_sets() -> None:
    """Completely disjoint token sets produce J = 0.0."""
    set_a = frozenset(["apple", "banana", "cherry"])
    set_b = frozenset(["car", "train", "airplane"])
    assert compute_jaccard_similarity(set_a, set_b) == 0.0


def test_jaccard_both_empty_sets_guarded() -> None:
    """Both sets empty safely returns 0.0 without ZeroDivisionError."""
    assert compute_jaccard_similarity(frozenset(), frozenset()) == 0.0


def test_jaccard_one_empty_set() -> None:
    """One empty set and one non-empty set produces 0.0."""
    set_a = frozenset(["galaxy", "star"])
    assert compute_jaccard_similarity(set_a, frozenset()) == 0.0
    assert compute_jaccard_similarity(frozenset(), set_a) == 0.0


def test_jaccard_exact_known_overlap() -> None:
    """Test known partial overlap: A={1,2,3,4}, B={3,4,5,6}, intersection=2, union=6 -> 1/3."""
    set_a = frozenset(["a", "b", "c", "d"])
    set_b = frozenset(["c", "d", "e", "f"])
    expected = 2.0 / 6.0
    assert pytest.approx(compute_jaccard_similarity(set_a, set_b), rel=1e-5) == expected


# =====================================================================
# 3. Near Duplicate & Threshold Boundaries (§12, §15, §20)
# =====================================================================


def test_near_duplicate_high_overlap_redundant() -> None:
    """Text with >85% token overlap is classified as REDUNDANT with is_near_duplicate=True."""
    detector = SourceIndependenceDetector()
    # 9 shared words, 1 unique word each -> 9/11 = 0.8182 (below 0.85)
    # Let's create 19 shared words, 1 unique each -> 19 / 21 = 0.9048 (above 0.85)
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    text_a = f"{shared} variant1"
    text_b = f"{shared} variant2"

    res = detector.compare({"document_id": "a", "text": text_a}, {"document_id": "b", "text": text_b})
    assert res.redundancy_class == RedundancyClass.REDUNDANT
    assert res.is_near_duplicate is True
    assert res.is_exact_duplicate is False
    assert res.text_similarity > 0.85
    assert res.independence_score < 0.15


def test_near_duplicate_exact_threshold_boundary() -> None:
    """Test behavior right at or above the near_duplicate_threshold."""
    # Using 17 shared tokens and 3 unique across both -> 17/20 = 0.85 exactly
    shared = [f"tok_{i}" for i in range(17)]
    text_a = " ".join(shared + ["unique_a1", "unique_a2"])
    text_b = " ".join(shared + ["unique_b1"])  # union = 17 + 2 + 1 = 20, intersection = 17 -> 17/20 = 0.85

    detector = SourceIndependenceDetector(SourceIndependenceSettings(near_duplicate_threshold=0.85))
    res = detector.compare({"document_id": "a", "text": text_a}, {"document_id": "b", "text": text_b})
    assert res.text_similarity == 0.85
    assert res.redundancy_class == RedundancyClass.REDUNDANT
    assert res.is_near_duplicate is True


def test_just_below_near_duplicate_threshold() -> None:
    """Similarity just below 0.85 is classified as WEAKLY_RELATED."""
    # 16 shared, 2 unique in a, 2 unique in b -> union = 20, inter = 16 -> 16/20 = 0.80
    shared = [f"word_{i}" for i in range(16)]
    text_a = " ".join(shared + ["extra_a1", "extra_a2"])
    text_b = " ".join(shared + ["extra_b1", "extra_b2"])

    detector = SourceIndependenceDetector(SourceIndependenceSettings(near_duplicate_threshold=0.85, weak_relation_threshold=0.50))
    res = detector.compare({"document_id": "a", "text": text_a}, {"document_id": "b", "text": text_b})
    assert res.text_similarity == 0.80
    assert res.redundancy_class == RedundancyClass.WEAKLY_RELATED
    assert res.is_near_duplicate is False


def test_unrelated_text_is_independent() -> None:
    """Completely unrelated text produces INDEPENDENT classification with high independence score."""
    detector = SourceIndependenceDetector()
    text_a = "Photosynthesis converts solar light energy into glucose."
    text_b = "Ancient Romans constructed stone aqueducts across Europe."

    res = detector.compare({"document_id": "a", "text": text_a}, {"document_id": "b", "text": text_b})
    assert res.redundancy_class == RedundancyClass.INDEPENDENT
    assert res.is_near_duplicate is False
    assert res.is_exact_duplicate is False
    assert res.text_similarity == 0.0
    assert res.independence_score == 1.0


# =====================================================================
# 4. Metadata Signals & Anti-Double-Counting Hierarchy (§16, §17, §18, §22)
# =====================================================================


def test_same_canonical_url_triggers_exact_duplicate() -> None:
    """Matching canonical URLs trigger EXACT_DUPLICATE even if text has slight reporting edits."""
    detector = SourceIndependenceDetector()
    url = "https://www.nature.com/articles/s41586-023-00001-x#abstract"
    doc_a = {
        "document_id": "doc_a",
        "text": "Quantum computers demonstrate algorithmic speedup on structured problems.",
        "source_url": url,
    }
    doc_b = {
        "document_id": "doc_b",
        "text": "Quantum computers demonstrate significant algorithmic speedup on specific problems.",
        "source_url": "https://nature.com/articles/s41586-023-00001-x/",
    }

    res = detector.compare(doc_a, doc_b)
    assert res.same_url is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.is_exact_duplicate is True
    assert res.independence_score == 0.0


def test_same_domain_low_similarity_is_not_redundant() -> None:
    """Anti-double-counting (§22): Same domain with low similarity is NOT REDUNDANT."""
    detector = SourceIndependenceDetector()
    doc_a = {
        "document_id": "reuters_1",
        "text": "Oil prices surged following disruptions in global shipping corridors through the Red Sea.",
        "domain": "reuters.com",
        "source": "Reuters",
    }
    doc_b = {
        "document_id": "reuters_2",
        "text": "A new species of deep sea coral was discovered near the Galapagos Marine Reserve.",
        "domain": "reuters.com",
        "source": "Reuters",
    }

    res = detector.compare(doc_a, doc_b)
    assert res.same_domain is True
    assert res.same_source is True
    # Crucial assertion: same domain + low similarity is NOT REDUNDANT
    assert res.redundancy_class != RedundancyClass.REDUNDANT
    assert res.redundancy_class != RedundancyClass.EXACT_DUPLICATE
    assert res.is_near_duplicate is False
    assert res.is_exact_duplicate is False
    assert res.redundancy_class == RedundancyClass.WEAKLY_RELATED
    assert res.independence_score >= 0.80


def test_different_domains_identical_text_is_exact_duplicate() -> None:
    """Syndicated articles across different publishers are recognized as exact text duplicates."""
    detector = SourceIndependenceDetector()
    text = "The Federal Reserve held benchmark interest rates steady at 5.25 percent today."
    doc_a = {"document_id": "ap_1", "text": text, "domain": "apnews.com", "source": "Associated Press"}
    doc_b = {"document_id": "yahoo_1", "text": text, "domain": "yahoo.com", "source": "Yahoo News"}

    res = detector.compare(doc_a, doc_b)
    assert res.same_domain is False
    assert res.same_source is False
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.is_exact_duplicate is True
    assert res.independence_score == 0.0


def test_missing_metadata_handled_cleanly() -> None:
    """Missing URL, domain, and source don't crash and default to text-based analysis."""
    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "doc_a", "text": "Mars rover Curiosity detected organic molecules in Gale crater."}
    doc_b = {"document_id": "doc_b", "text": "Curiosity detected complex organic molecules in Gale crater on Mars."}

    res = detector.compare(doc_a, doc_b)
    assert res.same_domain is False
    assert res.same_source is False
    assert res.same_url is False
    assert res.shared_metadata_signal is False
    assert res.redundancy_class in (RedundancyClass.REDUNDANT, RedundancyClass.WEAKLY_RELATED)


# =====================================================================
# 5. Edge Cases (§25, §26, §27)
# =====================================================================


def test_self_comparison_policy() -> None:
    """Self-comparison (id_a == id_b) produces EXACT_DUPLICATE with 0.0 independence (§27)."""
    detector = SourceIndependenceDetector()
    doc = {
        "document_id": "unique_doc_id",
        "text": "A standard evidence text for self-comparison verification.",
        "domain": "example.org",
    }

    res = detector.compare(doc, doc)
    assert res.document_id_a == "unique_doc_id"
    assert res.document_id_b == "unique_doc_id"
    assert res.is_exact_duplicate is True
    assert res.is_near_duplicate is True
    assert res.redundancy_class == RedundancyClass.EXACT_DUPLICATE
    assert res.text_similarity == 1.0
    assert res.independence_score == 0.0
    assert "Self-comparison" in res.explanation


def test_short_text_policy_below_meaningful_tokens() -> None:
    """Short text below minimum_meaningful_tokens cannot exceed WEAKLY_RELATED without matching URL (§26)."""
    detector = SourceIndependenceDetector(SourceIndependenceSettings(minimum_meaningful_tokens=5))
    doc_a = {"document_id": "a", "text": "the the"}
    doc_b = {"document_id": "b", "text": "the the"}

    res = detector.compare(doc_a, doc_b)
    # Token similarity is 1.0 (both contain only 'the'), but len(tokens)=1 < 5
    assert res.text_similarity == 1.0
    assert res.is_exact_duplicate is False
    assert res.is_near_duplicate is False
    assert res.redundancy_class == RedundancyClass.WEAKLY_RELATED
    assert "below the meaningful threshold" in res.explanation


def test_empty_and_whitespace_text() -> None:
    """Empty and whitespace-only text safely returns 0.0 similarity and INDEPENDENT."""
    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "a", "text": ""}
    doc_b = {"document_id": "b", "text": "   \n\t  "}

    res = detector.compare(doc_a, doc_b)
    assert res.text_similarity == 0.0
    assert res.redundancy_class == RedundancyClass.INDEPENDENT
    assert res.independence_score == 1.0


def test_malformed_input_raises_typed_error() -> None:
    """Malformed evidence input (None, missing ID, missing text) raises SourceIndependenceInputError."""
    detector = SourceIndependenceDetector()

    with pytest.raises(SourceIndependenceInputError):
        detector.compare(None, {"document_id": "b", "text": "valid text"})

    with pytest.raises(SourceIndependenceInputError):
        detector.compare({"text": "no id"}, {"document_id": "b", "text": "valid"})

    with pytest.raises(SourceIndependenceInputError):
        detector.compare({"document_id": "a"}, {"document_id": "b", "text": "valid"})


def test_compatibility_with_corpus_document_and_retrieval_result() -> None:
    """Works directly with real CorpusDocument and RetrievalResult models."""
    detector = SourceIndependenceDetector()
    corpus_doc = CorpusDocument(
        id="corpus_101",
        title="Hubble Launch",
        text="The Hubble Space Telescope was launched into low Earth orbit in 1990.",
        source="NASA",
        domain="nasa.gov",
    )
    retrieval_res = RetrievalResult(
        document_id="retrieval_202",
        title="Hubble Space Mission",
        text="The Hubble Space Telescope was launched into low Earth orbit in 1990 by Discovery.",
        source="NASA Press",
        domain="nasa.gov",
        score=0.95,
        retrieval_method="hybrid",
        rank=1,
    )

    res = detector.compare(corpus_doc, retrieval_res)
    assert res.document_id_a == "corpus_101"
    assert res.document_id_b == "retrieval_202"
    assert res.redundancy_class in (RedundancyClass.REDUNDANT, RedundancyClass.EXACT_DUPLICATE)


# =====================================================================
# 6. Determinism & Clock Isolation (§30, §31)
# =====================================================================


def test_determinism_identical_runs() -> None:
    """Fresh detector instances yield bit-for-bit identical results and explanations."""
    det1 = SourceIndependenceDetector()
    det2 = SourceIndependenceDetector()

    doc_a = {"document_id": "a", "text": "Synthetic biology enables metabolic pathway redesign."}
    doc_b = {"document_id": "b", "text": "Metabolic pathway engineering redesigns microbial genomes."}

    r1 = det1.compare(doc_a, doc_b)
    r2 = det2.compare(doc_a, doc_b)

    assert r1.text_similarity == r2.text_similarity
    assert r1.independence_score == r2.independence_score
    assert r1.redundancy_class == r2.redundancy_class
    assert r1.explanation == r2.explanation


def test_system_clock_monkeypatch_does_not_affect_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatching system clock functions confirms complete clock independence."""
    fake_date = datetime.date(2099, 12, 31)

    class FakeDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            return fake_date

    monkeypatch.setattr(datetime, "date", FakeDate)

    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "a", "text": "Determinism clock isolation test text A."}
    doc_b = {"document_id": "b", "text": "Determinism clock isolation test text B."}

    res = detector.compare(doc_a, doc_b)
    assert isinstance(res, IndependencePairResult)
    assert "2099" not in res.explanation


# =====================================================================
# 7. Batch Analysis & Canonical Pair Ordering (§28)
# =====================================================================


def test_compare_batch_canonical_ordering_and_no_duplicates() -> None:
    """Batch analysis returns strictly canonical pairs (0,1), (0,2), (1,2) with no (B,A)."""
    detector = SourceIndependenceDetector()
    items = [
        {"document_id": "A", "text": "First unique evidence article on astrophysics."},
        {"document_id": "B", "text": "Second unique evidence article on cellular biology."},
        {"document_id": "C", "text": "Third unique evidence article on macroeconomics."},
    ]

    pairs = detector.compare_batch(items)
    assert len(pairs) == 3

    assert pairs[0].document_id_a == "A" and pairs[0].document_id_b == "B"
    assert pairs[1].document_id_a == "A" and pairs[1].document_id_b == "C"
    assert pairs[2].document_id_a == "B" and pairs[2].document_id_b == "C"


def test_compare_batch_empty_and_single() -> None:
    """Batch analysis on 0 or 1 item returns an empty list."""
    detector = SourceIndependenceDetector()
    assert detector.compare_batch([]) == []
    assert detector.compare_batch([{"document_id": "one", "text": "one"}]) == []


def test_batch_equivalence_proof() -> None:
    """Batch results equal sequential individual compare() calls for all pairs (§50)."""
    detector = SourceIndependenceDetector()
    items = [
        {"document_id": f"doc_{i}", "text": f"Document text variation number {i} with astronomy terms."}
        for i in range(5)
    ]

    batch_res = detector.compare_batch(items)
    expected_pairs: list[IndependencePairResult] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            expected_pairs.append(detector.compare(items[i], items[j]))

    assert len(batch_res) == len(expected_pairs)
    for b_res, exp_res in zip(batch_res, expected_pairs):
        assert b_res.document_id_a == exp_res.document_id_a
        assert b_res.document_id_b == exp_res.document_id_b
        assert b_res.text_similarity == exp_res.text_similarity
        assert b_res.independence_score == exp_res.independence_score
        assert b_res.redundancy_class == exp_res.redundancy_class


# =====================================================================
# 8. Group-Level Analysis & Jaccard Non-Transitivity (§23, §24, §50)
# =====================================================================


def test_jaccard_non_transitivity_worked_example() -> None:
    """Demonstrate mathematically that Jaccard similarity is non-transitive.

    If A and B share 90% tokens, and B and C share 90% tokens, A and C can share only ~60%.
    Complete-linkage clustering correctly avoids chaining unrelated documents.
    """
    # Define token sets:
    # A has tokens 0..19
    # B has tokens 2..21 (overlap 18, union 22 -> 18/22 = 0.818)
    # C has tokens 4..23 (overlap with B: 18/22 = 0.818, overlap with A: 16/24 = 0.667)
    tokens_a = frozenset(f"t_{i}" for i in range(20))
    tokens_b = frozenset(f"t_{i}" for i in range(2, 22))
    tokens_c = frozenset(f"t_{i}" for i in range(4, 24))

    j_ab = compute_jaccard_similarity(tokens_a, tokens_b)
    j_bc = compute_jaccard_similarity(tokens_b, tokens_c)
    j_ac = compute_jaccard_similarity(tokens_a, tokens_c)

    assert j_ab > 0.80
    assert j_bc > 0.80
    assert j_ac < 0.70  # Transitivity fails: j_ac is significantly lower!


def test_analyze_set_complete_linkage_clustering() -> None:
    """Group-level clustering groups mutually redundant documents into clusters without chaining."""
    detector = SourceIndependenceDetector(SourceIndependenceSettings(near_duplicate_threshold=0.80))
    text_cluster_1 = "The international space station orbits the Earth approximately every ninety minutes."
    text_cluster_1_dup = "The international space station orbits the Earth approximately every ninety minutes in space."
    text_independent = "Ancient Egyptian hieroglyphs were deciphered using the bilingual inscriptions of the Rosetta Stone."

    docs = [
        {"document_id": "iss_1", "text": text_cluster_1},
        {"document_id": "iss_2", "text": text_cluster_1_dup},
        {"document_id": "rosetta", "text": text_independent},
    ]

    set_result = detector.analyze_set(docs)
    assert set_result.total_documents == 3
    assert len(set_result.clusters) == 1
    assert set_result.clusters[0].document_ids == ["iss_1", "iss_2"]
    assert "rosetta" in set_result.independent_document_ids


# =====================================================================
# 9. Configuration & Validation (§29)
# =====================================================================


def test_config_valid_parameters() -> None:
    """Valid settings instantiate properly."""
    cfg = SourceIndependenceSettings(
        near_duplicate_threshold=0.90,
        weak_relation_threshold=0.45,
        minimum_meaningful_tokens=6,
    )
    assert cfg.near_duplicate_threshold == 0.90
    assert cfg.weak_relation_threshold == 0.45
    assert cfg.minimum_meaningful_tokens == 6


def test_config_invalid_threshold_raises() -> None:
    """Invalid near_duplicate_threshold values raise SourceIndependenceConfigError."""
    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(near_duplicate_threshold=0.0)

    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(near_duplicate_threshold=1.5)

    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(near_duplicate_threshold=float("nan"))


def test_config_invalid_weak_threshold_raises() -> None:
    """weak_relation_threshold >= near_duplicate_threshold raises SourceIndependenceConfigError."""
    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(near_duplicate_threshold=0.80, weak_relation_threshold=0.85)

    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(near_duplicate_threshold=0.80, weak_relation_threshold=-0.1)


def test_config_invalid_tokens_raises() -> None:
    """minimum_meaningful_tokens < 1 raises SourceIndependenceConfigError."""
    with pytest.raises(SourceIndependenceConfigError):
        SourceIndependenceSettings(minimum_meaningful_tokens=0)


# =====================================================================
# 10. Serialization & JSON Round-Trip (§19)
# =====================================================================


def test_json_round_trip() -> None:
    """IndependencePairResult and IndependenceSetResult survive JSON round-tripping."""
    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "a", "text": "Validating JSON serialization round trip."}
    doc_b = {"document_id": "b", "text": "Validating JSON serialization and deserialization round trip."}

    pair_res = detector.compare(doc_a, doc_b)
    json_str = pair_res.model_dump_json()
    reconstituted = IndependencePairResult.model_validate_json(json_str)

    assert reconstituted.document_id_a == pair_res.document_id_a
    assert reconstituted.document_id_b == pair_res.document_id_b
    assert reconstituted.text_similarity == pair_res.text_similarity
    assert reconstituted.redundancy_class == pair_res.redundancy_class
    assert reconstituted.independence_score == pair_res.independence_score


# =====================================================================
# 11. Optional Add-ons: Sensitivity Sweep & Snapshots (§50)
# =====================================================================


def test_threshold_sensitivity_sweep() -> None:
    """Sweep near_duplicate_threshold across {0.70, 0.80, 0.85, 0.90, 0.95} on a fixed pair."""
    # Shared 17, union 20 -> similarity = 0.85
    shared = [f"token_{i}" for i in range(17)]
    doc_a = {"document_id": "a", "text": " ".join(shared + ["u1", "u2"])}
    doc_b = {"document_id": "b", "text": " ".join(shared + ["u3"])}

    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = {}
    for th in thresholds:
        det = SourceIndependenceDetector(SourceIndependenceSettings(near_duplicate_threshold=th, weak_relation_threshold=0.4))
        res = det.compare(doc_a, doc_b)
        results[th] = res.redundancy_class

    # At 0.70, 0.80, 0.85: similarity 0.85 >= th -> REDUNDANT
    assert results[0.70] == RedundancyClass.REDUNDANT
    assert results[0.80] == RedundancyClass.REDUNDANT
    assert results[0.85] == RedundancyClass.REDUNDANT
    # At 0.90, 0.95: similarity 0.85 < th -> WEAKLY_RELATED
    assert results[0.90] == RedundancyClass.WEAKLY_RELATED
    assert results[0.95] == RedundancyClass.WEAKLY_RELATED


def test_character_ngram_jaccard_fallback() -> None:
    """Verify compute_char_ngram_jaccard functionality."""
    sim = compute_char_ngram_jaccard("apple", "apples", n=3)
    assert 0.0 < sim < 1.0


def test_snapshots_stability() -> None:
    """Snapshot test for representative pairwise relationships."""
    detector = SourceIndependenceDetector()
    snapshots = [
        # 1. Exact text duplicate
        (
            {"document_id": "s1_a", "text": "The quick brown fox jumps over the lazy dog."},
            {"document_id": "s1_b", "text": "The quick brown fox jumps over the lazy dog."},
            RedundancyClass.EXACT_DUPLICATE,
            1.0,
            0.0,
        ),
        # 2. Independent
        (
            {"document_id": "s2_a", "text": "Mars atmosphere consists predominantly of carbon dioxide."},
            {"document_id": "s2_b", "text": "Ludwig van Beethoven composed nine symphonies during his lifetime."},
            RedundancyClass.INDEPENDENT,
            0.0,
            1.0,
        ),
        # 3. Matching URL
        (
            {"document_id": "s3_a", "text": "Article text variation alpha.", "source_url": "https://bbc.com/news/101"},
            {"document_id": "s3_b", "text": "Article text variation beta.", "source_url": "https://bbc.com/news/101#comments"},
            RedundancyClass.EXACT_DUPLICATE,
            0.0,
        ),
    ]

    for item in snapshots:
        if len(item) == 5:
            doc_a, doc_b, expected_class, expected_sim, expected_ind = item
            res = detector.compare(doc_a, doc_b)
            assert res.redundancy_class == expected_class
            assert res.text_similarity == expected_sim
            assert res.independence_score == expected_ind
        else:
            doc_a, doc_b, expected_class, expected_ind = item
            res = detector.compare(doc_a, doc_b)
            assert res.redundancy_class == expected_class
            assert res.independence_score == expected_ind


# =====================================================================
# 12. Performance Targets (§39)
# =====================================================================


def test_performance_single_pair_under_five_ms() -> None:
    """Warm single pair comparison runs well under 5.0 ms."""
    detector = SourceIndependenceDetector()
    doc_a = {"document_id": "a", "text": "Performance measurement text paragraph for single pair benchmark."}
    doc_b = {"document_id": "b", "text": "Performance measurement text paragraph for single pair benchmark validation."}

    # Warm-up
    for _ in range(20):
        detector.compare(doc_a, doc_b)

    t0 = time.perf_counter()
    for _ in range(100):
        detector.compare(doc_a, doc_b)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / 100) * 1000
    assert avg_ms < 5.0


def test_singleton_accessor() -> None:
    """get_source_independence_detector singleton lifecycle."""
    det1 = get_source_independence_detector()
    det2 = get_source_independence_detector()
    assert det1 is det2
    reset_source_independence_detector()
    det3 = get_source_independence_detector()
    assert det3 is not det1
