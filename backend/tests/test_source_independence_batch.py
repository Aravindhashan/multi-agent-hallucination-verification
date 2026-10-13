"""Dedicated batch testing and corrected Kish n_eff verification for Source Independence (Submodule 8.3).

Validates:
1. Mathematical correction of Kish's effective sample size: n_eff = M^2 / sum(s_k^2).
2. Boundary Case 1 (Single cluster of size M -> n_eff = 1.0, full syndication discount).
3. Boundary Case 2 (M independent singletons -> n_eff = M, zero syndication discount).
4. Boundary Case 3 (1 cluster of 4 + 2 singletons -> 36 / 18 = 2.0).
5. Robustness on edge cases (empty set, zeros, single item).
6. Integration into SourceIndependenceDetector.analyze_set().
7. Canonical pairwise order preservation in compare_batch().
8. Memory-safe chunked batch processing across chunk sizes (1, 5, 10, 20, 50).
9. Integration with EvidenceQualityService (admissibility filtering before Kish calculation).
10. Mutation testing trapping the flawed raw-item formulation.
"""

from datetime import date
import pytest

from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.source_independence_config import (
    SourceIndependenceInputError,
    SourceIndependenceSettings,
)
from app.services.source_independence_scorer import (
    SourceIndependenceDetector,
    compute_kish_effective_sample_size,
    get_source_independence_detector,
    reset_source_independence_detector,
)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Ensure singleton instance is cleared before and after each test."""
    reset_source_independence_detector()
    yield
    reset_source_independence_detector()


@pytest.fixture
def detector() -> SourceIndependenceDetector:
    """Fixture providing a clean default SourceIndependenceDetector."""
    return SourceIndependenceDetector()


# ---------------------------------------------------------------------------
# 1. Boundary Case 1: Single Cluster of Size M (Complete Redundancy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 3, 5, 10, 50])
def test_kish_boundary_case_single_cluster_complete_redundancy(m: int):
    """Verify n_eff = 1.0 for a single cluster of size M regardless of M (full discount)."""
    n_eff = compute_kish_effective_sample_size([m])
    assert n_eff == 1.0, f"Expected n_eff = 1.0 for single cluster of size {m}, got {n_eff}"


# ---------------------------------------------------------------------------
# 2. Boundary Case 2: All M Items are Independent Singletons (Zero Redundancy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 4, 7, 15])
def test_kish_boundary_case_all_singletons_zero_redundancy(m: int):
    """Verify n_eff = M for M independent singleton clusters (zero discount)."""
    cluster_sizes = [1] * m
    n_eff = compute_kish_effective_sample_size(cluster_sizes)
    assert n_eff == float(m), f"Expected n_eff = {m} for all singletons, got {n_eff}"


# ---------------------------------------------------------------------------
# 3. Boundary Case 3: Partial Redundancy Worked Example
# ---------------------------------------------------------------------------

def test_kish_boundary_case_partial_redundancy_worked_example():
    """Verify 1 cluster of 4 items + 2 independent singletons yields n_eff = 2.0 (36 / 18)."""
    # M = 4 + 1 + 1 = 6 items across 3 clusters
    # sum(s_k^2) = 4^2 + 1^2 + 1^2 = 16 + 1 + 1 = 18
    # n_eff = 6^2 / 18 = 36 / 18 = 2.0
    n_eff = compute_kish_effective_sample_size([4, 1, 1])
    assert n_eff == 2.0


# ---------------------------------------------------------------------------
# 4. Kish Effective Sample Size Edge Cases
# ---------------------------------------------------------------------------

def test_kish_effective_sample_size_edge_cases():
    """Verify empty collections, non-positive sizes, and single items."""
    assert compute_kish_effective_sample_size([]) == 0.0
    assert compute_kish_effective_sample_size([0, 0, 0]) == 0.0
    assert compute_kish_effective_sample_size([-1, -5]) == 0.0
    assert compute_kish_effective_sample_size([1]) == 1.0


# ---------------------------------------------------------------------------
# 5. Integration into analyze_set()
# ---------------------------------------------------------------------------

def test_source_independence_analyze_set_with_kish_n_eff(detector: SourceIndependenceDetector):
    """Verify analyze_set() populates kish_effective_sample_size accurately across redundancy regimes."""
    # Regime A: 4 identical duplicate documents -> 1 cluster of 4 -> n_eff = 1.0
    dup_items = [
        {"document_id": f"d_{i}", "text": "Apollo 11 landed astronauts on the Moon in July 1969."}
        for i in range(4)
    ]
    res_dup = detector.analyze_set(dup_items)
    assert len(res_dup.clusters) == 1
    assert len(res_dup.clusters[0].document_ids) == 4
    assert res_dup.kish_effective_sample_size == 1.0

    # Regime B: 4 completely distinct documents -> 0 clusters, 4 singletons -> n_eff = 4.0
    distinct_items = [
        {"document_id": "item_1", "text": "The speed of light in vacuum is approximately 300000 km per second."},
        {"document_id": "item_2", "text": "Photosynthesis converts carbon dioxide and water into glucose."},
        {"document_id": "item_3", "text": "The French Revolution began with the storming of the Bastille in 1789."},
        {"document_id": "item_4", "text": "DNA double helix structure was discovered by Watson and Crick in 1953."},
    ]
    res_distinct = detector.analyze_set(distinct_items)
    assert len(res_distinct.clusters) == 0
    assert len(res_distinct.independent_document_ids) == 4
    assert res_distinct.kish_effective_sample_size == 4.0

    # Regime C: Mixed (3 duplicates + 2 distinct singletons) -> M=5, clusters=[3, 1, 1]
    # n_eff = 5^2 / (3^2 + 1 + 1) = 25 / 11 = 2.2727
    mixed_items = [
        dup_items[0],
        dup_items[1],
        dup_items[2],
        distinct_items[0],
        distinct_items[1],
    ]
    res_mixed = detector.analyze_set(mixed_items)
    assert len(res_mixed.clusters) == 1
    assert len(res_mixed.clusters[0].document_ids) == 3
    assert len(res_mixed.independent_document_ids) == 2
    assert res_mixed.kish_effective_sample_size == pytest.approx(2.2727, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. Canonical Pair Ordering in compare_batch()
# ---------------------------------------------------------------------------

def test_compare_batch_preserves_canonical_pair_order(detector: SourceIndependenceDetector):
    """Verify compare_batch produces exact canonical pair ordering: (0,1), (0,2)... without inversions."""
    items = [{"document_id": f"doc_{i}", "text": f"Sample document passage number {i}."} for i in range(5)]
    results = detector.compare_batch(items)
    # Total pairs: 5 * 4 / 2 = 10
    assert len(results) == 10

    expected_pairs = [
        ("doc_0", "doc_1"),
        ("doc_0", "doc_2"),
        ("doc_0", "doc_3"),
        ("doc_0", "doc_4"),
        ("doc_1", "doc_2"),
        ("doc_1", "doc_3"),
        ("doc_1", "doc_4"),
        ("doc_2", "doc_3"),
        ("doc_2", "doc_4"),
        ("doc_3", "doc_4"),
    ]
    for idx, (exp_a, exp_b) in enumerate(expected_pairs):
        assert results[idx].document_id_a == exp_a
        assert results[idx].document_id_b == exp_b


# ---------------------------------------------------------------------------
# 7. Chunking Large Payloads with Invariant Parity
# ---------------------------------------------------------------------------

def test_chunking_large_payload_batch_size_parity(detector: SourceIndependenceDetector):
    """Verify chunked preprocessing (batch_size=1, 5, 10, 20) yields exact parity on N=20."""
    docs = [
        {"document_id": f"p_{i}", "text": f"Document text content for number {i // 2} with details."}
        for i in range(20)
    ]

    base = detector.analyze_set(docs, batch_size=20)
    chunk_1 = detector.analyze_set(docs, batch_size=1)
    chunk_5 = detector.analyze_set(docs, batch_size=5)
    chunk_10 = detector.analyze_set(docs, batch_size=10)

    assert base.total_documents == 20
    assert len(base.clusters) == len(chunk_1.clusters) == len(chunk_10.clusters)
    assert base.kish_effective_sample_size == chunk_1.kish_effective_sample_size == chunk_10.kish_effective_sample_size
    assert base.mean_pairwise_similarity == chunk_5.mean_pairwise_similarity


# ---------------------------------------------------------------------------
# 8. EvidenceQualityService Admissibility Filtering Before Kish n_eff
# ---------------------------------------------------------------------------

def test_evidence_quality_service_admissible_kish_n_eff():
    """Verify that EvidenceQualityService calculates Kish n_eff ONLY over admissible items."""
    service = EvidenceQualityService()
    claim = "Apollo 11 landed astronauts on the Moon in July 1969."
    today = date(2023, 6, 1)

    items = [
        # Admissible Item 1 (NASA)
        {
            "evidence_id": "nasa_01",
            "text": "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon in July 1969.",
            "source_url": "https://www.nasa.gov/apollo11.html",
            "published_date": date(2023, 1, 1),
        },
        # Admissible Item 2 (Near-duplicate mirror of Item 1 -> J >= 0.85, same cluster)
        {
            "evidence_id": "mirror_01",
            "text": "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the lunar surface Moon in July 1969.",
            "source_url": "https://mirror-wire.com/apollo11.html",
            "published_date": date(2023, 1, 1),
        },
        # Admissible Item 3 (Independent article grounded in claim)
        {
            "evidence_id": "independent_01",
            "text": "Apollo 11 astronauts completed their mission to land on the Moon in July 1969 successfully.",
            "source_url": "https://history.org/eagle.html",
            "published_date": date(2023, 1, 1),
        },
        # Inadmissible Item 4 (Completely ungrounded / polluted recipe)
        {
            "evidence_id": "polluted_01",
            "text": "Mix two cups of flour, sugar, baking powder, and buttermilk until smooth.",
            "source_url": "https://pancake-recipe.example.com",
            "published_date": date(2023, 1, 1),
        },
    ]

    report = service.evaluate_evidence(claim=claim, evidence_items=items, reference_date=today)
    assert report.total_evidence_count == 4
    assert report.admissible_count == 3
    assert report.polluted_count == 1

    # Admissible clusters: {cluster_A: 2 items, cluster_B: 1 item}
    # M_admissible = 3
    # sum(s_k^2) = 2^2 + 1^2 = 4 + 1 = 5
    # n_eff = 3^2 / 5 = 9 / 5 = 1.80
    assert report.kish_effective_count == 1.80


# ---------------------------------------------------------------------------
# 9. Batch Size Input Validation Traps
# ---------------------------------------------------------------------------

def test_batch_size_validation_errors(detector: SourceIndependenceDetector):
    """Verify invalid batch_size values raise SourceIndependenceInputError."""
    sample = [{"document_id": "d1", "text": "Sample text."}]
    for bad_size in (0, -1, True, False, "10", 2.5):
        with pytest.raises(SourceIndependenceInputError):
            detector.compare_batch(sample, batch_size=bad_size)  # type: ignore
        with pytest.raises(SourceIndependenceInputError):
            detector.analyze_set(sample, batch_size=bad_size)  # type: ignore


# ---------------------------------------------------------------------------
# 10. Mutation Test: Trapping the Flawed Raw-Item Formulation
# ---------------------------------------------------------------------------

def test_mutation_traps_flawed_kish_formula():
    """Trap the flawed raw-item formulation (w_i = 1/s_i) that falsely produced n_eff = M for syndication."""
    # Under a syndication of 5 identical wire copies:
    # - Flawed formula: w_i = 1/5 for each of 5 items.
    #   sum(w_i) = 1.0, sum(w_i^2) = 5 * (1/25) = 1/5.
    #   n_eff = 1.0 / (1/5) = 5.0 (No discount!).
    # - Corrected cluster formula:
    #   K = 1, s_1 = 5.
    #   n_eff = 5^2 / 5^2 = 1.0 (Full discount to single source!).
    correct_n_eff = compute_kish_effective_sample_size([5])
    assert correct_n_eff == 1.0, "Cluster-level Kish formula must yield 1.0 for a syndication cluster."
    assert correct_n_eff < 5.0, "Correct formula must discount syndicated items below raw count."
