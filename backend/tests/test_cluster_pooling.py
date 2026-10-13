"""Tests for Cluster-Level Log-Odds Pooling (Submodule 8.7).

Covers:
1. Reviewer §4 Reconciliation: strictly NO (1.0 + 0.10 * ln|C_k|) corroboration bonus.
2. Representative Selection: argmax_{e in C_k, adm} S_comp(e) where S_comp = C * F * (1 - P).
3. Deterministic tie-breaking across identical copies.
4. Fully polluted cluster handling (zero weight, excluded from pooled representatives).
5. Partially polluted cluster with admissible fallback.
6. Boundary cases: singletons, monolithic redundancy, mixed clusters, Kish n_eff consistency.
7. Empty evidence sets.
8. Log-odds mathematical primitives, invertibility, stability, and domain validation.
9. Weighted log-odds combination and NLI probability pooling.
10. Memory-safe chunked batching parity across batch_size in {1, 2, 5}.
11. Batch input validation and error handling.
12. API endpoints: POST /api/v1/evidence/pool-clusters and /pool-clusters-batch.
13. SessionStore audit persistence under session_claims.metadata_json['cluster_pooling'].
"""

from datetime import date, datetime, timezone
import math
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    EvidenceItem,
    EvidenceQualityRequest,
    PooledClusterBatchRequest,
    PooledClusterReport,
)
from app.services.cluster_pooling import (
    ClusterPoolingService,
    combine_log_odds,
    combine_nli_probabilities,
    compute_intrinsic_composite_score,
    get_cluster_pooling_service,
    log_odds_to_prob,
    prob_to_log_odds,
    select_cluster_representative,
)
from app.services.session_store import get_session_store


@pytest.fixture
def service():
    """Return a fresh ClusterPoolingService instance."""
    return ClusterPoolingService()


@pytest.fixture
def client():
    """Return a FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# 1. Reviewer §4 Reconciliation & Mathematical Correctness
# ============================================================================


def test_reviewer_section_4_no_corroboration_bonus(service):
    """Verify strictly NO (1.0 + 0.10 * ln|C_k|) corroboration bonus is applied (Reviewer §4).

    Syndicated wire copies represent mechanical repetition rather than independent
    confirmation. Cluster reliability must strictly equal S_comp(e_hat_k) without
    cluster-size inflation.
    """
    claim = "Water freezes at 0 degrees Celsius under standard atmospheric pressure."

    # 5 syndicated copies of the exact same wire article
    evidence_items = [
        {
            "evidence_id": f"wire_item_{i}",
            "text": f"Water freezes at 0 degrees Celsius under standard atmospheric pressure. Syndicate report {i}.",
            "source_url": f"https://syndicate{i}.example.com/physics/water",
            "domain": f"syndicate{i}.example.com",
            "published_date": date(2023, 1, 1),
        }
        for i in range(1, 6)
    ]

    report = service.pool_clusters(
        claim=claim,
        evidence_items=evidence_items,
        reference_date=date(2023, 1, 1),
    )

    assert report.total_raw_evidence_count == 5
    assert report.total_clusters == 1
    assert report.admissible_clusters == 1
    cluster = report.clusters[0]

    assert cluster.cluster_size == 5
    rep_cred = cluster.credibility_score
    rep_fresh = cluster.freshness_score
    rep_poll = cluster.pollution_score
    expected_s_comp = compute_intrinsic_composite_score(rep_cred, rep_fresh, rep_poll)

    # Reviewer §4 reconciliation assertion:
    # W(C_k) == S_comp(e_hat_k)
    assert cluster.cluster_reliability_weight == pytest.approx(expected_s_comp, abs=1e-4)

    # Assert that NO (1.0 + 0.10 * ln(5)) bonus was applied!
    log_bonus = 1.0 + 0.10 * math.log(5)
    inflated_weight = expected_s_comp * log_bonus
    assert cluster.cluster_reliability_weight < inflated_weight
    assert cluster.cluster_reliability_weight <= 1.0


def test_representative_selection_highest_composite_score(service):
    """Verify that argmax S_comp selects the highest-quality member as cluster representative."""
    claim = "The James Webb Space Telescope was launched in December 2021."

    # 3 items on the same topic forming a single redundancy cluster, but with different source quality
    evidence_items = [
        # Item A: Established domain (Tier 2, credibility ~0.75)
        {
            "evidence_id": "news_outlet_a",
            "text": "The James Webb Space Telescope was launched into space on an Ariane 5 rocket from French Guiana in December 2021.",
            "source_url": "https://reuters.example.com/science/jwst",
            "domain": "reuters.example.com",
            "published_date": date(2022, 1, 1),
        },
        # Item B: Authoritative NASA source (.gov, Tier 1, credibility >= 0.85)
        {
            "evidence_id": "nasa_official",
            "text": "The James Webb Space Telescope was launched into orbit on an Ariane 5 rocket from French Guiana in December 2021.",
            "source_url": "https://nasa.gov/mission_pages/webb/launch",
            "domain": "nasa.gov",
            "published_date": date(2022, 1, 1),
        },
        # Item C: Stale blog (.com, lower credibility)
        {
            "evidence_id": "blog_c",
            "text": "The James Webb Space Telescope was launched into space on an Ariane 5 rocket from Kourou French Guiana in December 2021.",
            "source_url": "https://astroblog.example.com/jwst",
            "domain": "astroblog.example.com",
            "published_date": date(2022, 1, 1),
        },
    ]

    report = service.pool_clusters(
        claim=claim,
        evidence_items=evidence_items,
        reference_date=date(2022, 1, 1),
    )

    assert report.total_clusters == 1
    cluster = report.clusters[0]

    # NASA official must be selected because it has the highest credibility and composite score
    assert cluster.representative_id == "nasa_official"
    assert "nasa.gov" in cluster.domain
    assert cluster.cluster_size == 3
    assert cluster.cluster_reliability_weight >= 0.85
    assert len(report.pooled_representatives) == 1
    assert report.pooled_representatives[0].evidence_id == "nasa_official"


def test_deterministic_tie_breaking(service):
    """Verify deterministic tie-breaking produces identical selections regardless of input order."""
    claim = "Mars has two small natural satellites, Phobos and Deimos."

    items_order_1 = [
        {
            "evidence_id": "doc_beta",
            "text": "Mars has two small natural moons named Phobos and Deimos orbiting around the planet.",
            "source_url": "https://science.example.edu/mars1",
            "domain": "science.example.edu",
            "published_date": date(2023, 1, 1),
        },
        {
            "evidence_id": "doc_alpha",
            "text": "Mars has two small natural moons named Phobos and Deimos orbiting near the planet.",
            "source_url": "https://science.example.edu/mars2",
            "domain": "science.example.edu",
            "published_date": date(2023, 1, 1),
        },
    ]

    items_order_2 = list(reversed(items_order_1))

    rep1 = service.pool_clusters(claim=claim, evidence_items=items_order_1)
    rep2 = service.pool_clusters(claim=claim, evidence_items=items_order_2)

    assert rep1.clusters[0].representative_id == rep2.clusters[0].representative_id
    assert rep1.clusters[0].cluster_reliability_weight == rep2.clusters[0].cluster_reliability_weight


# ============================================================================
# 2. Pollution & Admissibility Handling
# ============================================================================


def test_fully_polluted_cluster_handling(service):
    """Verify a cluster composed entirely of polluted/ungrounded items receives zero weight."""
    claim = "DNA contains four nucleotide bases: adenine, cytosine, guanine, and thymine."

    polluted_items = [
        {
            "evidence_id": "polluted_cookie_recipe_1",
            "text": "Baking chocolate cookies requires whisking two cups flour, one cup sugar, two fresh eggs, and vanilla extract together in a large bowl.",
            "source_url": "https://recipes.example.com/cookies1",
            "domain": "recipes.example.com",
            "published_date": date(2023, 1, 1),
        },
        {
            "evidence_id": "polluted_cookie_recipe_2",
            "text": "Baking chocolate cookies requires whisking two cups flour, one cup sugar, two fresh eggs, and vanilla extract together in a mixing bowl.",
            "source_url": "https://recipes.example.com/cookies2",
            "domain": "recipes.example.com",
            "published_date": date(2023, 1, 1),
        },
    ]

    report = service.pool_clusters(claim=claim, evidence_items=polluted_items)

    assert report.total_clusters == 1
    assert report.admissible_clusters == 0
    assert report.polluted_clusters == 1

    cluster = report.clusters[0]
    assert cluster.is_admissible is False
    assert cluster.cluster_reliability_weight == 0.0
    assert "Rejected" in cluster.admissibility_reason

    # Representative should NOT be passed downstream to Module 9 NLI
    assert len(report.pooled_representatives) == 0


def test_partially_polluted_cluster_admissible_fallback(service):
    """Verify that a cluster containing both clean and polluted members selects an admissible member."""
    claim = "Photosynthesis converts solar light energy into chemical energy in plants."

    items = [
        # Clean scientific explanation
        {
            "evidence_id": "clean_biology",
            "text": "Photosynthesis converts solar light energy into chemical energy stored in carbohydrates in plants.",
            "source_url": "https://nature.example.edu/biology/photosynthesis",
            "domain": "nature.example.edu",
            "published_date": date(2023, 1, 1),
        },
        # Syndicated copy polluted by self-referential query origin
        {
            "evidence_id": "syndicated_copy",
            "text": "Photosynthesis converts solar light energy into chemical energy stored in carbohydrates in plants.",
            "source_url": "https://myquery.example.com/origin",
            "domain": "myquery.example.com",
            "published_date": date(2023, 1, 1),
        },
    ]

    report = service.pool_clusters(
        claim=claim,
        evidence_items=items,
        claim_origin_url="https://myquery.example.com/origin",  # marks syndicated_copy as circular
    )

    assert report.total_clusters == 1
    assert report.admissible_clusters == 1
    cluster = report.clusters[0]

    assert cluster.is_admissible is True
    assert cluster.representative_id == "clean_biology"
    assert cluster.cluster_reliability_weight > 0.0
    assert len(report.pooled_representatives) == 1
    assert report.pooled_representatives[0].evidence_id == "clean_biology"


# ============================================================================
# 3. Boundary Cases & Set Topology
# ============================================================================


def test_boundary_cases_singletons_and_monolithic(service):
    """Verify behavior on monolithic redundancy, pure singletons, and mixed sets."""
    claim = "The atomic number of Carbon is 6."

    # Monolithic redundancy: 4 copies of same text
    mono_items = [
        {
            "evidence_id": f"mono_{i}",
            "text": "Carbon is a chemical element with symbol C and atomic number 6 in the periodic table.",
            "source_url": f"https://chem{i}.example.edu/carbon",
            "domain": f"chem{i}.example.edu",
            "published_date": date(2023, 1, 1),
        }
        for i in range(4)
    ]
    mono_report = service.pool_clusters(claim=claim, evidence_items=mono_items)
    assert mono_report.total_raw_evidence_count == 4
    assert mono_report.total_clusters == 1
    assert mono_report.admissible_clusters == 1
    assert mono_report.kish_effective_count == 1.0
    assert mono_report.nli_call_reduction_percent == 75.0
    assert len(mono_report.pooled_representatives) == 1

    # Pure singletons: 3 distinct topics
    singleton_items = [
        {
            "evidence_id": "carbon_doc",
            "text": "Carbon is a chemical element with symbol C and atomic number 6.",
            "source_url": "https://chem.example.edu/c",
            "domain": "chem.example.edu",
        },
        {
            "evidence_id": "diamond_doc",
            "text": "Diamond is a solid form of the pure element carbon with its atoms arranged in a crystal.",
            "source_url": "https://geology.example.edu/diamond",
            "domain": "geology.example.edu",
        },
        {
            "evidence_id": "graphite_doc",
            "text": "Graphite is another crystalline form of the element carbon found naturally in metamorphic rocks.",
            "source_url": "https://mineral.example.edu/graphite",
            "domain": "mineral.example.edu",
        },
    ]
    singleton_report = service.pool_clusters(claim=claim, evidence_items=singleton_items)
    assert singleton_report.total_raw_evidence_count == 3
    assert singleton_report.total_clusters == 3
    assert singleton_report.admissible_clusters == 3
    assert singleton_report.kish_effective_count == 3.0
    assert singleton_report.nli_call_reduction_percent == 0.0
    assert len(singleton_report.pooled_representatives) == 3


def test_empty_evidence_set(service):
    """Verify graceful handling of empty evidence input."""
    claim = "No evidence retrieved."
    report = service.pool_clusters(claim=claim, evidence_items=[])

    assert report.total_raw_evidence_count == 0
    assert report.total_clusters == 0
    assert report.admissible_clusters == 0
    assert report.polluted_clusters == 0
    assert report.kish_effective_count == 0.0
    assert report.nli_call_reduction_percent == 0.0
    assert report.clusters == []
    assert report.pooled_representatives == []


# ============================================================================
# 4. Mathematical Log-Odds Primitives
# ============================================================================


def test_log_odds_conversion_and_invertibility():
    """Verify log-odds conversions, numerical stability, and invertibility."""
    # 50/50 prior has log-odds 0.0
    assert prob_to_log_odds(0.5) == pytest.approx(0.0, abs=1e-6)
    assert log_odds_to_prob(0.0) == pytest.approx(0.5, abs=1e-6)

    # Invertibility check over test grid
    for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        lo = prob_to_log_odds(p)
        recovered_p = log_odds_to_prob(lo)
        assert recovered_p == pytest.approx(p, abs=1e-4)

    # Numerical boundary protection (no OverflowError or MathDomainError)
    lo_zero = prob_to_log_odds(0.0)
    lo_one = prob_to_log_odds(1.0)
    assert lo_zero < -10.0
    assert lo_one > 10.0

    p_pos_inf = log_odds_to_prob(500.0)
    p_neg_inf = log_odds_to_prob(-500.0)
    assert p_pos_inf == pytest.approx(1.0, abs=1e-5)
    assert p_neg_inf == pytest.approx(0.0, abs=1e-5)

    # Error handling on invalid inputs
    with pytest.raises(ValueError):
        prob_to_log_odds(-0.1)
    with pytest.raises(ValueError):
        prob_to_log_odds(1.1)
    with pytest.raises(ValueError):
        prob_to_log_odds(float("nan"))
    with pytest.raises(ValueError):
        log_odds_to_prob(float("nan"))


def test_weighted_log_odds_combination():
    """Verify Bayesian log-odds combination under various weight configurations."""
    # Flat uninformative prior L_0 = 0.0, unit weights
    log_odds = [1.5, 2.0]
    weights = [1.0, 1.0]
    assert combine_log_odds(log_odds, weights) == pytest.approx(3.5, abs=1e-6)

    # Zero weights (e.g. polluted clusters) produce zero update
    zero_weights = [0.0, 0.0]
    assert combine_log_odds(log_odds, zero_weights) == pytest.approx(0.0, abs=1e-6)

    # Partial reliability weights
    partial_weights = [0.8, 0.5]
    expected = 0.8 * 1.5 + 0.5 * 2.0  # 1.2 + 1.0 = 2.2
    assert combine_log_odds(log_odds, partial_weights) == pytest.approx(expected, abs=1e-6)

    # Informative prior L_0 = 1.0
    prior_lo = 1.0
    # Update = 1.0 + 0.5 * (2.0 - 1.0) + 0.5 * (3.0 - 1.0) = 1.0 + 0.5 + 1.0 = 2.5
    assert combine_log_odds([2.0, 3.0], [0.5, 0.5], prior_log_odds=prior_lo) == pytest.approx(2.5, abs=1e-6)

    # Length mismatch raises ValueError
    with pytest.raises(ValueError):
        combine_log_odds([1.0], [1.0, 0.5])

    # Negative weight raises ValueError
    with pytest.raises(ValueError):
        combine_log_odds([1.0], [-0.5])


def test_nli_probabilities_log_odds_pooling():
    """Verify combination of multi-cluster NLI distributions with cluster reliability weights."""
    # Strong supporting clusters
    supporting_probs = [
        (0.90, 0.05, 0.05),
        (0.85, 0.05, 0.10),
    ]
    weights = [0.90, 0.80]

    result = combine_nli_probabilities(supporting_probs, weights)
    assert result["net_direction"] == "SUPPORTING"
    assert result["posterior_support_probability"] > 0.90
    assert result["posterior_contradiction_probability"] < 0.10

    # Strong contradicting clusters
    contradicting_probs = [
        (0.05, 0.92, 0.03),
        (0.02, 0.95, 0.03),
    ]
    result_con = combine_nli_probabilities(contradicting_probs, weights)
    assert result_con["net_direction"] == "CONTRADICTING"
    assert result_con["posterior_contradiction_probability"] > 0.90


# ============================================================================
# 5. Batch Testing & Chunking Parity
# ============================================================================


def test_pool_clusters_batch_chunking_parity(service):
    """Verify that pool_clusters_batch produces 100% identical results across batch_size in {1, 2, 4}."""
    requests = [
        EvidenceQualityRequest(
            claim_text=f"Factual claim number {i} regarding scientific observations.",
            evidence_items=[
                EvidenceItem(
                    evidence_id=f"doc_{i}_1",
                    text=f"Factual claim number {i} regarding scientific observations is validated.",
                    source_url=f"https://science{i}.example.edu/doc1",
                    domain=f"science{i}.example.edu",
                ),
                EvidenceItem(
                    evidence_id=f"doc_{i}_2",
                    text=f"Factual claim number {i} regarding scientific observations is fully validated.",
                    source_url=f"https://science{i}.example.edu/doc2",
                    domain=f"science{i}.example.edu",
                ),
            ],
            claim_id=uuid.uuid4(),
        )
        for i in range(4)
    ]

    res_b1 = service.pool_clusters_batch(requests, batch_size=1)
    res_b2 = service.pool_clusters_batch(requests, batch_size=2)
    res_b4 = service.pool_clusters_batch(requests, batch_size=4)

    assert len(res_b1) == len(res_b2) == len(res_b4) == 4

    for r1, r2, r4 in zip(res_b1, res_b2, res_b4):
        assert r1.claim_text == r2.claim_text == r4.claim_text
        assert r1.total_clusters == r2.total_clusters == r4.total_clusters
        assert r1.admissible_clusters == r2.admissible_clusters == r4.admissible_clusters
        assert r1.kish_effective_count == r2.kish_effective_count == r4.kish_effective_count
        assert r1.nli_call_reduction_percent == r2.nli_call_reduction_percent == r4.nli_call_reduction_percent
        assert len(r1.pooled_representatives) == len(r2.pooled_representatives) == len(r4.pooled_representatives)


def test_pool_clusters_batch_validation_errors(service):
    """Verify validation of batch parameters and input types."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        service.pool_clusters_batch([], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        service.pool_clusters_batch([], batch_size=-5)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        service.pool_clusters_batch([], batch_size=True)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        service.pool_clusters_batch([], batch_size="5")

    # Empty batch returns empty list
    assert service.pool_clusters_batch([]) == []


# ============================================================================
# 6. API Endpoints & SessionStore Integration
# ============================================================================


def test_api_endpoints_pool_clusters(client):
    """Verify POST /api/v1/evidence/pool-clusters and /pool-clusters-batch."""
    payload = {
        "claim_text": "Helium is lighter than air and non-flammable.",
        "evidence_items": [
            {
                "evidence_id": "he_1",
                "text": "Helium is lighter than air and non-flammable under normal atmospheric conditions.",
                "source_url": "https://chem.example.edu/helium1",
                "domain": "chem.example.edu",
            },
            {
                "evidence_id": "he_2",
                "text": "Helium is lighter than air and non-flammable under normal atmospheric conditions.",
                "source_url": "https://chem.example.edu/helium2",
                "domain": "chem.example.edu",
            },
        ],
    }

    resp = client.post("/api/v1/evidence/pool-clusters", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["claim_text"] == "Helium is lighter than air and non-flammable."
    assert data["total_raw_evidence_count"] == 2
    assert data["total_clusters"] == 1
    assert data["admissible_clusters"] == 1
    assert data["nli_call_reduction_percent"] == 50.0
    assert len(data["pooled_representatives"]) == 1

    # Batch endpoint
    batch_payload = {"requests": [payload, payload]}
    batch_resp = client.post("/api/v1/evidence/pool-clusters-batch", json=batch_payload)
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert len(batch_data) == 2


def test_session_store_cluster_pooling_persistence(service):
    """Verify session_store.record_cluster_pooling_audit updates metadata_json."""
    query_id = uuid.uuid4()
    claim_id = uuid.uuid4()

    claim = "Gold has the chemical symbol Au and atomic number 79."
    evidence_items = [
        {
            "evidence_id": "gold_1",
            "text": "In science and chemistry, gold is an elemental heavy metal with the atomic number 79 and chemical symbol Au.",
            "source_url": "https://chem.example.edu/gold",
            "domain": "chem.example.edu",
        }
    ]

    report = service.pool_clusters(
        claim=claim,
        evidence_items=evidence_items,
        query_id=query_id,
        claim_id=claim_id,
    )

    store = get_session_store()
    records = store.get_claim_records(query_id)
    assert len(records) == 1
    claim_rec = records[0]

    import json
    # Query database directly for metadata_json
    with store._lock:
        cur = store._conn.execute(
            "SELECT metadata_json FROM session_claims WHERE query_id = ? AND claim_id = ?",
            (str(query_id), str(claim_id)),
        )
        row = cur.fetchone()
        assert row is not None
        meta = json.loads(row[0])
        assert "cluster_pooling" in meta
        pooling_meta = meta["cluster_pooling"]
        assert pooling_meta["total_raw_evidence_count"] == 1
        assert pooling_meta["total_clusters"] == 1
        assert pooling_meta["admissible_clusters"] == 1


# ============================================================================
# 5. Reviewer §3 Reconciliation: Retrieval Ranking vs Channel Reliability
# ============================================================================


def test_retrieval_ranking_vs_intrinsic_channel_reliability_reconciliation(service):
    """Verify architectural reconciliation between S_retrieval and S_intrinsic (Reviewer Assessment §3).

    1. S_retrieval = 0.5 * rel + 0.3 * cred + 0.2 * fresh (Modules 5-7 open-corpus ranking).
    2. S_intrinsic = cred * fresh * (1 - poll) (Modules 8.7-10 Bayesian channel reliability).
    3. Within-cluster tie-breaking: between copies with identical intrinsic scores,
       the copy with higher retrieval relevance_score is selected as representative.
    """
    claim = "Photosynthesis converts solar energy into chemical energy in green plants."

    # Two identical wire reports from the same domain tier and date, but different retrieval relevance
    items = [
        {
            "evidence_id": "copy_low_relevance",
            "text": "Photosynthesis converts solar light into chemical energy inside green plant chloroplasts. Dispatch variant alpha.",
            "source_url": "https://wire.example.org/biology/1",
            "domain": "wire.example.org",
            "published_date": date(2023, 1, 1),
            "relevance_score": 0.60,
        },
        {
            "evidence_id": "copy_high_relevance",
            "text": "Photosynthesis converts solar light into chemical energy inside green plant chloroplasts. Dispatch variant beta.",
            "source_url": "https://wire.example.org/biology/2",
            "domain": "wire.example.org",
            "published_date": date(2023, 1, 1),
            "relevance_score": 0.95,
        },
    ]

    report = service.pool_clusters(claim=claim, evidence_items=items, reference_date=date(2023, 1, 1))

    assert report.total_clusters == 1
    cluster = report.clusters[0]

    # Highest relevance copy must win the tie-break
    assert cluster.representative_id == "copy_high_relevance"

    # Formula comparison audit
    cred = cluster.credibility_score
    fresh = cluster.freshness_score
    poll = cluster.pollution_score
    rel = 0.95

    s_retrieval = 0.5 * rel + 0.3 * cred + 0.2 * fresh
    s_intrinsic = compute_intrinsic_composite_score(cred, fresh, poll)

    # Intentionally distinct: W(C_k) strictly equals S_intrinsic, not S_retrieval
    assert cluster.cluster_reliability_weight == pytest.approx(s_intrinsic, abs=1e-4)
    assert cluster.cluster_reliability_weight != pytest.approx(s_retrieval, abs=1e-4)

