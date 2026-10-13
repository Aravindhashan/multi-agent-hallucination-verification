"""Integration and Batch Tests for Dual-Perspective Debate Evidence Quality (Submodule 8.8).

Covers:
1. Dual-perspective independent clustering and Kish n_eff (separate supporting vs contradicting).
2. Cross-perspective joint analysis and overlap discount detection.
3. Boundary conditions (empty perspectives, one-sided debate, zero evidence).
4. Pollution gating within dual perspectives (purge ungrounded and circular items independently).
5. Memory-safe chunked batching parity across batch_size in {1, 2, 5}.
6. Batch input validation and error handling.
7. REST API endpoints: POST /api/v1/evidence/evaluate-debate-quality and evaluate-debate-quality-batch.
8. SessionStore SQLite persistence under session_claims.metadata_json['debate_quality'].
9. Integration with Module 7 DebateClaimResponse objects.
"""

from datetime import date, datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    DebateClaimResponse,
    DebateQualityBatchRequest,
    DebateQualityRequest,
    DebateQualityResponse,
    DebateRoundRecord,
    EvidenceItem,
)
from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.session_store import get_session_store


@pytest.fixture
def quality_service():
    """Return an EvidenceQualityService instance."""
    return get_evidence_quality_service()


@pytest.fixture
def client():
    """Return a FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# 1. Dual-Perspective Independent Clustering & Joint Kish n_eff
# ============================================================================


def test_dual_perspective_independent_clustering(quality_service):
    """Verify separate n_eff for supporting and contradicting pools plus joint n_eff."""
    claim = "The speed of light in vacuum is defined as exactly 299,792,458 metres per second."

    # Supporting pool: 3 syndicated copies of the same wire story
    supporting_items = [
        EvidenceItem(
            evidence_id=f"sup_wire_{i}",
            text=f"The speed of light in vacuum is defined as exactly 299,792,458 metres per second by international agreement. Wire variant {i}.",
            source_url=f"https://wire{i}.example.com/speed-of-light",
            domain=f"wire{i}.example.com",
            published_date=date(2023, 1, 1),
        )
        for i in range(1, 4)
    ]

    # Contradicting pool: 2 near-duplicate scientific challenge excerpts
    contradicting_items = [
        EvidenceItem(
            evidence_id="con_paper_a",
            text="Recent quantum gravity theories suggest photon dispersion might induce minute energy-dependent variations in the speed of light.",
            source_url="https://arxiv.example.org/abs/2101.0001",
            domain="arxiv.example.org",
            published_date=date(2023, 2, 1),
        ),
        EvidenceItem(
            evidence_id="con_paper_b",
            text="Recent quantum gravity hypotheses suggest photon dispersion might induce minute energy-dependent variations in the vacuum speed of light.",
            source_url="https://physrev.example.org/articles/10.1103/quantum",
            domain="physrev.example.org",
            published_date=date(2023, 2, 5),
        ),
    ]

    response = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=supporting_items,
        contradicting_evidence=contradicting_items,
        reference_date=date(2023, 3, 1),
    )

    # 1. Supporting pool verification
    assert response.supporting_report.total_raw_evidence_count == 3
    assert response.supporting_report.total_clusters == 1
    assert response.supporting_report.admissible_clusters == 1
    assert response.supporting_n_eff == 1.0  # complete redundancy discounted to 1.0
    assert len(response.pooled_supporting_evidence) == 1

    # 2. Contradicting pool verification
    assert response.contradicting_report.total_raw_evidence_count == 2
    assert response.contradicting_report.total_clusters == 1
    assert response.contradicting_report.admissible_clusters == 1
    assert response.contradicting_n_eff == 1.0  # near-duplicates discounted to 1.0
    assert len(response.pooled_contradicting_evidence) == 1

    # 3. Joint analysis verification
    # Across both pools: 3 items in cluster 1, 2 items in cluster 2 (M=5)
    # Kish joint = 5^2 / (3^2 + 2^2) = 25 / 13 = 1.92
    assert response.joint_n_eff == pytest.approx(1.92, abs=0.05)
    assert response.total_raw_evidence_count == 5
    assert response.total_admissible_count == 2
    assert response.total_pooled_representatives_count == 2

    # NLI call reduction: (5 - 2) / 5 = 60.0%
    assert response.overall_nli_call_reduction_percent == 60.0


def test_cross_perspective_redundancy_and_overlap_discount(quality_service):
    """Verify that cross-perspective overlap discount identifies when opponents retrieve duplicate evidence."""
    claim = "Exercise improves cardiovascular health in adults."

    # Supporting item
    sup_items = [
        EvidenceItem(
            evidence_id="sup_doc_1",
            text="Regular physical exercise significantly improves cardiovascular endurance and reduces cardiac morbidity in adult populations.",
            source_url="https://health.example.edu/exercise1",
            domain="health.example.edu",
        )
    ]

    # Case A: Orthogonal contradiction
    con_orthogonal = [
        EvidenceItem(
            evidence_id="con_doc_1",
            text="Excessive endurance exercise can impair cardiovascular health in adults with predisposed cardiac conditions.",
            source_url="https://sportsmed.example.edu/marathon",
            domain="sportsmed.example.edu",
        )
    ]

    resp_orthogonal = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=sup_items,
        contradicting_evidence=con_orthogonal,
    )

    # Both independent singletons -> joint n_eff = 2.0, overlap discount = 0.0
    assert resp_orthogonal.supporting_n_eff == 1.0
    assert resp_orthogonal.contradicting_n_eff == 1.0
    assert resp_orthogonal.joint_n_eff == 2.0
    assert resp_orthogonal.cross_perspective_overlap_discount == 0.0

    # Case B: Cross-perspective redundant mirror
    # Prosecutor retrieved the exact near-duplicate wire copy of the Defender's passage
    con_redundant = [
        EvidenceItem(
            evidence_id="con_mirror_1",
            text="Regular physical exercise significantly improves cardiovascular endurance and reduces cardiac morbidity in adult groups.",
            source_url="https://mirror.example.org/exercise1",
            domain="mirror.example.org",
        )
    ]

    resp_redundant = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=sup_items,
        contradicting_evidence=con_redundant,
    )

    # In separate pools, each had n_eff = 1.0 (sum = 2.0)
    # But jointly they form a single redundancy cluster of size 2 -> joint n_eff = 1.0!
    assert resp_redundant.joint_n_eff == 1.0
    # Overlap discount = (2.0 - 1.0) / 2.0 = 0.50 (50% redundancy discount!)
    assert resp_redundant.cross_perspective_overlap_discount == pytest.approx(0.50, abs=1e-4)


# ============================================================================
# 2. Boundary Conditions
# ============================================================================


def test_empty_perspective_boundaries(quality_service):
    """Verify behavior when one or both debate perspectives return zero evidence."""
    claim = "A single-sided factual claim."

    # Both empty
    resp_both_empty = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=[],
        contradicting_evidence=[],
    )
    assert resp_both_empty.total_raw_evidence_count == 0
    assert resp_both_empty.supporting_n_eff == 0.0
    assert resp_both_empty.contradicting_n_eff == 0.0
    assert resp_both_empty.joint_n_eff == 0.0
    assert resp_both_empty.cross_perspective_overlap_discount == 0.0
    assert resp_both_empty.overall_nli_call_reduction_percent == 0.0
    assert resp_both_empty.pooled_supporting_evidence == []
    assert resp_both_empty.pooled_contradicting_evidence == []

    # Supporting only
    sup_items = [
        EvidenceItem(
            evidence_id="sup_only_1",
            text="Document providing factual confirmation of the tested claim with external context.",
            source_url="https://doc.example.edu/1",
            domain="doc.example.edu",
        )
    ]
    resp_sup_only = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=sup_items,
        contradicting_evidence=[],
    )
    assert resp_sup_only.supporting_n_eff == 1.0
    assert resp_sup_only.contradicting_n_eff == 0.0
    assert resp_sup_only.joint_n_eff == 1.0
    assert resp_sup_only.cross_perspective_overlap_discount == 0.0
    assert len(resp_sup_only.pooled_supporting_evidence) == 1
    assert len(resp_sup_only.pooled_contradicting_evidence) == 0


def test_pollution_gating_in_dual_perspectives(quality_service):
    """Verify that pollution gates filter invalid evidence independently across both stances."""
    claim = "Photosynthesis requires chlorophyll pigments in chloroplasts."

    # Supporting has 1 clean document + 1 circular origin document
    sup_items = [
        EvidenceItem(
            evidence_id="clean_bio_sup",
            text="In plant biology, photosynthesis requires green chlorophyll pigments contained in plant chloroplasts.",
            source_url="https://nature.example.edu/bio",
            domain="nature.example.edu",
        ),
        EvidenceItem(
            evidence_id="circular_origin_sup",
            text="Photosynthesis requires chlorophyll pigments in chloroplasts.",
            source_url="https://origin.example.com/query",
            domain="origin.example.com",
        ),
    ]

    # Contradicting has 1 ungrounded recipe
    con_items = [
        EvidenceItem(
            evidence_id="ungrounded_con",
            text="Baking chocolate cookies uses two cups flour, one cup sugar, two fresh eggs, and vanilla extract in a large bowl.",
            source_url="https://recipes.example.com/bread",
            domain="recipes.example.com",
        )
    ]

    response = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=sup_items,
        contradicting_evidence=con_items,
        claim_origin_url="https://origin.example.com/query",
    )

    # Supporting: circular document rejected, clean document survives
    assert response.supporting_report.admissible_clusters == 1
    assert len(response.pooled_supporting_evidence) == 1
    assert response.pooled_supporting_evidence[0].evidence_id == "clean_bio_sup"

    # Contradicting: ungrounded recipe rejected, 0 admissible items
    assert response.contradicting_report.admissible_clusters == 0
    assert response.contradicting_n_eff == 0.0
    assert len(response.pooled_contradicting_evidence) == 0


# ============================================================================
# 3. Batch Testing & Chunking Parity
# ============================================================================


def test_debate_quality_batch_chunking_parity(quality_service):
    """Verify that evaluate_debate_quality_batch produces identical results across batch_size in {1, 2, 4}."""
    requests = [
        DebateQualityRequest(
            claim_text=f"Scientific statement number {i} regarding astrophysical principles.",
            supporting_evidence=[
                EvidenceItem(
                    evidence_id=f"sup_{i}_1",
                    text=f"Scientific statement number {i} regarding astrophysical principles is validated by telescope measurements.",
                    source_url=f"https://astro{i}.example.edu/paper1",
                    domain=f"astro{i}.example.edu",
                ),
                EvidenceItem(
                    evidence_id=f"sup_{i}_2",
                    text=f"Scientific statement number {i} regarding astrophysical principles is confirmed by telescope measurements.",
                    source_url=f"https://astro{i}.example.edu/paper2",
                    domain=f"astro{i}.example.edu",
                ),
            ],
            contradicting_evidence=[
                EvidenceItem(
                    evidence_id=f"con_{i}_1",
                    text=f"Scientific statement number {i} regarding astrophysical principles is challenged by cosmic microwave background anomalies.",
                    source_url=f"https://cosmo{i}.example.edu/paper1",
                    domain=f"cosmo{i}.example.edu",
                )
            ],
            claim_id=uuid.uuid4(),
        )
        for i in range(4)
    ]

    res_b1 = quality_service.evaluate_debate_quality_batch(requests, batch_size=1)
    res_b2 = quality_service.evaluate_debate_quality_batch(requests, batch_size=2)
    res_b4 = quality_service.evaluate_debate_quality_batch(requests, batch_size=4)

    assert len(res_b1) == len(res_b2) == len(res_b4) == 4

    for r1, r2, r4 in zip(res_b1, res_b2, res_b4):
        assert r1.claim_text == r2.claim_text == r4.claim_text
        assert r1.supporting_n_eff == r2.supporting_n_eff == r4.supporting_n_eff
        assert r1.contradicting_n_eff == r2.contradicting_n_eff == r4.contradicting_n_eff
        assert r1.joint_n_eff == r2.joint_n_eff == r4.joint_n_eff
        assert r1.cross_perspective_overlap_discount == r2.cross_perspective_overlap_discount == r4.cross_perspective_overlap_discount
        assert r1.overall_nli_call_reduction_percent == r2.overall_nli_call_reduction_percent == r4.overall_nli_call_reduction_percent
        assert len(r1.pooled_supporting_evidence) == len(r2.pooled_supporting_evidence) == len(r4.pooled_supporting_evidence)
        assert len(r1.pooled_contradicting_evidence) == len(r2.pooled_contradicting_evidence) == len(r4.pooled_contradicting_evidence)


def test_batch_validation_error_handling(quality_service):
    """Verify validation of batch parameters and input types."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        quality_service.evaluate_debate_quality_batch([], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        quality_service.evaluate_debate_quality_batch([], batch_size=-3)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        quality_service.evaluate_debate_quality_batch([], batch_size=True)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        quality_service.evaluate_debate_quality_batch([], batch_size="10")

    # Empty batch returns empty list
    assert quality_service.evaluate_debate_quality_batch([]) == []


# ============================================================================
# 4. REST API & SessionStore Integration
# ============================================================================


def test_api_evaluate_debate_quality_endpoints(client):
    """Verify POST /api/v1/evidence/evaluate-debate-quality and batch endpoint."""
    payload = {
        "claim_text": "Penicillin was discovered by Alexander Fleming in 1928.",
        "supporting_evidence": [
            {
                "evidence_id": "pen_sup_1",
                "text": "Scottish physician Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital London from contaminated laboratory bacterial culture samples.",
                "source_url": "https://historyofmed.example.edu/fleming1",
                "domain": "historyofmed.example.edu",
            },
            {
                "evidence_id": "pen_sup_2",
                "text": "Scottish physician Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital London from contaminated laboratory bacterial culture plates.",
                "source_url": "https://historyofmed.example.edu/fleming2",
                "domain": "historyofmed.example.edu",
            },
        ],
        "contradicting_evidence": [
            {
                "evidence_id": "pen_con_1",
                "text": "French medical student Ernest Duchesne originally noted the antibacterial properties of penicillium moulds in 1897 prior to Fleming.",
                "source_url": "https://duchesne.example.fr/discovery",
                "domain": "duchesne.example.fr",
            }
        ],
    }

    resp = client.post("/api/v1/evidence/evaluate-debate-quality", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["claim_text"] == "Penicillin was discovered by Alexander Fleming in 1928."
    assert data["total_raw_evidence_count"] == 3
    assert data["supporting_n_eff"] == 1.0
    assert data["contradicting_n_eff"] == 1.0
    assert len(data["pooled_supporting_evidence"]) == 1
    assert len(data["pooled_contradicting_evidence"]) == 1
    assert data["overall_nli_call_reduction_percent"] > 0.0

    # Batch endpoint
    batch_payload = {"requests": [payload, payload]}
    batch_resp = client.post("/api/v1/evidence/evaluate-debate-quality-batch", json=batch_payload)
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert len(batch_data) == 2


def test_session_store_debate_quality_persistence(quality_service):
    """Verify session_store.record_debate_quality_audit persists dual n_eff and metrics in SQLite."""
    query_id = uuid.uuid4()
    claim_id = uuid.uuid4()

    claim = "The Great Barrier Reef is the world's largest coral reef system."
    sup_items = [
        EvidenceItem(
            evidence_id="gbr_1",
            text="The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs.",
            source_url="https://marine.example.gov.au/gbr",
            domain="marine.example.gov.au",
        )
    ]
    con_items = [
        EvidenceItem(
            evidence_id="gbr_con_1",
            text="Mass coral bleaching events have severely degraded substantial sections of the northern Great Barrier Reef.",
            source_url="https://climate.example.edu/bleaching",
            domain="climate.example.edu",
        )
    ]

    response = quality_service.evaluate_debate_quality(
        claim=claim,
        supporting_evidence=sup_items,
        contradicting_evidence=con_items,
        query_id=query_id,
        claim_id=claim_id,
    )

    store = get_session_store()
    records = store.get_claim_records(query_id)
    assert len(records) == 1

    import json
    with store._lock:
        cur = store._conn.execute(
            "SELECT metadata_json FROM session_claims WHERE query_id = ? AND claim_id = ?",
            (str(query_id), str(claim_id)),
        )
        row = cur.fetchone()
        assert row is not None
        meta = json.loads(row[0])
        assert "debate_quality" in meta
        dq_meta = meta["debate_quality"]
        assert dq_meta["supporting_raw_count"] == 1
        assert dq_meta["supporting_n_eff"] == 1.0
        assert dq_meta["contradicting_raw_count"] == 1
        assert dq_meta["contradicting_n_eff"] == 1.0
        assert dq_meta["joint_n_eff"] == 2.0


def test_evaluate_debate_claim_response_convenience(quality_service):
    """Verify convenience method evaluate_debate_claim_response consuming Module 7 outputs."""
    debate_response = DebateClaimResponse(
        claim_id=uuid.uuid4(),
        query_id=uuid.uuid4(),
        claim_text="The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        rounds=[
            DebateRoundRecord(
                round_number=1,
                defender_query="Pacific Ocean largest ocean",
                prosecutor_query="Pacific ocean size challenges",
            )
        ],
        final_supporting_evidence=[
            EvidenceItem(
                evidence_id="pac_sup_1",
                text="The Pacific Ocean is the largest and deepest of Earth's oceanic divisions extending from the Arctic to Antarctic.",
                source_url="https://ocean.example.gov/pacific",
                domain="ocean.example.gov",
            )
        ],
        final_contradicting_evidence=[
            EvidenceItem(
                evidence_id="pac_con_1",
                text="Oceanographic surveys challenge whether the Pacific Ocean remains the deepest and largest of Earth's oceanic divisions across all tectonic epochs.",
                source_url="https://geography.example.edu/oceans",
                domain="geography.example.edu",
            )
        ],
        debated_at=datetime.now(timezone.utc),
    )

    quality_resp = quality_service.evaluate_debate_claim_response(debate_response)

    assert quality_resp.claim_id == debate_response.claim_id
    assert quality_resp.query_id == debate_response.query_id
    assert quality_resp.claim_text == debate_response.claim_text
    assert quality_resp.supporting_n_eff == 1.0
    assert quality_resp.contradicting_n_eff == 1.0
    assert len(quality_resp.pooled_supporting_evidence) == 1
    assert len(quality_resp.pooled_contradicting_evidence) == 1
