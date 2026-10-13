"""Tests for unified evidence quality batch processing and API endpoints (Submodule 8.5).

Covers:
1. Single claim /evaluate-quality endpoint with anchor-awareness and Kish n_eff.
2. Batch endpoint /evaluate-quality-batch with list payload and wrapper payload.
3. Strict 1-to-1 input order preservation across claims and items.
4. Memory-safe chunking parity across batch sizes (1, 2, 3, 5, 10).
5. Empty batch payload handling ([] and {"requests": []}).
6. Sibling claim SQLite session_claims quality audit isolation under shared query_id.
7. Input validation error handling (whitespace claim, empty evidence, invalid batch_size).
8. Boundary disclaimer presence.
"""

from datetime import date
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EvidenceItem, EvidenceQualityBatchRequest, EvidenceQualityRequest
from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.session_store import get_session_store, reset_session_store
from app.services.source_independence_config import DISCLAIMER_PHRASE


@pytest.fixture(autouse=True)
def clean_state():
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    return TestClient(app)


def _make_evidence_item(
    doc_id: str,
    text: str,
    url: str = "https://example.gov/doc",
    published_date: str = "2023-01-01",
) -> dict:
    return {
        "evidence_id": doc_id,
        "text": text,
        "source_url": url,
        "domain": "example.gov",
        "source_title": "Example Gov Source",
        "published_date": published_date,
        "retrieval_method": "hybrid",
        "relevance_score": 0.9,
    }


def test_batch_endpoint_list_and_wrapper_payloads(client):
    """Verify POST /api/v1/evidence/evaluate-quality-batch accepts both list and wrapper payloads."""
    claim1 = "The Hubble Space Telescope was launched in 1990."
    claim2 = "The Voyager 1 probe entered interstellar space in 2012."

    req1 = {
        "claim_text": claim1,
        "claim_id": "c1",
        "evidence_items": [
            _make_evidence_item("h1", "Hubble Space Telescope was launched into low Earth orbit in 1990."),
            _make_evidence_item("h2", "NASA launched the Hubble Space Telescope in April 1990."),
        ],
    }
    req2 = {
        "claim_text": claim2,
        "claim_id": "c2",
        "evidence_items": [
            _make_evidence_item("v1", "Voyager 1 crossed the heliopause into interstellar space in August 2012."),
        ],
    }

    # Test 1: Direct list payload [...]
    resp_list = client.post("/api/v1/evidence/evaluate-quality-batch", json=[req1, req2])
    assert resp_list.status_code == 200
    data_list = resp_list.json()
    assert len(data_list) == 2
    assert data_list[0]["claim_id"] == "c1"
    assert data_list[0]["claim_text"] == claim1
    assert data_list[0]["total_evidence_count"] == 2
    assert data_list[1]["claim_id"] == "c2"
    assert data_list[1]["claim_text"] == claim2
    assert data_list[1]["total_evidence_count"] == 1

    # Test 2: Wrapper object payload {"requests": [...]}
    resp_wrap = client.post("/api/v1/evidence/evaluate-quality-batch", json={"requests": [req1, req2]})
    assert resp_wrap.status_code == 200
    data_wrap = resp_wrap.json()
    assert len(data_wrap) == 2
    assert data_wrap[0]["claim_id"] == "c1"
    assert data_wrap[1]["claim_id"] == "c2"

    # Verify semantic field equality between both invocation formats
    assert len(data_list) == len(data_wrap)
    for d_l, d_w in zip(data_list, data_wrap):
        assert d_l["claim_id"] == d_w["claim_id"]
        assert d_l["claim_text"] == d_w["claim_text"]
        assert d_l["total_evidence_count"] == d_w["total_evidence_count"]
        assert d_l["admissible_count"] == d_w["admissible_count"]
        assert d_l["polluted_count"] == d_w["polluted_count"]
        assert d_l["cluster_count"] == d_w["cluster_count"]
        assert d_l["kish_effective_count"] == d_w["kish_effective_count"]
        assert d_l["evaluated_items"] == d_w["evaluated_items"]
        assert d_l["disclaimer"] == d_w["disclaimer"]


def test_batch_chunking_parity():
    """Verify that batch chunking (batch_size in 1, 2, 3, 5, 10) produces strictly identical results."""
    service = EvidenceQualityService()

    claims_data = [
        ("Claim 1: Mercury is the closest planet to the Sun.", "https://nasa.gov/mercury"),
        ("Claim 2: Venus has a dense carbon dioxide atmosphere.", "https://esa.int/venus"),
        ("Claim 3: Mars possesses two small moons named Phobos and Deimos.", "https://nasa.gov/mars"),
        ("Claim 4: Jupiter is the largest planet in the Solar System.", "https://nasa.gov/jupiter"),
        ("Claim 5: Saturn is known for its extensive planetary ring system.", "https://esa.int/saturn"),
    ]

    requests = []
    for idx, (claim_text, url) in enumerate(claims_data):
        requests.append(
            EvidenceQualityRequest(
                claim_text=claim_text,
                claim_id=f"claim_{idx}",
                evidence_items=[
                    EvidenceItem(
                        evidence_id=f"ev_{idx}_a",
                        text=claim_text,
                        source_url=url,
                        domain="nasa.gov" if "nasa" in url else "esa.int",
                        published_date="2023-01-01",
                    ),
                    EvidenceItem(
                        evidence_id=f"ev_{idx}_b",
                        text=claim_text + " Confirmed by space probe observations.",
                        source_url=url,
                        domain="nasa.gov" if "nasa" in url else "esa.int",
                        published_date="2023-02-01",
                    ),
                ],
            )
        )

    # Run across different chunk sizes
    baseline = service.evaluate_claims_batch(requests, batch_size=10)
    for chunk_size in [1, 2, 3, 5]:
        result = service.evaluate_claims_batch(requests, batch_size=chunk_size)
        assert len(result) == len(baseline)
        for b_rep, r_rep in zip(baseline, result):
            assert b_rep.claim_id == r_rep.claim_id
            assert b_rep.claim_text == r_rep.claim_text
            assert b_rep.total_evidence_count == r_rep.total_evidence_count
            assert b_rep.admissible_count == r_rep.admissible_count
            assert b_rep.polluted_count == r_rep.polluted_count
            assert b_rep.cluster_count == r_rep.cluster_count
            assert b_rep.kish_effective_count == r_rep.kish_effective_count
            assert len(b_rep.evaluated_items) == len(r_rep.evaluated_items)
            for b_it, r_it in zip(b_rep.evaluated_items, r_rep.evaluated_items):
                assert b_it["evidence_id"] == r_it["evidence_id"]
                assert b_it["credibility_score"] == r_it["credibility_score"]
                assert b_it["freshness_score"] == r_it["freshness_score"]
                assert b_it["pollution_score"] == r_it["pollution_score"]
                assert b_it["is_admissible"] == r_it["is_admissible"]


def test_batch_empty_payload(client):
    """Verify that empty batch payloads return empty results with HTTP 200."""
    service = EvidenceQualityService()

    # Direct service call
    assert service.evaluate_claims_batch([]) == []

    # API call with empty list
    resp1 = client.post("/api/v1/evidence/evaluate-quality-batch", json=[])
    assert resp1.status_code == 200
    assert resp1.json() == []

    # API call with empty wrapper
    resp2 = client.post("/api/v1/evidence/evaluate-quality-batch", json={"requests": []})
    assert resp2.status_code == 200
    assert resp2.json() == []


def test_batch_sibling_claims_session_store_isolation(client):
    """Verify quality audits for sibling claims under the same query_id are isolated in session_claims."""
    query_id = str(uuid4())
    session_store = get_session_store()

    # Pre-register session
    client.post("/api/v1/query", json={"query": "Solar system queries", "query_id": query_id})

    # Prepare 2 sibling claims under the same query_id
    # Claim A: 2 clean items
    req_a = {
        "claim_text": "Europa has a subsurface liquid water ocean.",
        "claim_id": "claim_europa",
        "query_id": query_id,
        "evidence_items": [
            _make_evidence_item("e1", "Europa is believed to harbor a global subsurface ocean beneath its ice crust.", "https://nasa.gov/europa1"),
            _make_evidence_item("e2", "Evidence from the Galileo spacecraft showed magnetic signatures of Europa ocean.", "https://nasa.gov/europa2"),
        ],
    }

    # Claim B: 1 clean item, 1 completely ungrounded polluted item
    req_b = {
        "claim_text": "Titan is the only moon in the Solar System known to have a dense atmosphere.",
        "claim_id": "claim_titan",
        "query_id": query_id,
        "evidence_items": [
            _make_evidence_item("t1", "Titan is the only moon with a dense nitrogen-rich atmosphere.", "https://esa.int/titan"),
            _make_evidence_item("t2", "Bake the cake at 350 degrees Fahrenheit for 25 minutes until golden brown.", "https://baking.example.com/recipe"),
        ],
    }

    resp = client.post("/api/v1/evidence/evaluate-quality-batch", json=[req_a, req_b])
    assert resp.status_code == 200

    # Retrieve per-claim records from session_store
    claim_a_rec = session_store.get_claim_record(uuid4().__class__(query_id), "claim_europa")
    claim_b_rec = session_store.get_claim_record(uuid4().__class__(query_id), "claim_titan")

    assert claim_a_rec is not None, "claim_europa must exist in session_claims"
    assert claim_b_rec is not None, "claim_titan must exist in session_claims"

    audit_a = claim_a_rec["metadata"].get("evidence_quality")
    audit_b = claim_b_rec["metadata"].get("evidence_quality")

    assert audit_a is not None
    assert audit_b is not None

    # Claim A: 2 admissible, 0 polluted
    assert audit_a["total_evidence_count"] == 2
    assert audit_a["admissible_count"] == 2
    assert audit_a["polluted_count"] == 0

    # Claim B: 1 admissible, 1 polluted
    assert audit_b["total_evidence_count"] == 2
    assert audit_b["admissible_count"] == 1
    assert audit_b["polluted_count"] == 1


def test_temporal_anchor_awareness_in_pipeline():
    """Verify that temporal_anchor correctly prevents penalizing historical evidence."""
    service = EvidenceQualityService()
    claim = "Apollo 11 landed on the Moon in July 1969."
    ref_date = date(2024, 1, 1)

    items = [
        {
            "evidence_id": "hist_01",
            "text": "Apollo 11 landed astronauts on the lunar surface on July 20, 1969.",
            "source_url": "https://history.nasa.gov/apollo11",
            "domain": "nasa.gov",
            "published_date": date(1970, 1, 1),  # 6 months after mission
        }
    ]

    # Without temporal anchor: 1970 doc vs 2024 reference date has age ~54 years -> freshness ~ 0.0
    report_no_anchor = service.evaluate_evidence(
        claim=claim,
        evidence_items=items,
        reference_date=ref_date,
        temporal_anchor=None,
    )
    rep_no_anchor = report_no_anchor.evaluated_items[0]
    assert rep_no_anchor.freshness_score < 0.01

    # With temporal anchor "1969": 1970 doc is contemporaneous (delta ~ 182 days <= 730d window) -> freshness >= 0.95
    report_with_anchor = service.evaluate_evidence(
        claim=claim,
        evidence_items=items,
        reference_date=ref_date,
        temporal_anchor="1969",
    )
    rep_with_anchor = report_with_anchor.evaluated_items[0]
    assert rep_with_anchor.freshness_score >= 0.95


def test_kish_n_eff_admissible_clusters_only():
    """Verify that Kish's n_eff = M^2 / sum(s_k^2) is calculated strictly over admissible clusters."""
    service = EvidenceQualityService()
    claim = "The quick brown fox jumped across the park."

    items = [
        # Cluster 1: 2 near-duplicate clean documents
        {
            "evidence_id": "d1",
            "text": "The fast brown fox jumped across the green park yesterday.",
            "source_url": "https://gov.example.com/1",
            "domain": "gov.example.com",
            "published_date": date(2023, 1, 1),
        },
        {
            "evidence_id": "d2",
            "text": "The fast brown fox jumped across the green park.",
            "source_url": "https://gov.example.com/2",
            "domain": "gov.example.com",
            "published_date": date(2023, 1, 1),
        },
        # Cluster 2: 1 clean singleton document
        {
            "evidence_id": "d3",
            "text": "A nimble brown fox was seen leaping through the park trees.",
            "source_url": "https://edu.example.org/1",
            "domain": "edu.example.org",
            "published_date": date(2023, 1, 1),
        },
        # Cluster 3: 1 ungrounded polluted document (rejected)
        {
            "evidence_id": "d4",
            "text": "Baking recipe ingredients include flour butter sugar eggs and vanilla extract.",
            "source_url": "https://recipe.com/bake",
            "domain": "recipe.com",
            "published_date": date(2023, 1, 1),
        },
    ]

    report = service.evaluate_evidence(
        claim=claim,
        evidence_items=items,
        reference_date=date(2023, 6, 1),
    )

    # d4 must be rejected
    d4_rep = next(it for it in report.evaluated_items if it.evidence_id == "d4")
    assert not d4_rep.is_admissible
    assert d4_rep.pollution_score >= 0.80

    assert report.total_evidence_count == 4
    assert report.admissible_count == 3
    assert report.polluted_count == 1

    # Admissible clusters:
    # Cluster 1 has 2 admissible items (s_1 = 2)
    # Cluster 2 has 1 admissible item (s_2 = 1)
    # M = 2 + 1 = 3
    # sum(s_k^2) = 2^2 + 1^2 = 4 + 1 = 5
    # Kish n_eff = M^2 / sum(s_k^2) = 9 / 5 = 1.80
    assert report.kish_effective_count == 1.80


def test_input_validation_errors(client):
    """Verify input validation: whitespace claim, empty evidence, and invalid batch_size."""
    service = EvidenceQualityService()

    # 1. Whitespace claim rejected with 422
    resp_blank = client.post(
        "/api/v1/evidence/evaluate-quality",
        json={
            "claim_text": "    ",
            "evidence_items": [_make_evidence_item("x1", "Some text")],
        },
    )
    assert resp_blank.status_code == 422

    # 2. Empty evidence items rejected with 422
    resp_no_ev = client.post(
        "/api/v1/evidence/evaluate-quality",
        json={
            "claim_text": "Valid claim",
            "evidence_items": [],
        },
    )
    assert resp_no_ev.status_code == 422

    # 3. Invalid batch_size in evaluate_claims_batch
    req = EvidenceQualityRequest(
        claim_text="Valid claim",
        evidence_items=[
            EvidenceItem(
                evidence_id="e1",
                text="Evidence text",
                source_url="https://example.com",
                published_date="2023-01-01",
            )
        ],
    )

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        service.evaluate_claims_batch([req], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        service.evaluate_claims_batch([req], batch_size=-5)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        service.evaluate_claims_batch([req], batch_size="ten")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        service.evaluate_claims_batch([req], batch_size=True)  # type: ignore[arg-type]


def test_heuristic_disclaimer_present(client):
    """Verify that disclaimer field contains the Research Module 8 notice."""
    resp = client.post(
        "/api/v1/evidence/evaluate-quality",
        json={
            "claim_text": "The Sun is the star at the center of the Solar System.",
            "evidence_items": [_make_evidence_item("sun1", "The Sun is the star at the center of the Solar System.")],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disclaimer"] == DISCLAIMER_PHRASE
