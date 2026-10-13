"""End-to-End Router and Pipeline Integration Tests.

Validates the complete lifecycle flow through all FastAPI routers:
1. POST /api/v1/query (Module 1 query registration)
2. POST /api/v1/generate (Module 2 response generation)
3. POST /api/v1/extract-claims (Module 3 claim extraction with qualifiers)
4. POST /api/v1/analyze-claim (Module 4 semantic complexity analysis)
5. POST /api/v1/evidence/evaluate-quality (Module 4 evidence quality evaluation)
6. SQLite Session Store persistence across the entire pipeline
7. Input validation error handling (HTTP 422)
"""

from datetime import date
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture(autouse=True)
def clean_state():
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    return TestClient(app)


def test_query_to_generation_pipeline(client):
    """Test correlation from POST /query to POST /generate with session persistence."""
    query_text = "What are the primary scientific goals of the James Webb Space Telescope?"
    
    # 1. Register query
    res_query = client.post("/api/v1/query", json={"query": query_text})
    assert res_query.status_code == 200
    q_data = res_query.json()
    assert "query_id" in q_data
    query_id = q_data["query_id"]

    # Verify session store created
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert session.query == query_text
    assert session.response_text is None

    # 2. Generate response linking same query_id
    res_gen = client.post(
        "/api/v1/generate",
        json={
            "query": query_text,
            "query_id": query_id,
            "max_tokens": 150,
            "temperature": 0.2,
        },
    )
    assert res_gen.status_code == 200
    g_data = res_gen.json()
    assert g_data["query_id"] == query_id
    assert len(g_data["response_text"]) > 10

    # Verify session store updated
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert session.response_text == g_data["response_text"]
    assert session.model_name == g_data["model_name"]


def test_generation_to_claim_extraction(client):
    """Test extracting claims from generated response and ensuring atomic qualifiers are preserved."""
    query_text = "When was the James Webb Space Telescope launched?"
    response_text = "The James Webb Space Telescope was launched on December 25, 2021 aboard an Ariane 5 rocket."

    # Register session
    res_query = client.post("/api/v1/query", json={"query": query_text})
    query_id = res_query.json()["query_id"]

    # Extract claims
    res_claims = client.post(
        "/api/v1/extract-claims",
        json={
            "query": query_text,
            "response_text": response_text,
            "query_id": query_id,
            "strategy": "rule_based",
        },
    )
    assert res_claims.status_code == 200
    c_data = res_claims.json()
    assert c_data["query_id"] == query_id
    assert len(c_data["claims"]) >= 1

    first_claim = c_data["claims"][0]
    assert "subject" in first_claim
    assert "predicate" in first_claim
    assert "temporal_anchor" in first_claim

    # Verify qualifiers are extracted
    assert first_claim["subject"] is not None
    assert first_claim["temporal_anchor"] is not None
    assert "2021" in first_claim["temporal_anchor"] or "December" in first_claim["temporal_anchor"]

    # Check persistence
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert len(session.claims) == len(c_data["claims"])


def test_claim_to_analysis(client):
    """Test claim complexity analysis reacting to temporal and numerical anchors."""
    claim_text = "On July 20, 1969, Apollo 11 landed 2 astronauts on the Moon."
    query_id = str(uuid4())

    # Create session first
    client.post("/api/v1/query", json={"query": "Moon landing", "query_id": query_id})

    res_analysis = client.post(
        "/api/v1/analyze-claim",
        json={
            "claim_id": str(uuid4()),
            "text": claim_text,
            "query_id": query_id,
            "temporal_anchor": "July 20, 1969",
            "numerical_value": "2",
            "subject": "Apollo 11",
            "spatial_anchor": "Moon",
        },
    )
    assert res_analysis.status_code == 200
    a_data = res_analysis.json()
    assert a_data["query_id"] == query_id
    assert a_data["complexity_score"] > 0.0
    assert len(a_data["verification_plan"]) > 0

    # Check that verification plan contains targeted checks for temporal and numerical anchors
    plan_str = " ".join(a_data["verification_plan"]).lower()
    assert "temporal" in plan_str or "date" in plan_str or "historical" in plan_str
    assert "numeric" in plan_str or "quantit" in plan_str or "count" in plan_str

    # Check persistence
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert session.complexity_score == a_data["complexity_score"]
    assert session.verification_plan == a_data["verification_plan"]


def test_evidence_quality_route_and_metadata(client):
    """Test evidence quality evaluation including clustering and pollution detection."""
    claim = "The James Webb Space Telescope operates at Sun-Earth Lagrange Point 2."
    query_id = str(uuid4())

    # Seed session
    client.post("/api/v1/query", json={"query": "JWST location", "query_id": query_id})

    payload = {
        "claim_text": claim,
        "query_id": query_id,
        "reference_date": "2023-01-01",
        "evidence_items": [
            {
                "evidence_id": "ev_nasa",
                "text": "The James Webb Space Telescope operates at Sun-Earth Lagrange Point 2, approximately 1.5 million kilometers away.",
                "source_url": "https://jwst.nasa.gov/orbit",
                "domain": "nasa.gov",
                "source_title": "NASA JWST",
                "published_date": "2022-06-01",
                "relevance_score": 0.95,
            },
            {
                "evidence_id": "ev_nasa_mirror",
                "text": "The James Webb Space Telescope was deployed to orbit at Sun-Earth Lagrange Point 2 for deep space astronomy.",
                "source_url": "https://syndicate-space.org/jwst-orbit",
                "domain": "syndicate-space.org",
                "source_title": "Syndicate Mirror",
                "published_date": "2022-06-01",
                "relevance_score": 0.92,
            },
            {
                "evidence_id": "ev_polluted_irrelevant",
                "text": "Baking sourdough bread requires water, bread flour, salt, and active sourdough starter fermenting over 24 hours.",
                "source_url": "https://breadbaking.org/sourdough",
                "domain": "breadbaking.org",
                "source_title": "Artisan Bread",
                "published_date": "2022-01-01",
                "relevance_score": 0.10,
            },
        ],
    }

    res = client.post("/api/v1/evidence/evaluate-quality", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["total_evidence_count"] == 3
    assert data["admissible_count"] == 2
    assert data["polluted_count"] == 1
    # Effective sample size < 3 because of near-duplicate mirror
    assert data["kish_effective_count"] < 3.0

    # Verify session store metadata updated
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert "evidence_quality_evaluation" in session.metadata
    meta = session.metadata["evidence_quality_evaluation"]
    assert meta["total_evidence_count"] == 3
    assert meta["admissible_count"] == 2
    assert meta["kish_effective_count"] == data["kish_effective_count"]


def test_full_pipeline_end_to_end_chain(client):
    """Verify complete end-to-end journey across all 5 routers in one correlated flow."""
    # Step 1: Query
    res1 = client.post("/api/v1/query", json={"query": "When was the Hubble Space Telescope launched?"})
    assert res1.status_code == 200
    query_id = res1.json()["query_id"]

    # Step 2: Generation
    res2 = client.post("/api/v1/generate", json={
        "query": "When was the Hubble Space Telescope launched?",
        "query_id": query_id,
        "max_tokens": 80,
    })
    assert res2.status_code == 200
    response_text = "The Hubble Space Telescope was launched into low Earth orbit on April 24, 1990 by Space Shuttle Discovery."

    # Step 3: Claim Extraction
    res3 = client.post("/api/v1/extract-claims", json={
        "query": "When was the Hubble Space Telescope launched?",
        "response_text": response_text,
        "query_id": query_id,
    })
    assert res3.status_code == 200
    claims = res3.json()["claims"]
    assert len(claims) >= 1
    primary_claim = claims[0]
    assert primary_claim["temporal_anchor"] is not None

    # Step 4: Claim Analysis
    res4 = client.post("/api/v1/analyze-claim", json={
        "claim_id": primary_claim["claim_id"],
        "text": primary_claim["text"],
        "query_id": query_id,
        "temporal_anchor": primary_claim["temporal_anchor"],
        "subject": primary_claim["subject"],
        "predicate": primary_claim["predicate"],
        "object": primary_claim.get("object"),
    })
    assert res4.status_code == 200
    analysis = res4.json()
    assert analysis["complexity_score"] > 0

    # Step 5: Evidence Quality Evaluation
    res5 = client.post("/api/v1/evidence/evaluate-quality", json={
        "claim_id": primary_claim["claim_id"],
        "claim_text": primary_claim["text"],
        "query_id": query_id,
        "evidence_items": [
            {
                "evidence_id": "hubble_nasa_doc",
                "text": "The Hubble Space Telescope was launched on April 24, 1990 aboard the Space Shuttle Discovery (STS-31).",
                "source_url": "https://www.nasa.gov/mission_pages/hubble/main/index.html",
                "domain": "nasa.gov",
                "source_title": "NASA Hubble Home",
                "published_date": "2020-04-24",
                "relevance_score": 0.98,
            }
        ],
    })
    assert res5.status_code == 200
    quality = res5.json()
    assert quality["admissible_count"] == 1
    assert quality["polluted_count"] == 0

    # Step 6: Defender Supporting Evidence Gathering
    res6 = client.post("/api/v1/defend-claim", json={
        "claim_id": primary_claim["claim_id"],
        "claim_text": primary_claim["text"],
        "query_id": query_id,
    })
    assert res6.status_code == 200
    defender_data = res6.json()
    assert defender_data["claim_id"] == primary_claim["claim_id"]
    assert "query_used" in defender_data
    assert "supporting_evidence" in defender_data

    # Step 7: Prosecutor Contradicting Evidence Gathering
    res7 = client.post("/api/v1/prosecute-claim", json={
        "claim_id": primary_claim["claim_id"],
        "claim_text": primary_claim["text"],
        "query_id": query_id,
    })
    assert res7.status_code == 200
    prosecutor_data = res7.json()
    assert prosecutor_data["claim_id"] == primary_claim["claim_id"]
    assert "query_used" in prosecutor_data
    assert "contradicting_evidence" in prosecutor_data

    # Step 8: Multi-Turn Evidentiary Debate Orchestration (Module 7)
    res8 = client.post("/api/v1/debate-claim", json={
        "claim_id": primary_claim["claim_id"],
        "claim_text": primary_claim["text"],
        "query_id": query_id,
    })
    assert res8.status_code == 200
    debate_data = res8.json()
    assert debate_data["claim_id"] == primary_claim["claim_id"]
    assert debate_data["total_rounds"] == 2
    assert "final_supporting_evidence" in debate_data
    assert "final_contradicting_evidence" in debate_data
    assert len(debate_data["rounds"]) == 2
    assert "jaccard_complementarity" in debate_data
    assert "incremental_evidence_lift_percent" in debate_data

    # Verify comprehensive final session in SQLite
    store = get_session_store()
    final_session = store.get_session(UUID(query_id))
    assert final_session is not None
    assert final_session.query == "When was the Hubble Space Telescope launched?"
    assert len(final_session.claims) == len(claims)
    assert final_session.complexity_score == analysis["complexity_score"]
    assert "evidence_quality_evaluation" in final_session.metadata

    # Verify claim record in SQLite contains BOTH defender and prosecutor evidence and queries
    claim_rec = store.get_claim_record(UUID(query_id), primary_claim["claim_id"])
    assert claim_rec is not None
    assert "supporting_evidence" in claim_rec
    assert "contradicting_evidence" in claim_rec
    assert claim_rec.get("defender_query_used") is not None
    assert claim_rec.get("prosecutor_query_used") is not None
    assert claim_rec.get("defender_round_1_query") is not None
    assert claim_rec.get("prosecutor_round_1_query") is not None
    assert len(claim_rec.get("debate_rounds", [])) == 2
    assert claim_rec.get("debate", {}).get("total_rounds") == 2
    assert claim_rec.get("metadata", {}).get("debate") is not None



def test_input_validation_errors(client):
    """Test HTTP 422 Unprocessable Entity responses for malformed payloads."""
    # 1. Empty query
    res = client.post("/api/v1/query", json={"query": "   "})
    assert res.status_code == 422

    # 2. Invalid query_id in generate
    res = client.post("/api/v1/generate", json={"query": "What is Python?", "query_id": "not-a-valid-uuid"})
    assert res.status_code == 422

    # 3. Empty claims text
    res = client.post("/api/v1/extract-claims", json={"query": "test", "response_text": ""})
    assert res.status_code == 422

    # 4. Empty text in claim analysis
    res = client.post("/api/v1/analyze-claim", json={"claim_id": str(uuid4()), "text": ""})
    assert res.status_code == 422

    # 5. Empty evidence list in quality evaluation
    res = client.post("/api/v1/evidence/evaluate-quality", json={"claim_text": "Valid claim", "evidence_items": []})
    assert res.status_code == 422

    # 6. Empty claim text in defend-claim
    res = client.post("/api/v1/defend-claim", json={"claim_text": "   "})
    assert res.status_code == 422

    # 7. Empty claim text in debate-claim
    res = client.post("/api/v1/debate-claim", json={"claim_text": "   "})
    assert res.status_code == 422
