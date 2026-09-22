"""Unit and route integration tests for SQLite PipelineSessionStore and Modules 1 to 4 correlation."""

from datetime import datetime, timezone
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.session_store import (
    PipelineSessionStore,
    get_session_store,
    reset_session_store,
)
from app.services.simple_claim_analyzer import SimpleClaimAnalyzer


@pytest.fixture(autouse=True)
def clean_session_store():
    """Ensure session store is clean before and after each test."""
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


def test_session_store_crud():
    """Verify session store creation, retrieval, updates, and clear."""
    store = PipelineSessionStore(db_path=":memory:")
    q_id = uuid4()

    session = store.create_session(query_id=q_id, query="What is gravity?")
    assert session.query_id == q_id
    assert session.query == "What is gravity?"
    assert session.response_text is None

    # Retrieve
    retrieved = store.get_session(q_id)
    assert retrieved is not None
    assert retrieved.query_id == q_id

    # Record generation
    updated = store.record_generation(
        query_id=q_id,
        query="What is gravity?",
        response_text="Gravity is a fundamental interaction.",
        model_name="test-model",
    )
    assert updated.response_text == "Gravity is a fundamental interaction."
    assert updated.model_name == "test-model"

    # Record claims
    store.record_claims(query_id=q_id, claims=[{"claim_id": "c1", "claim_text": "Claim 1"}])
    session = store.get_session(q_id)
    assert session.claims == [{"claim_id": "c1", "claim_text": "Claim 1"}]

    # Record analysis
    store.record_analysis(query_id=q_id, complexity_score=0.75, verification_plan=["Step 1"])
    session = store.get_session(q_id)
    assert session.complexity_score == 0.75
    assert session.verification_plan == ["Step 1"]

    # Clear
    store.clear()
    assert store.get_session(q_id) is None
    assert len(store) == 0


def test_sqlite_persistence_across_store_instances(tmp_path):
    """Verify that SQLite file database persists sessions across store re-instantiation."""
    db_file = tmp_path / "test_sessions.db"
    store1 = PipelineSessionStore(db_path=db_file)
    q_id = uuid4()

    store1.create_session(query_id=q_id, query="Persisted query test")
    store1.record_generation(
        query_id=q_id,
        query="Persisted query test",
        response_text="Persisted response text",
        model_name="sqlite-test-model",
    )
    store1.record_claims(
        query_id=q_id,
        claims=[{"claim_id": "c1", "claim_text": "Fact 1", "subject": "Fact"}],
    )

    # Re-instantiate a new store object pointing to the exact same SQLite file
    store2 = PipelineSessionStore(db_path=db_file)
    loaded = store2.get_session(q_id)

    assert loaded is not None
    assert loaded.query_id == q_id
    assert loaded.query == "Persisted query test"
    assert loaded.response_text == "Persisted response text"
    assert loaded.model_name == "sqlite-test-model"
    assert len(loaded.claims) == 1
    assert loaded.claims[0]["subject"] == "Fact"


def test_module_1_query_endpoint(client):
    """Verify POST /api/v1/query assigns and preserves query_id and logs session."""
    # Without pre-supplied query_id
    res1 = client.post("/api/v1/query", json={"query": "What is machine learning?"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert "query_id" in data1
    assert data1["query"] == "What is machine learning?"
    q_id1 = data1["query_id"]

    # Verify session recorded
    session1 = get_session_store().get_session(uuid4().__class__(q_id1))
    assert session1 is not None
    assert session1.query == "What is machine learning?"

    # With pre-supplied query_id
    custom_id = str(uuid4())
    res2 = client.post("/api/v1/query", json={"query": "What is deep learning?", "query_id": custom_id})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["query_id"] == custom_id


def test_module_1_to_module_2_correlation(client):
    """Verify that query_id from Module 1 flows into Module 2 and updates the same session."""
    # 1. Module 1: Query input
    res_query = client.post("/api/v1/query", json={"query": "What is quantum computing?"})
    assert res_query.status_code == 200
    query_id = res_query.json()["query_id"]

    # 2. Module 2: Generate response with the preserved query_id
    res_gen = client.post("/api/v1/generate", json={"query": "What is quantum computing?", "query_id": query_id})
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert gen_data["query_id"] == query_id
    assert len(gen_data["response_text"]) > 0

    # 3. Verify session store now contains both query and generated response under query_id
    session = get_session_store().get_session(uuid4().__class__(query_id))
    assert session is not None
    assert session.query == "What is quantum computing?"
    assert session.response_text == gen_data["response_text"]
    assert session.model_name == gen_data["model_name"]


def test_module_1_to_module_3_pipeline_correlation(client):
    """Verify complete query -> generate -> extract-claims correlation and SQLite session persistence."""
    # Step 1: Submit query
    q_res = client.post("/api/v1/query", json={"query": "When was Apollo 11 launched?"})
    assert q_res.status_code == 200
    query_id = q_res.json()["query_id"]

    # Step 2: Generate response carrying query_id
    g_res = client.post(
        "/api/v1/generate",
        json={"query": "When was Apollo 11 launched?", "query_id": query_id},
    )
    assert g_res.status_code == 200
    assert g_res.json()["query_id"] == query_id

    # Step 3: Extract claims carrying query_id
    factual_text = "Apollo 11 was launched on July 16, 1969 from Kennedy Space Center."
    c_res = client.post(
        "/api/v1/extract-claims",
        json={
            "query": "When was Apollo 11 launched?",
            "response_text": factual_text,
            "query_id": query_id,
            "method": "nlp",
        },
    )
    assert c_res.status_code == 200
    c_data = c_res.json()
    assert c_data["query_id"] == query_id
    assert len(c_data["claims"]) > 0

    # Step 4: Verify that claims preserve atomic qualifiers
    first_claim = c_data["claims"][0]
    assert first_claim["order"] == 0
    assert "Apollo 11" in first_claim["text"]
    assert first_claim["temporal_anchor"] is not None

    # Step 5: Verify SQLite session store has all stages captured
    session = get_session_store().get_session(uuid4().__class__(query_id))
    assert session is not None
    assert session.query_id == uuid4().__class__(query_id)
    assert len(session.claims) > 0
    assert session.claims[0]["temporal_anchor"] is not None


def test_module_4_semantic_claim_analysis_unit():
    """Verify SimpleClaimAnalyzer reacts to structured atomic semantic qualifiers."""
    analyzer = SimpleClaimAnalyzer()

    # Plain statement
    res_simple = analyzer.analyze("Water boils at sea level.")
    assert res_simple.complexity_score < 0.60
    assert not res_simple.requires_deep_verification

    # Statement with structured anchors
    structured_claim = {
        "claim_text": "Apollo 11 landed on the Moon in July 1969.",
        "subject": "Apollo 11",
        "temporal_anchor": "July 1969",
        "spatial_anchor": "the Moon",
        "qualifiers": ["in July 1969"],
    }
    res_structured = analyzer.analyze(structured_claim["claim_text"], atomic_claim=structured_claim)
    assert res_structured.complexity_score >= 0.60
    assert any("numerical or date-specific" in step for step in res_structured.verification_plan)
    assert any("geographical and spatial" in step for step in res_structured.verification_plan)
    assert any("named entities" in step for step in res_structured.verification_plan)

    # Superlative / absolute claim triggering deep verification
    res_absolute = analyzer.analyze("Chennai is the only city with 100 percent solar energy.")
    assert res_absolute.requires_deep_verification
    assert any("prosecutor agent" in step.lower() for step in res_absolute.verification_plan)


def test_full_pipeline_module_1_to_4_correlation(client):
    """Verify complete query -> generate -> extract-claims -> analyze-claim correlation in SQLite."""
    # Step 1: Submit query
    q_res = client.post("/api/v1/query", json={"query": "Who developed Python and when was it released?"})
    assert q_res.status_code == 200
    query_id = q_res.json()["query_id"]

    # Step 2: Generate response
    g_res = client.post(
        "/api/v1/generate",
        json={"query": "Who developed Python and when was it released?", "query_id": query_id},
    )
    assert g_res.status_code == 200
    assert g_res.json()["query_id"] == query_id

    # Step 3: Extract claims
    sample_response = "Python was created by Guido van Rossum and first released in 1991."
    c_res = client.post(
        "/api/v1/extract-claims",
        json={
            "query": "Who developed Python and when was it released?",
            "response_text": sample_response,
            "query_id": query_id,
        },
    )
    assert c_res.status_code == 200
    claims = c_res.json()["claims"]
    assert len(claims) > 0
    claim_to_analyze = claims[0]

    # Step 4: Analyze claim with query_id and pre-extracted anchors
    a_res = client.post(
        "/api/v1/analyze-claim",
        json={
            "claim_text": claim_to_analyze["text"],
            "claim_id": str(claim_to_analyze["claim_id"]),
            "query_id": query_id,
            "subject": claim_to_analyze.get("subject"),
            "predicate": claim_to_analyze.get("predicate"),
            "temporal_anchor": claim_to_analyze.get("temporal_anchor"),
            "numerical_value": claim_to_analyze.get("numerical_value"),
            "spatial_anchor": claim_to_analyze.get("spatial_anchor"),
            "qualifiers": claim_to_analyze.get("qualifiers", []),
        },
    )
    assert a_res.status_code == 200
    a_data = a_res.json()
    assert str(a_data["claim_id"]) == str(claim_to_analyze["claim_id"])
    assert "verification_plan" in a_data
    assert a_data["complexity_score"] > 0.0

    # Step 5: Verify SQLite session store has all 4 stages linked!
    session = get_session_store().get_session(uuid4().__class__(query_id))
    assert session is not None
    assert session.query_id == uuid4().__class__(query_id)
    assert session.query == "Who developed Python and when was it released?"
    assert len(session.claims) > 0
    assert session.complexity_score is not None
    assert session.complexity_score == a_data["complexity_score"]
    assert len(session.verification_plan) > 0


def test_multi_claim_session_persistence_no_overwrite(client):
    """Verify that sequential /analyze-claim calls for different claims do not overwrite each other (Rank 3)."""
    q_res = client.post("/api/v1/query", json={"query": "Tell me about Apollo 11."})
    assert q_res.status_code == 200
    query_id = q_res.json()["query_id"]

    # Analyze Claim 1 (simple claim)
    res1 = client.post(
        "/api/v1/analyze-claim",
        json={
            "claim_text": "Apollo 11 was an American spaceflight.",
            "claim_id": "c1",
            "query_id": query_id,
        },
    )
    assert res1.status_code == 200
    score1 = res1.json()["complexity_score"]

    # Analyze Claim 2 (complex claim with numbers and dates)
    res2 = client.post(
        "/api/v1/analyze-claim",
        json={
            "claim_text": "On July 20, 1969, Apollo 11 landed 2 astronauts on the Moon.",
            "claim_id": "c2",
            "query_id": query_id,
            "temporal_anchor": "July 20, 1969",
            "numerical_value": "2",
            "spatial_anchor": "Moon",
        },
    )
    assert res2.status_code == 200
    score2 = res2.json()["complexity_score"]

    # Verify both claim records are retained in session_store without data loss
    store = get_session_store()
    claim_records = store.get_claim_records(uuid4().__class__(query_id))
    assert len(claim_records) == 2

    c1_rec = store.get_claim_record(uuid4().__class__(query_id), "c1")
    c2_rec = store.get_claim_record(uuid4().__class__(query_id), "c2")

    assert c1_rec is not None
    assert c1_rec["claim_id"] == "c1"
    assert c1_rec["complexity_score"] == score1

    assert c2_rec is not None
    assert c2_rec["claim_id"] == "c2"
    assert c2_rec["complexity_score"] == score2
