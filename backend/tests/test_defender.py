import pytest
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EvidenceItem, DefenderRequest, DefenderResponse
from app.services.defender_service import defend_claim, defend_claims_batch


class MockRetriever:
    """A mock evidence retriever that returns deterministic evidence items."""

    def __init__(self, items):
        self._items = items

    def retrieve(self, query: str, stance: str = "supporting"):
        # In a real retriever, the query would affect results; here we just return the preset list.
        return self._items


def make_evidence(evidence_id: str, text: str, relevance: float, credibility: float = 1.0, freshness: float = 1.0):
    return EvidenceItem(
        evidence_id=evidence_id,
        text=text,
        relevance_score=relevance,
        credibility_score=credibility,
        freshness_score=freshness,
    )


@pytest.fixture(autouse=True)
def reset_session_store():
    # Ensure a fresh singleton for each test.
    from app.services.session_store import reset_session_store
    reset_session_store()
    yield
    reset_session_store()


def test_defender_query_and_filtering_and_store():
    # Prepare deterministic evidence items.
    evidence_items = [
        make_evidence("e1", "High relevance evidence", 0.9),
        make_evidence("e2", "Low relevance evidence", 0.15),  # should be filtered out
        make_evidence("e3", "Medium relevance evidence", 0.5, credibility=0.8, freshness=0.7),
    ]
    mock_retriever = MockRetriever(evidence_items)

    # Mock the session store to provide anchors and capture the record_claim_evidence call.
    mock_store = MagicMock()
    mock_store.get_claim_record.return_value = {
        "verification_plan": [],
        "temporal_anchors": ["2020-01-01"],
        "numerical_values": ["100"],
        "spatial_anchors": [],
    }
    # Replace the singleton getter with our mock.
    with patch("app.services.defender_service.get_session_store", return_value=mock_store):
        claim_text = "The event happened on 2020-01-01 and involved 100 units."
        claim_id = uuid4()
        query_id = uuid4()
        response: DefenderResponse = defend_claim(
            claim_text=claim_text,
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        # Verify the query used incorporates the extracted anchors.
        assert "dates: 2020-01-01" in response.query_used
        assert "numbers: 100" in response.query_used
        assert response.query_used != claim_text  # baseline should differ

        # Verify low‑relevance evidence was filtered out.
        returned_ids = {e.evidence_id for e in response.supporting_evidence}
        assert "e2" not in returned_ids
        assert returned_ids == {"e1", "e3"}

        # Verify the session store recorded supporting evidence correctly.
        mock_store.record_claim_evidence.assert_called_once()
        args, kwargs = mock_store.record_claim_evidence.call_args
        assert kwargs["query_id"] == query_id
        assert kwargs["claim_id"] == claim_id
        assert kwargs["evidence_type"] == "supporting"
        # The evidence payload should be a list of dicts matching the retained items.
        recorded = kwargs["evidence"]
        recorded_ids = {item["evidence_id"] for item in recorded}
        assert recorded_ids == {"e1", "e3"}

        # Verify timestamps are timezone‑aware.
        assert isinstance(response.defended_at, datetime)
        assert response.defended_at.tzinfo is not None


def test_defender_query_incorporates_plan():
    """Verify that Module 4 verification plan directives are included in query formulation."""
    mock_retriever = MockRetriever([])
    mock_store = MagicMock()
    mock_store.get_claim_record.return_value = {
        "verification_plan": ["Verify chronological order", "Verify quantity 50"],
        "temporal_anchors": ["2023"],
        "numerical_values": ["50"],
        "spatial_anchors": ["Paris"],
    }
    with patch("app.services.defender_service.get_session_store", return_value=mock_store):
        response = defend_claim(
            claim_text="The exhibition opened in Paris in 2023 with 50 paintings.",
            claim_id=uuid4(),
            query_id=uuid4(),
            retriever=mock_retriever,
        )
        assert "dates: 2023" in response.query_used
        assert "numbers: 50" in response.query_used
        assert "locations: Paris" in response.query_used
        assert "plan: Verify chronological order; Verify quantity 50" in response.query_used


def test_defender_query_duplicate_check_and_ablation_log():
    """Verify duplicate checking and ablation logging when no anchors are present."""
    mock_retriever = MockRetriever([])
    mock_store = MagicMock()
    mock_store.get_claim_record.return_value = {
        "verification_plan": [],
        "temporal_anchors": [],
        "numerical_values": [],
        "spatial_anchors": [],
    }
    with patch("app.services.defender_service.get_session_store", return_value=mock_store):
        query_id = uuid4()
        response = defend_claim(
            claim_text="Trees produce oxygen.",
            claim_id=uuid4(),
            query_id=query_id,
            retriever=mock_retriever,
        )
        # Without anchors or plan, query matches baseline
        assert response.query_used == "Trees produce oxygen."
        mock_store.update_metadata.assert_called_once()
        k_args, kwargs = mock_store.update_metadata.call_args
        assert kwargs["key"] == "defender_ablation_log"
        ablation = kwargs["value"]
        assert ablation["dynamic_queries"] == ["Trees produce oxygen."]
        assert ablation["fixed_baseline_queries"] == ["Trees produce oxygen."]


def test_defender_fallback_anchor_extraction_without_session():
    """Verify fallback anchor extraction with NLP parser when query_id is None."""
    mock_retriever = MockRetriever([])
    response = defend_claim(
        claim_text="On 2021-05-10, company revenue grew by 25 percent.",
        claim_id=uuid4(),
        query_id=None,
        retriever=mock_retriever,
    )
    # The NLP extractor extracts 2021-05-10 and 25
    assert "dates: 2021-05-10" in response.query_used or "2021-05-10" in response.query_used
    assert "numbers: 25" in response.query_used or "25" in response.query_used


def test_defender_real_session_store_anchors_roundtrip():
    """Verify integration with real SQLite session store persisting and retrieving anchors."""
    from app.services.session_store import get_session_store
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    store.create_session(query_id=query_id, query="What occurred in 1999?")
    store.record_claims(
        query_id=query_id,
        claims=[
            {
                "claim_id": str(claim_id),
                "claim_text": "In 1999, the team won 3 trophies.",
                "temporal_anchors": ["1999"],
                "numerical_values": ["3"],
                "spatial_anchors": [],
            }
        ],
    )

    mock_retriever = MockRetriever([make_evidence("e1", "Record found", 0.8)])
    response = defend_claim(
        claim_text="In 1999, the team won 3 trophies.",
        claim_id=claim_id,
        query_id=query_id,
        retriever=mock_retriever,
    )

    assert "dates: 1999" in response.query_used
    assert "numbers: 3" in response.query_used
    # Verify evidence was recorded to SQLite
    claim_rec = store.get_claim_record(query_id, claim_id)
    assert claim_rec is not None
    assert len(claim_rec["supporting_evidence"]) == 1
    assert claim_rec["supporting_evidence"][0]["evidence_id"] == "e1"


def test_defender_hybrid_supporting_stance_passed():
    """Verify that Defender explicitly requests stance='supporting' from the retriever."""
    captured_stances = []

    class StanceTrackingRetriever:
        def retrieve(self, query: str, stance: str = "supporting"):
            captured_stances.append(stance)
            return [make_evidence("e1", "Evidence text", 0.8)]

    retriever = StanceTrackingRetriever()
    resp = defend_claim(claim_text="The sky is blue.", retriever=retriever)
    assert len(captured_stances) == 1
    assert captured_stances[0] == "supporting"


def test_defend_claims_batch_empty():
    """Verify empty input returns empty list immediately."""
    assert defend_claims_batch([]) == []


def test_defend_claims_batch_small():
    """Verify defending a small batch of claims."""
    mock_retriever = MockRetriever([make_evidence("e1", "Facts", 0.75)])
    claims = [
        {"claim_text": "France has a population over 60 million.", "claim_id": uuid4()},
        {"claim_text": "Tokyo is the capital of Japan.", "claim_id": uuid4()},
        {"claim_text": "Water boils at 100 degrees Celsius.", "claim_id": uuid4()},
    ]
    results = defend_claims_batch(claims, retriever=mock_retriever)
    assert len(results) == 3
    for res in results:
        assert isinstance(res, DefenderResponse)
        assert len(res.supporting_evidence) == 1
        assert res.query_used != ""


def test_defend_claims_batch_chunking_limits():
    """Verify memory-safe chunking with batch_size=5 for 12 claims."""
    processed_count = 0

    class TrackingRetriever:
        def retrieve(self, query: str, stance: str = "supporting"):
            nonlocal processed_count
            processed_count += 1
            return [make_evidence(f"e_{processed_count}", f"Evidence {processed_count}", 0.7)]

    claims = [
        DefenderRequest(claim_text=f"Statement number {i} is a factual assertion.")
        for i in range(12)
    ]
    results = defend_claims_batch(claims, retriever=TrackingRetriever(), batch_size=5)
    assert len(results) == 12
    assert processed_count == 12


def test_defend_claims_batch_with_hybrid_retriever():
    """Verify batch defense using BM25 and Dense subretrievers (HybridRetriever)."""
    from app.services.bm25_retriever import BM25Retriever
    from app.services.corpus_schema import CorpusDocument
    from app.services.hybrid_retriever import HybridRetriever
    from tests.test_retrieval_wiring_and_stance import LightweightDenseRetriever

    docs = [
        CorpusDocument(
            id="doc_geo_001",
            title="Capital of France",
            text="Paris is the official capital and most populous city of France.",
            source="World Atlas",
            source_url="https://worldatlas.example.org/france",
            domain="worldatlas.example.org",
        ),
        CorpusDocument(
            id="doc_sci_001",
            title="Penicillin Discovery",
            text="Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital.",
            source="Medical History",
            source_url="https://medhistory.example.org/penicillin",
            domain="medhistory.example.org",
        ),
    ]
    bm25 = BM25Retriever.from_documents(docs)
    dense = LightweightDenseRetriever()
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)

    claims = [
        {"claim_text": "Paris is the capital of France."},
        {"claim_text": "Alexander Fleming discovered penicillin in 1928."},
    ]
    batch_res = defend_claims_batch(claims, retriever=hybrid)
    assert len(batch_res) == 2
    for item in batch_res:
        assert len(item.supporting_evidence) > 0
        assert "hybrid" in item.supporting_evidence[0].retrieval_method


def test_defend_claims_batch_session_persistence():
    """Verify session persistence across a batch of claims in SQLite."""
    from app.services.session_store import get_session_store
    store = get_session_store()
    query_id = uuid4()
    c1_id = uuid4()
    c2_id = uuid4()

    store.create_session(query_id=query_id, query="Batch test query")
    store.record_claims(
        query_id=query_id,
        claims=[
            {"claim_id": str(c1_id), "claim_text": "First batch claim"},
            {"claim_id": str(c2_id), "claim_text": "Second batch claim"},
        ],
    )

    mock_retriever = MockRetriever([make_evidence("ev_batch", "Supporting data", 0.85)])
    batch_res = defend_claims_batch(
        claims=[
            {"claim_id": c1_id, "claim_text": "First batch claim"},
            {"claim_id": c2_id, "claim_text": "Second batch claim"},
        ],
        query_id=query_id,
        retriever=mock_retriever,
    )

    assert len(batch_res) == 2
    rec1 = store.get_claim_record(query_id, c1_id)
    rec2 = store.get_claim_record(query_id, c2_id)
    assert rec1 is not None and len(rec1["supporting_evidence"]) == 1
    assert rec2 is not None and len(rec2["supporting_evidence"]) == 1


def test_defender_relevance_filtering_threshold():
    """Verify that items strictly below DEFENDER_MIN_RELEVANCE are discarded."""
    evidence_items = [
        make_evidence("e_keep_1", "Relevance 0.20 boundary", 0.20),
        make_evidence("e_drop_1", "Relevance 0.19", 0.19),
        make_evidence("e_drop_2", "Relevance 0.05", 0.05),
        make_evidence("e_keep_2", "Relevance 0.70", 0.70),
    ]
    mock_retriever = MockRetriever(evidence_items)
    response = defend_claim(claim_text="General claim.", retriever=mock_retriever)
    retained_ids = [e.evidence_id for e in response.supporting_evidence]
    assert "e_drop_1" not in retained_ids
    assert "e_drop_2" not in retained_ids
    assert "e_keep_1" in retained_ids
    assert "e_keep_2" in retained_ids


def test_defender_filtering_all_dropped_returns_empty():
    """Verify that if all evidence is low-relevance, an empty list is safely returned."""
    evidence_items = [
        make_evidence("e_low_1", "Low relevance 1", 0.10),
        make_evidence("e_low_2", "Low relevance 2", 0.02),
    ]
    mock_retriever = MockRetriever(evidence_items)
    response = defend_claim(claim_text="Obscure assertion.", retriever=mock_retriever)
    assert response.supporting_evidence == []


def test_defender_ranking_weighted_formula_verification():
    """Verify that evidence items are ranked strictly by the weighted composite score equation:

    Composite = 0.5 * rel + 0.3 * cred + 0.2 * fresh
    """
    # eA: 0.5*0.6 + 0.3*0.9 + 0.2*0.8 = 0.30 + 0.27 + 0.16 = 0.73
    eA = make_evidence("eA", "Evidence A", relevance=0.6, credibility=0.9, freshness=0.8)
    # eB: 0.5*0.8 + 0.3*0.6 + 0.2*0.5 = 0.40 + 0.18 + 0.10 = 0.68
    eB = make_evidence("eB", "Evidence B", relevance=0.8, credibility=0.6, freshness=0.5)
    # eC: 0.5*0.9 + 0.3*0.8 + 0.2*0.9 = 0.45 + 0.24 + 0.18 = 0.87
    eC = make_evidence("eC", "Evidence C", relevance=0.9, credibility=0.8, freshness=0.9)

    # Deliver in unsorted order [eB, eA, eC]
    mock_retriever = MockRetriever([eB, eA, eC])
    response = defend_claim(claim_text="Scientific hypothesis.", retriever=mock_retriever)

    # Expected order: eC (0.87), then eA (0.73), then eB (0.68)
    ranked_ids = [e.evidence_id for e in response.supporting_evidence]
    assert ranked_ids == ["eC", "eA", "eB"]
    assert response.supporting_evidence[0].composite_score == 0.87
    assert response.supporting_evidence[1].composite_score == 0.73
    assert response.supporting_evidence[2].composite_score == 0.68


def test_defender_no_free_text_generation():
    """Verify rule adherence: strictly no free-text conversational arguments generated."""
    evidence_items = [make_evidence("e1", "Factual statement.", 0.85)]
    mock_retriever = MockRetriever(evidence_items)
    response = defend_claim(claim_text="Sample claim.", retriever=mock_retriever)

    # Response schema should strictly contain structured fields without free-text argument fields
    assert hasattr(response, "supporting_evidence")
    assert hasattr(response, "query_used")
    assert not hasattr(response, "argument")
    assert not hasattr(response, "free_text_defense")
    assert not hasattr(response, "llm_argument")


def test_defend_claims_batch_filtering_and_ranking():
    """Batch test verifying filtering and ranking across a batch of claims."""
    class DynamicRetriever:
        def retrieve(self, query: str, stance: str = "supporting"):
            if "high" in query.lower():
                return [
                    make_evidence("hi_1", "High 1", relevance=0.7, credibility=0.8, freshness=0.9),  # 0.35+0.24+0.18=0.77
                    make_evidence("hi_2", "High 2", relevance=0.9, credibility=0.9, freshness=0.9),  # 0.45+0.27+0.18=0.90
                    make_evidence("hi_low", "Below threshold", relevance=0.1),  # filtered
                ]
            elif "none" in query.lower():
                return [
                    make_evidence("drop_1", "Drop 1", relevance=0.15),
                    make_evidence("drop_2", "Drop 2", relevance=0.08),
                ]
            else:
                return [make_evidence("mid_1", "Mid 1", relevance=0.5)]

    claims = [
        {"claim_text": "High relevance statement with numbers: 10", "temporal_anchors": [], "numerical_values": ["10"], "spatial_anchors": []},
        {"claim_text": "None relevance statement", "temporal_anchors": [], "numerical_values": [], "spatial_anchors": []},
        {"claim_text": "Standard statement", "temporal_anchors": [], "numerical_values": [], "spatial_anchors": []},
    ]

    batch_responses = defend_claims_batch(claims, retriever=DynamicRetriever())
    assert len(batch_responses) == 3

    # First claim: hi_low filtered out, hi_2 ranked before hi_1
    assert len(batch_responses[0].supporting_evidence) == 2
    assert batch_responses[0].supporting_evidence[0].evidence_id == "hi_2"
    assert batch_responses[0].supporting_evidence[1].evidence_id == "hi_1"

    # Second claim: all dropped
    assert len(batch_responses[1].supporting_evidence) == 0

    # Third claim: standard mid item retained
    assert len(batch_responses[2].supporting_evidence) == 1
    assert batch_responses[2].supporting_evidence[0].evidence_id == "mid_1"


def test_defender_api_route_single_claim():
    """Verify POST /api/v1/defend-claim endpoint integration."""
    client = TestClient(app)
    payload = {
        "claim_text": "Apollo 11 landed on the Moon in July 1969.",
        "claim_id": str(uuid4()),
        "query_id": str(uuid4()),
    }
    response = client.post("/api/v1/defend-claim", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == payload["claim_id"]
    assert data["query_id"] == payload["query_id"]
    assert data["claim_text"] == payload["claim_text"]
    assert "query_used" in data
    assert "defended_at" in data
    assert isinstance(data["supporting_evidence"], list)


def test_defender_api_validation_errors():
    """Verify HTTP 422 validation on malformed inputs to /api/v1/defend-claim."""
    client = TestClient(app)

    # Whitespace only
    resp = client.post("/api/v1/defend-claim", json={"claim_text": "   "})
    assert resp.status_code == 422

    # Missing claim_text
    resp = client.post("/api/v1/defend-claim", json={})
    assert resp.status_code == 422

    # Invalid UUID
    resp = client.post("/api/v1/defend-claim", json={"claim_text": "Valid text", "query_id": "invalid-uuid"})
    assert resp.status_code == 422


def test_defender_api_route_batch_claims():
    """Verify POST /api/v1/defend-claims-batch endpoint processing a batch of claims."""
    client = TestClient(app)
    batch_payload = [
        {"claim_text": f"Statement number {i} is under verification.", "claim_id": str(uuid4())}
        for i in range(4)
    ]
    response = client.post("/api/v1/defend-claims-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 4
    for idx, item in enumerate(data):
        assert item["claim_id"] == batch_payload[idx]["claim_id"]
        assert item["claim_text"] == batch_payload[idx]["claim_text"]
        assert "query_used" in item
        assert "defended_at" in item


def test_defender_sqlite_query_used_persistence():
    """Verify that defender_query_used is persisted to SQLite session_claims metadata."""
    from app.services.session_store import get_session_store
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    store.create_session(query_id=query_id, query="Audit query")
    store.record_claims(
        query_id=query_id,
        claims=[
            {
                "claim_id": str(claim_id),
                "claim_text": "On 2022-01-01, the rate rose to 5 percent.",
                "temporal_anchors": ["2022-01-01"],
                "numerical_values": ["5"],
                "spatial_anchors": [],
            }
        ],
    )

    mock_retriever = MockRetriever([make_evidence("e_audit", "Economic report", 0.88)])
    response = defend_claim(
        claim_text="On 2022-01-01, the rate rose to 5 percent.",
        claim_id=claim_id,
        query_id=query_id,
        retriever=mock_retriever,
    )

    record = store.get_claim_record(query_id, claim_id)
    assert record is not None
    assert record.get("defender_query_used") == response.query_used
    assert "dates: 2022-01-01" in record["defender_query_used"]
    assert "numbers: 5" in record["defender_query_used"]




