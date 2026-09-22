"""Unit tests for stance differentiation, singleton retrieval, and router dependency injection.

These tests run fast and offline without triggering heavy transformer or neural model allocations.
"""

from uuid import uuid4
from fastapi.testclient import TestClient
import pytest

from app.dependencies import get_evidence_retriever
from app.main import app
from app.schemas import EvidenceItem
from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_schema import CorpusDocument
from app.services.defender_service import defend_claim
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.services.mock_evidence_retriever import MockEvidenceRetriever
from app.services.prosecutor_service import prosecute_claim
from app.services.retrieval_models import RetrievalResult


class LightweightDenseRetriever:
    """Mock dense retriever for fast, zero-memory-overhead hybrid testing."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                document_id="doc_geo_001",
                text="Paris is the official capital and most populous city of France.",
                title="Capital of France",
                source="World Atlas",
                source_url="https://worldatlas.example.org/france",
                domain="worldatlas.example.org",
                score=0.92,
                retrieval_method="DENSE",
                rank=1,
            )
        ]


@pytest.fixture
def lightweight_hybrid_retriever():
    """Build a lightweight HybridRetriever using BM25 and LightweightDenseRetriever."""
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
            text="Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
            source="Medical History",
            source_url="https://medhistory.example.org/penicillin",
            domain="medhistory.example.org",
        ),
    ]
    bm25 = BM25Retriever.from_documents(docs)
    dense = LightweightDenseRetriever()
    return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)


def test_hybrid_retriever_stance_differentiation(lightweight_hybrid_retriever, monkeypatch):
    """Verify retrieve() adapts search query based on supporting vs contradicting stance."""
    captured_queries = []
    original_retrieve_hybrid = lightweight_hybrid_retriever.retrieve_hybrid

    def mock_retrieve_hybrid(query: str, top_k: int = 5):
        captured_queries.append(query)
        return original_retrieve_hybrid(query, top_k)

    monkeypatch.setattr(lightweight_hybrid_retriever, "retrieve_hybrid", mock_retrieve_hybrid)

    claim = "Alexander Fleming discovered penicillin in 1928."
    lightweight_hybrid_retriever.retrieve(claim, stance="supporting")
    assert len(captured_queries) == 1
    assert captured_queries[0] == claim

    lightweight_hybrid_retriever.retrieve(claim, stance="contradicting")
    assert len(captured_queries) == 2
    assert "controversy" in captured_queries[1]
    assert "disputed" in captured_queries[1]


def test_defender_and_prosecutor_services_real_retrieval(lightweight_hybrid_retriever):
    """Verify defender_service and prosecutor_service execute real retrieval when provided."""
    claim = "Paris is the capital of France."
    def_resp = defend_claim(claim_text=claim, retriever=lightweight_hybrid_retriever)
    assert len(def_resp.supporting_evidence) > 0
    # Real evidence returned from corpus, not mock placeholder
    assert "MOCK" not in def_resp.supporting_evidence[0].text
    assert "paris" in def_resp.supporting_evidence[0].text.lower()
    assert "hybrid" in def_resp.supporting_evidence[0].retrieval_method

    pros_resp = prosecute_claim(claim_text=claim, retriever=lightweight_hybrid_retriever)
    assert len(pros_resp.contradicting_evidence) > 0
    assert "MOCK" not in pros_resp.contradicting_evidence[0].text


def test_defender_prosecutor_router_dependency_injection(lightweight_hybrid_retriever):
    """Verify FastAPI Depends(get_evidence_retriever) in /defend-claim and /prosecute-claim."""
    client = TestClient(app)
    claim = "Paris is the capital of France."

    # 1. Override dependency with our lightweight real retriever
    app.dependency_overrides[get_evidence_retriever] = lambda: lightweight_hybrid_retriever
    try:
        res = client.post("/api/v1/defend-claim", json={"claim_text": claim})
        assert res.status_code == 200
        data = res.json()
        assert len(data["supporting_evidence"]) > 0
        assert "MOCK" not in data["supporting_evidence"][0]["text"]
        assert "paris" in data["supporting_evidence"][0]["text"].lower()

        pros_res = client.post("/api/v1/prosecute-claim", json={"claim_text": claim})
        assert pros_res.status_code == 200
        pros_data = pros_res.json()
        assert len(pros_data["contradicting_evidence"]) > 0
        assert "MOCK" not in pros_data["contradicting_evidence"][0]["text"]
    finally:
        app.dependency_overrides.pop(get_evidence_retriever, None)

    # 2. Dependency override with mock retriever works cleanly
    app.dependency_overrides[get_evidence_retriever] = lambda: MockEvidenceRetriever()
    try:
        mock_res = client.post("/api/v1/defend-claim", json={"claim_text": claim})
        assert mock_res.status_code == 200
        mock_data = mock_res.json()
        assert len(mock_data["supporting_evidence"]) > 0
        assert "MOCK" in mock_data["supporting_evidence"][0]["text"]
    finally:
        app.dependency_overrides.pop(get_evidence_retriever, None)
