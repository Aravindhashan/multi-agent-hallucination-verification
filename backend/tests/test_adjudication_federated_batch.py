"""Comprehensive Integration & Batch Tests for Component 5:
Downstream Module 8 & 9 DeBERTa Cross-Encoder Adjudication & Verdict Engine.

Verifies end-to-end integration across:
- Component 3: FederatedEvidenceRetriever (SciFact, FEVER, Live Web)
- Module 7: Multi-Turn Evidentiary Debate (Defender vs Prosecutor)
- Module 8: Evidence Quality Firewall & Cluster Representative Pooling
- Module 9: DeBERTa-v3 NLI Cross-Encoder Adjudication Engine
- Module 10: Calibrated Tri-Class Verdict Taxonomy (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
"""

from types import SimpleNamespace
from uuid import uuid4
from fastapi.testclient import TestClient
import pytest
import torch

from app.main import app
from app.schemas import (
    AdjudicationBatchRequest,
    AdjudicationBatchResponse,
    AdjudicationRequest,
    AdjudicationResponse,
    AdjudicationVerdict,
    EvidenceItem,
)
from app.services.adjudication_service import (
    AdjudicationService,
    get_adjudication_service,
    reset_adjudication_service,
)
from app.services.federated_evidence_retriever import FederatedEvidenceRetriever
from app.services.nli_engine import NLIEngine
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture
def client() -> TestClient:
    """Return FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(scope="module")
def federated_retriever() -> FederatedEvidenceRetriever:
    """Return shared FederatedEvidenceRetriever in hybrid mode."""
    return FederatedEvidenceRetriever(mode="hybrid", top_k=2, escalation_threshold=0.50)


# Mock helpers for fast deterministic testing
class FastMockTokenizer:
    def __call__(self, premises, hypotheses=None, padding=False, truncation=False, max_length=None, return_tensors=None):
        if isinstance(premises, str):
            premises = [premises]
        batch_size = len(premises)
        return {
            "input_ids": torch.zeros((batch_size, 10), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
        }


class FastMockModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        return SimpleNamespace(logits=self.logits.repeat(batch_size, 1))


# ============================================================================
# 1. Real SciFact Biomedical Claim Adjudication (Supported Claim)
# ============================================================================

def test_biomedical_scifact_claim_adjudication(federated_retriever: FederatedEvidenceRetriever):
    """Verify real biomedical claim routes to SciFact, clusters evidence, and is adjudicated
    as SUPPORTED by DeBERTa cross-encoder with high confidence.
    """
    service = get_adjudication_service(retriever=federated_retriever)
    claim = "Aspirin reduces cardiovascular mortality in clinical trials"

    resp = service.adjudicate_claim(claim_text=claim, max_debate_rounds=1)

    assert isinstance(resp, AdjudicationResponse)
    assert resp.claim_text == claim
    assert resp.verdict == AdjudicationVerdict.SUPPORTED
    assert resp.support_probability >= 0.60
    assert resp.contradiction_probability <= 0.20
    assert resp.confidence_score >= 0.60

    # Evidence comes from PubMed/SciFact
    assert len(resp.evaluated_evidence) >= 1
    assert any("ncbi.nlm.nih.gov" in (ev.domain or "") for ev in resp.evaluated_evidence)
    assert resp.supporting_n_eff >= 1.0


# ============================================================================
# 2. Refuted Claim Adjudication (Contradiction Detection)
# ============================================================================

def test_refuted_claim_adjudication():
    """Verify claim that contradicts medical evidence is adjudicated as REFUTED."""
    # Strong contradiction logits: [entailment=-3.0, neutral=-1.0, contradiction=5.0]
    mock_model = FastMockModel(logits=torch.tensor([[-3.0, -1.0, 5.0]], dtype=torch.float32))
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    service = AdjudicationService(nli_engine=mock_engine)

    claim = "Aspirin has zero clinical benefits and increases vascular mortality"
    supporting_evidence = [
        EvidenceItem(
            evidence_id="pubmed-trial-1",
            text="Randomized trials of 14,000 patients confirm low-dose aspirin reduces vascular mortality.",
            domain="ncbi.nlm.nih.gov",
            source_title="PubMed Prevention Trial",
            relevance_score=0.95,
            credibility_score=0.90,
            freshness_score=0.85,
        )
    ]

    resp = service.adjudicate_claim(
        claim_text=claim,
        supporting_evidence=supporting_evidence,
        run_debate=False,
    )

    assert resp.verdict == AdjudicationVerdict.REFUTED
    assert resp.contradiction_probability >= 0.60
    assert resp.support_probability <= 0.20
    assert len(resp.evaluated_evidence) == 1
    assert resp.evaluated_evidence[0].label == "contradiction"


# ============================================================================
# 3. Contemporary Astronomy Live Web Claim Adjudication
# ============================================================================

def test_contemporary_astronomy_claim_adjudication():
    """Verify contemporary astronomy claim with authentic NASA evidence is adjudicated as SUPPORTED."""
    claim = "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b"
    evidence = [
        EvidenceItem(
            evidence_id="nasa-jwst-co2",
            text="NASA's James Webb Space Telescope has captured the first clear evidence for carbon dioxide in the atmosphere of exoplanet WASP-39b.",
            domain="science.nasa.gov",
            source_url="https://science.nasa.gov/jwst-wasp39b-co2",
            source_title="NASA JWST News",
            relevance_score=0.98,
            credibility_score=0.95,
            freshness_score=0.90,
        )
    ]

    service = get_adjudication_service()
    resp = service.adjudicate_claim(
        claim_text=claim,
        supporting_evidence=evidence,
        run_debate=False,
    )

    assert resp.verdict == AdjudicationVerdict.SUPPORTED
    assert resp.support_probability >= 0.85
    assert resp.contradiction_probability <= 0.05
    assert resp.evaluated_evidence[0].domain == "science.nasa.gov"
    assert resp.evaluated_evidence[0].label == "entailment"


# ============================================================================
# 4. Inconclusive / Insufficient Evidence Adjudication
# ============================================================================

def test_inconclusive_claim_adjudication():
    """Verify that ungrounded claim with neutral evidence returns NOT_ENOUGH_INFO."""
    mock_model = FastMockModel(logits=torch.tensor([[-2.0, 5.0, -2.0]], dtype=torch.float32))
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    service = AdjudicationService(nli_engine=mock_engine)

    claim = "Atmospheric conditions on Titan trigger methane rainstorms every Tuesday"
    evidence = [
        EvidenceItem(
            evidence_id="neutral-astronomy-doc",
            text="Titan is the largest moon of Saturn and has a dense atmosphere containing nitrogen and methane.",
            domain="en.wikipedia.org",
            relevance_score=0.50,
            credibility_score=0.80,
            freshness_score=0.70,
        )
    ]

    resp = service.adjudicate_claim(
        claim_text=claim,
        supporting_evidence=evidence,
        run_debate=False,
    )

    assert resp.verdict == AdjudicationVerdict.NOT_ENOUGH_INFO
    assert resp.neutral_probability >= 0.50
    assert resp.evaluated_evidence[0].label == "neutral"


# ============================================================================
# 5. Batch Adjudication with Memory-Safe Chunking
# ============================================================================

def test_adjudicate_claims_batch_memory_safe_chunking():
    """Verify batch claim adjudication slices requests into chunks <= batch_size
    and preserves input ordering.
    """
    mock_model = FastMockModel(logits=torch.tensor([[5.0, -2.0, -2.0]], dtype=torch.float32))
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    service = AdjudicationService(nli_engine=mock_engine)

    claims = [
        f"Synthetic verifiable claim number {i} for batch evaluation"
        for i in range(1, 5)
    ]
    evidence = [
        EvidenceItem(
            evidence_id=f"ev-doc-{i}",
            text=f"Synthetic verifiable claim number {i} for batch evaluation is substantiated.",
            relevance_score=0.90,
            credibility_score=0.85,
            freshness_score=0.80,
        )
        for i in range(1, 5)
    ]

    batch_requests = [
        AdjudicationRequest(
            claim_text=claims[i],
            supporting_evidence=[evidence[i]],
            run_debate=False,
        )
        for i in range(4)
    ]

    responses = service.adjudicate_claims_batch(
        claims=batch_requests,
        batch_size=2,  # Memory-safe chunk size
    )

    assert len(responses) == 4
    for idx, resp in enumerate(responses):
        assert resp.claim_text == claims[idx]
        assert resp.verdict == AdjudicationVerdict.SUPPORTED
        assert resp.confidence_score >= 0.60


# ============================================================================
# 6. SQLite SessionStore Audit Persistence
# ============================================================================

def test_adjudication_session_store_persistence():
    """Verify claim adjudication records are persisted under session_claims.metadata_json['adjudication']."""
    reset_session_store()
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    store.create_session(query_id=query_id, query="Medical research verification query")

    mock_model = FastMockModel(logits=torch.tensor([[4.0, -1.0, -2.0]], dtype=torch.float32))
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    service = AdjudicationService(nli_engine=mock_engine)

    claim = "Low-dose aspirin reduces long-term vascular mortality"
    evidence = [
        EvidenceItem(
            evidence_id="doc-session-test",
            text="Meta-analysis shows daily low-dose aspirin reduces 20-year vascular mortality by 34%.",
            domain="ncbi.nlm.nih.gov",
            relevance_score=0.95,
            credibility_score=0.90,
            freshness_score=0.85,
        )
    ]

    resp = service.adjudicate_claim(
        claim_text=claim,
        claim_id=claim_id,
        query_id=query_id,
        supporting_evidence=evidence,
        run_debate=False,
    )

    # Verify session store record in SQLite
    claim_rec = store.get_claim_record(query_id=query_id, claim_id=claim_id)
    assert claim_rec is not None
    assert claim_rec["claim_id"] == str(claim_id)
    assert claim_rec.get("adjudication") is not None

    audit = claim_rec["adjudication"]
    assert audit["verdict"] == "SUPPORTED"
    assert audit["confidence_score"] == resp.confidence_score
    assert audit["support_probability"] == resp.support_probability
    assert audit["total_evidence_evaluated"] == 1


# ============================================================================
# 7. FastAPI REST API Endpoints (/adjudicate & /adjudicate-batch)
# ============================================================================

def test_api_adjudicate_endpoint(client: TestClient):
    """Verify POST /api/v1/adjudicate endpoint works with valid payloads and handles errors."""
    from app.routers.adjudication import get_adjudication_service_dep

    mock_model = FastMockModel(logits=torch.tensor([[4.0, -1.0, -2.0]], dtype=torch.float32))
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    app.dependency_overrides[get_adjudication_service_dep] = lambda: AdjudicationService(nli_engine=mock_engine)

    try:
        # Valid adjudication request
        res = client.post(
            "/api/v1/adjudicate",
            json={
                "claim_text": "Aspirin reduces cardiovascular mortality in clinical trials.",
                "supporting_evidence": [
                    {
                        "evidence_id": "test-doc-1",
                        "text": "Clinical trials demonstrate aspirin reduces vascular mortality.",
                        "domain": "ncbi.nlm.nih.gov",
                        "relevance_score": 0.95,
                        "credibility_score": 0.90,
                        "freshness_score": 0.85,
                    }
                ],
                "run_debate": False,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["verdict"] == "SUPPORTED"
        assert data["confidence_score"] >= 0.50
        assert len(data["evaluated_evidence"]) == 1

        # Blank claim text rejection (422)
        blank_res = client.post("/api/v1/adjudicate", json={"claim_text": "   "})
        assert blank_res.status_code == 422
    finally:
        app.dependency_overrides.clear()
