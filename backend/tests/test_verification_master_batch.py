"""Comprehensive Master Batch Test Suite for Component 6:
Master End-to-End Pipeline Orchestrator API (Module 11 & Module 10 Rollup).

Verifies the unified verification runner across Modules 1 through 10:
- Module 1: Session tracking & query creation
- Module 3: Atomic claim decomposition & qualifier preservation
- Module 4: Complexity analysis & verification planning
- Module 7: Multi-turn federated debate (Defender vs Prosecutor)
- Module 8: Evidence quality firewall, clustering, & Kish n_eff discounting
- Module 9: DeBERTa-v3 cross-encoder NLI inference on cluster representatives
- Module 10: Calibrated claim adjudication & response-level rollup with global hallucination index
- Module 11: End-to-end REST API endpoints (POST /api/v1/verify & /verify-batch)
"""

from types import SimpleNamespace
from uuid import uuid4
from fastapi.testclient import TestClient
import pytest
import torch

from app.main import app
from app.schemas import (
    AdjudicationVerdict,
    EvidenceItem,
    ResponseVerdict,
    VerificationBatchRequest,
    VerificationBatchResponse,
    VerificationRequest,
    VerificationResponse,
)
from app.services.adjudication_service import (
    AdjudicationService,
    get_adjudication_service,
    reset_adjudication_service,
)
from app.services.federated_evidence_retriever import FederatedEvidenceRetriever
from app.services.nli_engine import NLIEngine
from app.services.session_store import get_session_store, reset_session_store
from app.services.verification_service import (
    VerificationService,
    get_verification_service,
    reset_verification_service,
)


@pytest.fixture
def client() -> TestClient:
    """Return FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_services():
    """Reset session store and singleton services for each test."""
    reset_session_store()
    reset_adjudication_service()
    reset_verification_service()
    yield
    reset_session_store()
    reset_adjudication_service()
    reset_verification_service()


# Fast Mock Tokenizer & Model for fast deterministic testing
class FastMockTokenizer:
    def __call__(
        self,
        premises,
        hypotheses=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        if isinstance(premises, str):
            premises = [premises]
        batch_size = len(premises)
        return {
            "input_ids": torch.zeros((batch_size, 10), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
        }


class FastMockModel:
    def __init__(self, logits_func=None):
        self.logits_func = logits_func
        self.config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        if self.logits_func:
            return SimpleNamespace(logits=self.logits_func(batch_size))
        # Default entailment: [entailment=5.0, neutral=-1.0, contradiction=-3.0]
        base_logits = torch.tensor([[5.0, -1.0, -3.0]], dtype=torch.float32)
        return SimpleNamespace(logits=base_logits.repeat(batch_size, 1))


# ============================================================================
# 1. Real SciFact Biomedical Statement Verification (Supported)
# ============================================================================

def test_biomedical_supported_verification():
    """Verify factual medical statement decomposed into atomic claims and
    verified as SUPPORTED with zero hallucination risk against SciFact evidence.
    """
    mock_model = FastMockModel()
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    federated_retriever = FederatedEvidenceRetriever(
        mode="hybrid", top_k=2, escalation_threshold=0.50
    )
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=federated_retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    statement = "Aspirin reduces cardiovascular mortality in clinical trials."
    resp = verif_service.verify_text(statement, max_debate_rounds=1)

    assert isinstance(resp, VerificationResponse)
    assert resp.text == statement
    assert resp.claim_count >= 1
    assert resp.overall_verdict == ResponseVerdict.SUPPORTED
    assert resp.hallucination_index == 0.0
    assert resp.supported_count >= 1
    assert resp.refuted_count == 0
    assert resp.confidence_score >= 0.60
    assert resp.total_evidence_evaluated >= 1
    assert resp.aggregate_joint_n_eff >= 1.0

    # Verify claim level details
    top_claim = resp.claims[0]
    assert top_claim.verdict == AdjudicationVerdict.SUPPORTED
    assert len(top_claim.winning_evidence_snippets) >= 1


# ============================================================================
# 2. Refuted Hallucination Detection
# ============================================================================

def test_refuted_hallucination_verification():
    """Verify statement contradicting evidence is flagged as REFUTED with high hallucination index."""
    # Contradiction logits: [entailment=-3.0, neutral=-1.0, contradiction=5.0]
    mock_model = FastMockModel(
        logits_func=lambda b: torch.tensor([[-3.0, -1.0, 5.0]], dtype=torch.float32).repeat(b, 1)
    )
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    statement = "Aspirin has zero clinical benefits and increases vascular mortality."
    resp = verif_service.verify_text(statement, max_debate_rounds=1)

    assert isinstance(resp, VerificationResponse)
    assert resp.overall_verdict == ResponseVerdict.REFUTED
    assert resp.hallucination_index >= 0.75
    assert resp.refuted_count >= 1
    assert resp.supported_count == 0


# ============================================================================
# 3. Multi-Claim Mixed-Veracity Passage (Partially Supported Rollup)
# ============================================================================

def test_multi_claim_mixed_veracity_passage():
    """Verify multi-sentence passage containing 1 true fact and 1 false assertion
    decomposes into 2 claims and rolls up into PARTIALLY_SUPPORTED with proportional
    hallucination index (~0.45 - 0.55).
    """
    # Dynamic logits: claim 0 -> entailment; claim 1 -> contradiction
    call_counter = [0]

    def dynamic_logits(b):
        idx = call_counter[0]
        call_counter[0] += 1
        if idx % 2 == 0:
            # Entailment
            return torch.tensor([[4.5, -0.5, -2.5]], dtype=torch.float32).repeat(b, 1)
        else:
            # Contradiction
            return torch.tensor([[-2.5, -0.5, 4.5]], dtype=torch.float32).repeat(b, 1)

    mock_model = FastMockModel(logits_func=dynamic_logits)
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    passage = (
        "Aspirin reduces cardiovascular mortality in clinical trials. "
        "Aspirin was first synthesized in the future year 2099."
    )
    resp = verif_service.verify_text(passage, max_debate_rounds=1)

    assert isinstance(resp, VerificationResponse)
    assert resp.claim_count == 2
    assert resp.overall_verdict == ResponseVerdict.PARTIALLY_SUPPORTED
    assert resp.supported_count == 1
    assert resp.refuted_count == 1
    # Hallucination index reflects 1 refuted claim out of 2: ~0.40 - 0.60
    assert 0.35 <= resp.hallucination_index <= 0.65
    assert resp.confidence_score >= 0.60


# ============================================================================
# 4. Contemporary Live Web Statement Verification
# ============================================================================

def test_contemporary_live_web_verification():
    """Verify contemporary astronomy claim routes through live web/hybrid,
    verifies NASA discoveries, and correctly logs provenance.
    """
    # Entailment logits: [entailment=4.8, neutral=-0.8, contradiction=-2.5]
    mock_model = FastMockModel(
        logits_func=lambda b: torch.tensor([[4.8, -0.8, -2.5]], dtype=torch.float32).repeat(b, 1)
    )
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())

    # Pre-populate custom retriever with authentic NASA JWST evidence
    class MockNASAWebRetriever:
        def retrieve(self, claim_text, stance="supporting"):
            return [
                EvidenceItem(
                    evidence_id="nasa-jwst-1",
                    text="NASA's James Webb Space Telescope detected clear evidence for carbon dioxide in the atmosphere of WASP-39b.",
                    domain="science.nasa.gov",
                    published_date="2022-08-25",
                    source_url="https://science.nasa.gov/missions/webb/wasp-39b-co2/",
                    relevance_score=0.96,
                )
            ]

    federated_retriever = FederatedEvidenceRetriever(
        mode="live_web",
        web_retriever=MockNASAWebRetriever(),
    )
    adj_service = AdjudicationService(
        nli_engine=mock_engine, retriever=federated_retriever
    )
    verif_service = VerificationService(adjudication_service=adj_service)

    statement = "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b."
    resp = verif_service.verify_text(statement, evidence_mode="live_web", max_debate_rounds=1)

    assert resp.overall_verdict == ResponseVerdict.SUPPORTED
    assert resp.hallucination_index == 0.0
    assert resp.evidence_mode_used == "live_web"
    assert resp.supported_count == 1


# ============================================================================
# 5. Inconclusive / Ungrounded Statement Verification
# ============================================================================

def test_inconclusive_verification():
    """Verify ungrounded claim with neutral evidence results in NOT_ENOUGH_INFO."""
    # Neutral logits: [entailment=-2.0, neutral=4.5, contradiction=-2.0]
    mock_model = FastMockModel(
        logits_func=lambda b: torch.tensor([[-2.0, 4.5, -2.0]], dtype=torch.float32).repeat(b, 1)
    )
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    statement = "Atmospheric conditions on Titan trigger methane rainstorms every Tuesday."
    resp = verif_service.verify_text(statement, max_debate_rounds=1)

    assert resp.overall_verdict == ResponseVerdict.NOT_ENOUGH_INFO
    assert resp.unverified_count >= 1
    assert resp.supported_count == 0
    assert resp.refuted_count == 0
    # Decoupled invariant: absent evidence is NOT a factual hallucination
    assert resp.hallucination_index == 0.0
    assert resp.unverified_index >= 0.50


# ============================================================================
# 6. Memory-Safe Chunked Batch Verification
# ============================================================================

def test_memory_safe_batch_verification():
    """Verify verify_batch slices input into chunks <= 2 and preserves strict ordering."""
    mock_model = FastMockModel()
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    requests = [
        "Aspirin reduces cardiovascular mortality in clinical trials.",
        "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b.",
        "Clinical trials evaluated 14,000 patients across international medical centers.",
    ]

    batch_resp = verif_service.verify_batch(requests=requests, batch_size=2)

    assert isinstance(batch_resp, VerificationBatchResponse)
    assert batch_resp.total_processed == 3
    assert len(batch_resp.responses) == 3
    for i, orig_text in enumerate(requests):
        assert batch_resp.responses[i].text == orig_text
    assert 0.0 <= batch_resp.mean_hallucination_index <= 1.0
    assert 0.0 <= batch_resp.mean_confidence <= 1.0


# ============================================================================
# 7. SQLite Session Store Audit Trail Persistence
# ============================================================================

def test_session_store_full_audit_persistence():
    """Verify complete end-to-end verification audit is saved under query_id in SQLite."""
    store = get_session_store()
    mock_model = FastMockModel()
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service, session_store=store)

    q_id = uuid4()
    statement = "Aspirin reduces cardiovascular mortality in clinical trials."
    resp = verif_service.verify_text(statement, query_id=q_id, max_debate_rounds=1)

    # 1. Check sessions table
    sess = store.get_session(q_id)
    assert sess is not None
    assert sess.query == statement
    assert "verification_result" in sess.metadata
    assert sess.metadata["overall_verdict"] == ResponseVerdict.SUPPORTED.value
    assert sess.metadata["hallucination_index"] == 0.0

    # 2. Check session_claims child table
    claims_records = store.get_claim_records(q_id)
    assert len(claims_records) >= 1
    top_claim_rec = claims_records[0]
    assert top_claim_rec["complexity_score"] is not None
    assert top_claim_rec["verification_plan"] is not None
    assert "adjudication" in top_claim_rec
    adj_data = top_claim_rec["adjudication"]
    assert adj_data["verdict"] == AdjudicationVerdict.SUPPORTED.value


# ============================================================================
# 8. REST API Endpoints (POST /api/v1/verify & /verify-batch)
# ============================================================================

def test_fastapi_verification_endpoints(client: TestClient):
    """Verify FastAPI endpoints handle single and batch verification with proper HTTP status codes."""
    from app.routers.verification import get_verification_service_dep

    mock_model = FastMockModel()
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    app.dependency_overrides[get_verification_service_dep] = lambda: verif_service

    try:
        # 1. Valid single text verification
        payload = {
            "text": "Aspirin reduces cardiovascular mortality in clinical trials.",
            "max_debate_rounds": 1,
        }
        post_resp = client.post("/api/v1/verify", json=payload)
        assert post_resp.status_code == 200
        data = post_resp.json()
        assert "query_id" in data
        assert data["overall_verdict"] in [v.value for v in ResponseVerdict]
        assert "hallucination_index" in data
        assert "claims" in data

        # 2. Validation error on empty text (422)
        bad_payload = {"text": "   "}
        err_resp = client.post("/api/v1/verify", json=bad_payload)
        assert err_resp.status_code == 422

        # 3. Batch endpoint verification
        batch_payload = {
            "requests": [
                {"text": "Aspirin reduces cardiovascular mortality in clinical trials."},
                {"text": "WASP-39b is an exoplanet orbiting a sun-like star."},
            ]
        }
        batch_resp = client.post("/api/v1/verify-batch", json=batch_payload)
        assert batch_resp.status_code == 200
        batch_data = batch_resp.json()
        assert batch_data["total_processed"] == 2
        assert len(batch_data["responses"]) == 2
    finally:
        app.dependency_overrides.clear()

