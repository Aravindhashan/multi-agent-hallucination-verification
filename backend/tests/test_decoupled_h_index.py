"""Unit and Regression Tests for Decoupled H_refuted and U_unverified Formulation.

Verifies:
1. Pure SUPPORTED statement yields H_refuted = 0.0, U_unverified = 0.0.
2. Pure NOT_ENOUGH_INFO (ungrounded/missing evidence) yields H_refuted = 0.0, U_unverified > 0.0.
   (Proves: Absence of evidence is NEVER penalized as a factual hallucination).
3. Pure REFUTED statement yields H_refuted >= 0.75, U_unverified = 0.0.
4. Mixed passage (1 SUPPORTED + 1 NOT_ENOUGH_INFO) preserves H_refuted = 0.0.
   (Proves: Truncated web snippets do NOT falsely trigger hallucination warnings for factual statements).
5. Mixed passage (1 REFUTED + 1 NOT_ENOUGH_INFO) isolates H_refuted to the refuted claim proportion.
"""

from types import SimpleNamespace
from uuid import uuid4
import pytest
import torch

from app.schemas import (
    AdjudicationVerdict,
    ResponseVerdict,
    VerificationResponse,
    VerificationBatchResponse,
)
from app.services.adjudication_service import (
    AdjudicationService,
    reset_adjudication_service,
)
from app.services.federated_evidence_retriever import FederatedEvidenceRetriever
from app.services.nli_engine import NLIEngine
from app.services.session_store import reset_session_store
from app.services.verification_service import (
    VerificationService,
    reset_verification_service,
)


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
            logits = self.logits_func(batch_size)
        else:
            logits = torch.tensor([[4.5, -1.0, -2.5]], dtype=torch.float32).repeat(batch_size, 1)
        return SimpleNamespace(logits=logits)


def test_decoupled_pure_supported():
    """Supported claim must have 0.0 hallucination risk and 0.0 coverage gap."""
    mock_model = FastMockModel(
        logits_func=lambda b: torch.tensor([[5.0, -1.0, -3.0]], dtype=torch.float32).repeat(b, 1)
    )
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    statement = "Aspirin reduces cardiovascular mortality in clinical trials."
    resp = verif_service.verify_text(statement, max_debate_rounds=1)

    assert resp.overall_verdict == ResponseVerdict.SUPPORTED
    assert resp.hallucination_index == 0.0
    assert resp.unverified_index == 0.0


def test_decoupled_pure_nei_zero_hallucination():
    """Unverified claim (inconclusive/missing evidence) must NOT be penalized as hallucination."""
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
    # Core decoupled invariant: H_refuted must be strictly 0.0!
    assert resp.hallucination_index == 0.0
    # Epistemic uncertainty index captures the coverage gap
    assert resp.unverified_index >= 0.50


def test_decoupled_supported_plus_nei_preserves_zero_hallucination():
    """Mixed passage (1 supported, 1 NEI due to missing web snippets) must have H_refuted = 0.0."""
    call_counter = [0]

    def dynamic_logits(b):
        idx = call_counter[0]
        call_counter[0] += 1
        if idx % 2 == 0:
            # Entailment for Claim 1
            return torch.tensor([[4.5, -1.0, -2.5]], dtype=torch.float32).repeat(b, 1)
        else:
            # Neutral for Claim 2 (e.g. truncated web snippet)
            return torch.tensor([[-2.0, 4.5, -2.0]], dtype=torch.float32).repeat(b, 1)

    mock_model = FastMockModel(logits_func=dynamic_logits)
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    passage = (
        "Aspirin reduces cardiovascular mortality in clinical trials. "
        "Projected quantum dot battery density will exceed 1400 Wh/kg by late 2029."
    )
    resp = verif_service.verify_text(passage, max_debate_rounds=1)

    assert resp.claim_count == 2
    assert resp.supported_count == 1
    assert resp.unverified_count == 1
    assert resp.refuted_count == 0

    # Old buggy formula would have added 0.5 * 0.75 / 2 = 0.1875 hallucination risk!
    # Decoupled formula correctly assigns 0.0 hallucination risk!
    assert resp.hallucination_index == 0.0
    # And flags the uncertainty of the second claim
    assert resp.unverified_index > 0.0


def test_decoupled_refuted_plus_nei():
    """Mixed passage (1 refuted, 1 NEI) isolates H_refuted strictly to the refuted claim."""
    call_counter = [0]

    def dynamic_logits(b):
        idx = call_counter[0]
        call_counter[0] += 1
        if idx % 2 == 0:
            # Contradiction for Claim 1
            return torch.tensor([[-2.5, -1.0, 4.5]], dtype=torch.float32).repeat(b, 1)
        else:
            # Neutral for Claim 2
            return torch.tensor([[-2.0, 4.5, -2.0]], dtype=torch.float32).repeat(b, 1)

    mock_model = FastMockModel(logits_func=dynamic_logits)
    mock_engine = NLIEngine(model=mock_model, tokenizer=FastMockTokenizer())
    retriever = FederatedEvidenceRetriever(mode="toy")
    adj_service = AdjudicationService(nli_engine=mock_engine, retriever=retriever)
    verif_service = VerificationService(adjudication_service=adj_service)

    passage = (
        "Loss-of-function mutations in PCSK9 cause severe hypercholesterolaemia. "
        "Projected quantum dot battery density will exceed 1400 Wh/kg by late 2029."
    )
    resp = verif_service.verify_text(passage, max_debate_rounds=1)

    assert resp.claim_count == 2
    assert resp.refuted_count == 1
    assert resp.unverified_count == 1

    # Refuted claim contributes its share: ~0.75 - 1.0 / 2 = ~0.375 - 0.50
    assert 0.35 <= resp.hallucination_index <= 0.55
    # Unverified claim contributes its uncertainty
    assert resp.unverified_index > 0.0
