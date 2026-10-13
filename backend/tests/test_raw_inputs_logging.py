"""Unit and integration tests for logging and persisting raw NLI inputs.

Verifies that for every verified claim:
1. The exact premise text sent to DeBERTa is saved.
2. The truncated token length and raw token length are tracked.
3. The three probabilities (p_entailment, p_neutral, p_contradiction) are preserved.
4. AdjudicationItemResult, EvidenceSnippetSummary, and ClaimVerificationSummary reflect these fields.
5. Logger logs the raw input parameters during DeBERTa cross-encoder inference.
"""

import logging
from types import SimpleNamespace
import pytest
import torch

from app.schemas import (
    AdjudicationItemResult,
    AdjudicationResponse,
    AdjudicationVerdict,
    ClaimVerificationSummary,
    EvidenceSnippetSummary,
)
from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import NLILabel, NLIResult
from tests.test_nli_engine import MockModel, MockTokenizer


def test_nli_engine_records_raw_inputs_and_token_length():
    """Verify NLIEngine populates premise_text, token_length, raw_token_length, and truncated."""
    mock_engine = NLIEngine(
        config=NLISettings(max_length=20, truncation_policy="truncate"),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )

    claim = "Apollo 11 landed on the Moon in July 1969."
    evidence = (
        "The Apollo 11 lunar module Eagle landed on the Moon on July 20, 1969, "
        "carrying astronauts Neil Armstrong and Buzz Aldrin."
    )

    result = mock_engine.predict(claim=claim, evidence=evidence)

    assert result.premise_text == evidence
    assert result.token_length is not None
    assert result.token_length > 0
    assert result.raw_token_length is not None
    assert result.raw_token_length >= result.token_length
    assert result.p_entailment >= 0.0
    assert result.p_neutral >= 0.0
    assert result.p_contradiction >= 0.0


def test_nli_engine_logging_captures_raw_inputs(caplog):
    """Verify NLIEngine logger logs premise text, token lengths, and 3-class probabilities."""
    mock_engine = NLIEngine(
        config=NLISettings(max_length=50, truncation_policy="truncate"),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )

    claim = "Water boils at 100C."
    evidence = "Standard boiling point of water is 100 degrees Celsius."

    with caplog.at_level(logging.INFO, logger="hallucination_verifier.nli_engine"):
        res = mock_engine.predict(claim=claim, evidence=evidence)

    assert res.premise_text == evidence
    # Verify log output contains raw inputs
    log_text = caplog.text
    assert "DeBERTa NLI Raw Input & Prediction:" in log_text
    assert "[Claim (Hypothesis)]: Water boils at 100C." in log_text
    assert "[Premise (Evidence)]: Standard boiling point of water is 100 degrees Celsius." in log_text
    assert "[Token Length]:" in log_text
    assert "[Probabilities]: Entailment=" in log_text


def test_adjudication_item_result_and_summary_schemas():
    """Verify AdjudicationItemResult and EvidenceSnippetSummary preserve raw DeBERTa inputs."""
    adj_item = AdjudicationItemResult(
        evidence_id="ev_001",
        text="Alexander Fleming discovered penicillin in 1928 at St Mary Hospital.",
        premise_text="Alexander Fleming discovered penicillin in 1928 at St Mary Hospital.",
        token_length=14,
        raw_token_length=14,
        truncated=False,
        domain="nih.gov",
        source_url="https://nih.gov/penicillin",
        source_title="History of Penicillin",
        stance="supporting",
        cluster_reliability_weight=0.95,
        p_entailment=0.9850,
        p_contradiction=0.0020,
        p_neutral=0.0130,
        label="entailment",
        confidence=0.9850,
        inference_time_ms=12.5,
    )

    assert adj_item.premise_text == "Alexander Fleming discovered penicillin in 1928 at St Mary Hospital."
    assert adj_item.token_length == 14
    assert adj_item.p_entailment == 0.9850
    assert adj_item.p_neutral == 0.0130
    assert adj_item.p_contradiction == 0.0020

    snippet = EvidenceSnippetSummary(
        text=adj_item.text,
        premise_text=adj_item.premise_text,
        token_length=adj_item.token_length,
        p_entailment=adj_item.p_entailment,
        p_neutral=adj_item.p_neutral,
        p_contradiction=adj_item.p_contradiction,
        source_title=adj_item.source_title,
        source_url=adj_item.source_url,
        domain=adj_item.domain,
        stance="supporting",
        confidence=0.9850,
    )

    assert snippet.premise_text == adj_item.premise_text
    assert snippet.token_length == 14
    assert snippet.p_entailment == 0.9850

    claim_summary = ClaimVerificationSummary(
        claim_id="c_001",
        claim_text="Alexander Fleming discovered penicillin.",
        order=0,
        verdict=AdjudicationVerdict.SUPPORTED,
        confidence_score=0.9850,
        support_probability=0.9850,
        contradiction_probability=0.0020,
        neutral_probability=0.0130,
        supportive_claims=[snippet],
        opposing_claims=[],
        raw_nli_inputs=[
            {
                "evidence_id": adj_item.evidence_id,
                "stance": adj_item.stance,
                "premise_text": adj_item.premise_text,
                "token_length": adj_item.token_length,
                "raw_token_length": adj_item.raw_token_length,
                "truncated": adj_item.truncated,
                "p_entailment": adj_item.p_entailment,
                "p_neutral": adj_item.p_neutral,
                "p_contradiction": adj_item.p_contradiction,
                "label": adj_item.label,
                "confidence": adj_item.confidence,
            }
        ],
    )

    assert len(claim_summary.raw_nli_inputs) == 1
    raw_saved = claim_summary.raw_nli_inputs[0]
    assert raw_saved["premise_text"] == adj_item.premise_text
    assert raw_saved["token_length"] == 14
    assert raw_saved["p_entailment"] == 0.9850
    assert raw_saved["p_neutral"] == 0.0130
    assert raw_saved["p_contradiction"] == 0.0020
