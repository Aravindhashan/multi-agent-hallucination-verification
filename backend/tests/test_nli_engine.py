"""Unit and integration tests for DeBERTa Cross-Encoder NLI Engine."""

from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine, get_nli_engine, reset_nli_engine
from app.services.nli_models import (
    NLIConfigurationError,
    NLIInferenceError,
    NLIInputError,
    NLILabel,
    NLIModelLoadError,
    NLIResult,
)


# ---------------------------------------------------------------------------
# Helpers & Mock Fixtures
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Mock Hugging Face tokenizer for fast, deterministic unit testing."""

    def __init__(self, max_allowed_tokens: int = 512):
        self.max_allowed_tokens = max_allowed_tokens

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
            hypotheses = [hypotheses] if hypotheses is not None else [None]

        input_ids_list = []
        for i, p in enumerate(premises):
            h = hypotheses[i] if hypotheses else ""
            total_words = len(str(p).split()) + len(str(h or "").split()) + 2
            # Create synthetic token IDs
            if truncation and max_length and total_words > max_length:
                seq_len = max_length
            else:
                seq_len = total_words
            input_ids_list.append(list(range(seq_len)))

        if return_tensors == "pt":
            max_len = max(len(ids) for ids in input_ids_list)
            padded_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids_list]
            mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }

        return {"input_ids": input_ids_list}


class MockModel:
    """Mock PyTorch sequence classification model."""

    def __init__(self, fixed_logits: torch.Tensor | None = None, id2label: dict | None = None):
        self.fixed_logits = fixed_logits
        self.config = SimpleNamespace(
            id2label=id2label
            or {0: "entailment", 1: "neutral", 2: "contradiction"}
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        if self.fixed_logits is not None:
            logits = self.fixed_logits
            if logits.shape[0] != batch_size:
                logits = logits.repeat(batch_size, 1)
        else:
            # Default to strong entailment [5.0, -2.0, -2.0]
            logits = torch.tensor([[5.0, -2.0, -2.0]], dtype=torch.float32).repeat(batch_size, 1)

        return SimpleNamespace(logits=logits)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Ensure singleton instance is cleared before and after each test."""
    import gc
    reset_nli_engine()
    gc.collect()
    yield
    reset_nli_engine()
    gc.collect()


# ---------------------------------------------------------------------------
# Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_nli_config_defaults():
    config = NLISettings()
    assert config.model_name == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    assert config.device in {"auto", "cpu", "cuda", "mps"}
    assert config.max_length == 256
    assert config.batch_size == 8
    assert config.max_claim_chars == 500
    assert config.max_evidence_chars == 1000
    assert config.truncation_policy == "truncate"


def test_nli_config_invalid_parameters():
    with pytest.raises(NLIConfigurationError):
        NLISettings(model_name="  ")

    with pytest.raises(NLIConfigurationError):
        NLISettings(device="invalid_dev")

    with pytest.raises(NLIConfigurationError):
        NLISettings(max_length=0)

    with pytest.raises(NLIConfigurationError):
        NLISettings(batch_size=-1)

    with pytest.raises(NLIConfigurationError):
        NLISettings(max_claim_chars=0)

    with pytest.raises(NLIConfigurationError):
        NLISettings(max_evidence_chars=0)

    with pytest.raises(NLIConfigurationError):
        NLISettings(truncation_policy="invalid_policy")


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

def test_nli_engine_rejects_invalid_inputs():
    mock_engine = NLIEngine(
        config=NLISettings(max_claim_chars=50, max_evidence_chars=100, truncation_policy="strict"),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )

    # None inputs
    with pytest.raises(NLIInputError):
        mock_engine.predict(None, "Valid evidence")  # type: ignore

    with pytest.raises(NLIInputError):
        mock_engine.predict("Valid claim", None)  # type: ignore

    # Empty / whitespace
    with pytest.raises(NLIInputError):
        mock_engine.predict("", "Valid evidence")

    with pytest.raises(NLIInputError):
        mock_engine.predict("   ", "Valid evidence")

    with pytest.raises(NLIInputError):
        mock_engine.predict("Valid claim", "")

    with pytest.raises(NLIInputError):
        mock_engine.predict("Valid claim", " \t \n ")

    # Oversized inputs
    with pytest.raises(NLIInputError):
        mock_engine.predict("A" * 51, "Valid evidence")

    with pytest.raises(NLIInputError):
        mock_engine.predict("Valid claim", "B" * 101)


def test_nli_engine_strict_truncation_raises():
    mock_engine = NLIEngine(
        config=NLISettings(max_length=10, truncation_policy="strict"),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )
    long_claim = "word " * 20
    long_evidence = "evidence " * 20

    with pytest.raises(NLIInputError, match="exceeds max_length"):
        mock_engine.predict(long_claim, long_evidence)


def test_nli_engine_truncate_policy_succeeds():
    mock_engine = NLIEngine(
        config=NLISettings(max_length=10, truncation_policy="truncate"),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )
    long_claim = "word " * 20
    long_evidence = "evidence " * 20

    result = mock_engine.predict(long_claim, long_evidence)
    assert result.truncated is True
    assert result.label == NLILabel.ENTAILMENT


# ---------------------------------------------------------------------------
# Probability Contract & Data Model Tests
# ---------------------------------------------------------------------------

def test_nli_result_valid_contract():
    result = NLIResult(
        claim="The sky is blue.",
        evidence="The clear sky appeared bright blue.",
        label=NLILabel.ENTAILMENT,
        confidence=0.95,
        p_entailment=0.95,
        p_neutral=0.03,
        p_contradiction=0.02,
        model_name="mock-deberta",
        device="cpu",
    )
    assert result.label == NLILabel.ENTAILMENT
    assert result.confidence == 0.95
    assert result.scores[NLILabel.ENTAILMENT.value] == 0.95


def test_nli_result_invalid_probabilities_raises():
    # Probabilities don't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        NLIResult(
            claim="Claim",
            evidence="Evidence",
            label=NLILabel.ENTAILMENT,
            confidence=0.5,
            p_entailment=0.5,
            p_neutral=0.2,
            p_contradiction=0.1,  # sum = 0.8
            model_name="mock",
            device="cpu",
        )

    # Label doesn't match argmax
    with pytest.raises(ValueError, match="does not match argmax"):
        NLIResult(
            claim="Claim",
            evidence="Evidence",
            label=NLILabel.ENTAILMENT,
            confidence=0.8,
            p_entailment=0.1,
            p_neutral=0.1,
            p_contradiction=0.8,  # argmax is CONTRADICTION
            model_name="mock",
            device="cpu",
        )

    # Confidence doesn't match maximum probability
    with pytest.raises(ValueError, match="does not match maximum"):
        NLIResult(
            claim="Claim",
            evidence="Evidence",
            label=NLILabel.CONTRADICTION,
            confidence=0.5,  # Should be 0.8
            p_entailment=0.1,
            p_neutral=0.1,
            p_contradiction=0.8,
            model_name="mock",
            device="cpu",
        )


def test_nli_property_random_logits():
    """Property-based verification: 100 random logit vectors produce strictly valid NLIResults."""
    np.random.seed(42)
    for _ in range(100):
        logits = torch.tensor(np.random.randn(1, 3), dtype=torch.float32)
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).tolist()

        p_e, p_n, p_c = probs[0], probs[1], probs[2]
        scores = {
            NLILabel.ENTAILMENT: p_e,
            NLILabel.NEUTRAL: p_n,
            NLILabel.CONTRADICTION: p_c,
        }
        best_lbl = max(scores, key=scores.get)
        best_conf = scores[best_lbl]

        result = NLIResult(
            claim="Random claim",
            evidence="Random evidence",
            label=best_lbl,
            confidence=best_conf,
            p_entailment=p_e,
            p_neutral=p_n,
            p_contradiction=p_c,
            model_name="random-test",
            device="cpu",
        )
        assert 0.0 <= result.p_entailment <= 1.0
        assert 0.0 <= result.p_neutral <= 1.0
        assert 0.0 <= result.p_contradiction <= 1.0
        assert abs(result.p_entailment + result.p_neutral + result.p_contradiction - 1.0) < 1e-4
        assert result.confidence == max(result.p_entailment, result.p_neutral, result.p_contradiction)


# ---------------------------------------------------------------------------
# Single & Batch Inference Tests
# ---------------------------------------------------------------------------

def test_predict_entailment():
    logits = torch.tensor([[4.5, -2.0, -2.5]], dtype=torch.float32)
    engine = NLIEngine(model=MockModel(fixed_logits=logits), tokenizer=MockTokenizer())
    result = engine.predict("Claim", "Evidence")

    assert result.label == NLILabel.ENTAILMENT
    assert result.p_entailment > 0.95
    assert result.p_contradiction < 0.05
    assert result.p_neutral < 0.05


def test_predict_contradiction():
    logits = torch.tensor([[-3.0, -2.0, 5.0]], dtype=torch.float32)
    engine = NLIEngine(model=MockModel(fixed_logits=logits), tokenizer=MockTokenizer())
    result = engine.predict("Claim", "Evidence")

    assert result.label == NLILabel.CONTRADICTION
    assert result.p_contradiction > 0.95
    assert result.p_entailment < 0.05
    assert result.p_neutral < 0.05


def test_predict_neutral():
    logits = torch.tensor([[-2.0, 4.0, -2.0]], dtype=torch.float32)
    engine = NLIEngine(model=MockModel(fixed_logits=logits), tokenizer=MockTokenizer())
    result = engine.predict("Claim", "Evidence")

    assert result.label == NLILabel.NEUTRAL
    assert result.p_neutral > 0.95
    assert result.p_entailment < 0.05
    assert result.p_contradiction < 0.05


def test_predict_batch_order_and_chunking():
    engine = NLIEngine(
        config=NLISettings(batch_size=2),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )
    pairs = [
        ("Claim 1", "Evidence 1"),
        ("Claim 2", "Evidence 2"),
        ("Claim 3", "Evidence 3"),
        ("Claim 4", "Evidence 4"),
        ("Claim 5", "Evidence 5"),
    ]
    results = engine.predict_batch(pairs)

    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.claim == f"Claim {i+1}"
        assert r.evidence == f"Evidence {i+1}"


def test_predict_batch_empty():
    engine = NLIEngine(model=MockModel(), tokenizer=MockTokenizer())
    assert engine.predict_batch([]) == []


def test_predict_batch_prevalidation_failure():
    engine = NLIEngine(model=MockModel(), tokenizer=MockTokenizer())
    invalid_pairs = [
        ("Claim 1", "Evidence 1"),
        ("   ", "Evidence 2"),  # Invalid
        ("Claim 3", "Evidence 3"),
    ]
    with pytest.raises(NLIInputError):
        engine.predict_batch(invalid_pairs)


# ---------------------------------------------------------------------------
# Error Handling & Edge Cases
# ---------------------------------------------------------------------------

def test_inference_error_on_forward_pass_crash():
    bad_model = MagicMock()
    bad_model.config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    bad_model.to.return_value = bad_model
    bad_model.eval.return_value = bad_model
    bad_model.side_effect = RuntimeError("CUDA out of memory simulation")

    engine = NLIEngine(model=bad_model, tokenizer=MockTokenizer())
    with pytest.raises(NLIInferenceError, match="forward pass failed"):
        engine.predict("Claim", "Evidence")


def test_inference_error_on_nan_logits():
    nan_logits = torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float32)
    engine = NLIEngine(model=MockModel(fixed_logits=nan_logits), tokenizer=MockTokenizer())
    with pytest.raises(NLIInferenceError, match="NaN or Inf"):
        engine.predict("Claim", "Evidence")


def test_model_load_error():
    engine = NLIEngine(config=NLISettings(model_name="nonexistent/model-checkpoint-xyz"), lazy_load=True)
    with pytest.raises(NLIModelLoadError):
        engine.predict("Claim", "Evidence")


# ---------------------------------------------------------------------------
# Singleton Lifecycle Tests
# ---------------------------------------------------------------------------

def test_singleton_get_and_reset():
    engine1 = get_nli_engine()
    engine2 = get_nli_engine()
    assert engine1 is engine2

    reset_nli_engine()
    engine3 = get_nli_engine()
    assert engine1 is not engine3


# ---------------------------------------------------------------------------
# Real Model Live Probe Tests (Marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_deberta_nli_probes():
    """Verify live DeBERTa-v3 checkpoint accuracy on 3 unambiguous gold pairs."""
    engine = get_nli_engine()

    premise = "Angela Merkel served as the Chancellor of Germany from 2005 to 2021."

    # Entailment probe
    res_e = engine.predict("Angela Merkel was the Chancellor of Germany.", premise)
    assert res_e.label == NLILabel.ENTAILMENT
    assert res_e.p_entailment > 0.85
    assert res_e.confidence == res_e.p_entailment

    # Contradiction probe
    res_c = engine.predict("Angela Merkel was never involved in German politics.", premise)
    assert res_c.label == NLILabel.CONTRADICTION
    assert res_c.p_contradiction > 0.85
    assert res_c.confidence == res_c.p_contradiction

    # Neutral probe
    res_n = engine.predict("Angela Merkel enjoys drinking black tea every morning.", premise)
    assert res_n.label == NLILabel.NEUTRAL
    assert res_n.p_neutral > 0.85
    assert res_n.confidence == res_n.p_neutral
