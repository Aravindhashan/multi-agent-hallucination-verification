"""Unit tests for NLI engine label mapping resolution and permutation robustness."""

from types import SimpleNamespace
import torch

from app.services.nli_engine import NLIEngine
from app.services.nli_models import NLILabel
from tests.test_nli_engine import MockModel, MockTokenizer


def test_label_mapping_standard():
    """Verify standard {0: entailment, 1: neutral, 2: contradiction} mapping."""
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model = MockModel(id2label=id2label)
    engine = NLIEngine(model=model, tokenizer=MockTokenizer())

    assert engine.label_mapping[0] == NLILabel.ENTAILMENT
    assert engine.label_mapping[1] == NLILabel.NEUTRAL
    assert engine.label_mapping[2] == NLILabel.CONTRADICTION
    assert engine.entailment_idx == 0
    assert engine.neutral_idx == 1
    assert engine.contradiction_idx == 2


def test_label_mapping_permuted_indices():
    """Verify handling when model outputs indices in permuted order."""
    # E.g. 0: contradiction, 1: entailment, 2: neutral
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    # Logits strongly favoring index 1 (entailment)
    logits = torch.tensor([[-2.0, 5.0, -2.0]], dtype=torch.float32)
    model = MockModel(fixed_logits=logits, id2label=id2label)
    engine = NLIEngine(model=model, tokenizer=MockTokenizer())

    assert engine.label_mapping[0] == NLILabel.CONTRADICTION
    assert engine.label_mapping[1] == NLILabel.ENTAILMENT
    assert engine.label_mapping[2] == NLILabel.NEUTRAL
    assert engine.contradiction_idx == 0
    assert engine.entailment_idx == 1
    assert engine.neutral_idx == 2

    res = engine.predict("Claim", "Evidence")
    assert res.label == NLILabel.ENTAILMENT
    assert res.p_entailment > 0.95


def test_label_mapping_case_insensitivity():
    """Verify case-insensitive matching for labels (e.g. ENTAILMENT, CONTRADICTION)."""
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    model = MockModel(id2label=id2label)
    engine = NLIEngine(model=model, tokenizer=MockTokenizer())

    assert engine.label_mapping[0] == NLILabel.CONTRADICTION
    assert engine.label_mapping[1] == NLILabel.NEUTRAL
    assert engine.label_mapping[2] == NLILabel.ENTAILMENT


def test_label_mapping_fallback_on_unrecognized():
    """Verify fallback to standard MNLI mapping if id2label is missing or corrupt."""
    model = MockModel(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    engine = NLIEngine(model=model, tokenizer=MockTokenizer())

    assert engine.label_mapping[0] == NLILabel.ENTAILMENT
    assert engine.label_mapping[1] == NLILabel.NEUTRAL
    assert engine.label_mapping[2] == NLILabel.CONTRADICTION
