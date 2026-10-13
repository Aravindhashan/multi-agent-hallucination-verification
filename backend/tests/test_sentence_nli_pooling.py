"""Unit tests for sentence-level NLI scoring and max-pooling."""

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import torch

from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import DocumentNLIPoolResult, NLIInputError, NLILabel, SentencePrediction
from tests.test_nli_engine import MockModel, MockTokenizer


@pytest.fixture
def mock_sentence_engine():
    """Build an NLIEngine wired with mocks that can simulate custom logits per sentence."""
    settings = NLISettings(
        model_name="test-deberta",
        device="cpu",
        batch_size=8,
        enable_sentence_windows=True,
    )
    engine = NLIEngine(settings)
    engine.is_loaded = True
    engine.tokenizer = MockTokenizer()
    engine.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    engine.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    engine.entailment_idx = 0
    engine.neutral_idx = 1
    engine.contradiction_idx = 2
    return engine


def test_sentence_nli_entailment_max_pooling(mock_sentence_engine):
    """Test that max-pooling across sentences picks the single decisive entailing sentence."""
    claim = "ADAR1 binds to Dicer."
    abstract = (
        "MicroRNAs regulate post-transcriptional gene expression in cells. "
        "In this study, ADAR1 was found to bind directly to Dicer to enhance cleavage. "
        "Overall cellular viability remained unchanged."
    )

    # Mock model output: sentence 0 is neutral, sentence 1 is entailment, sentence 2 is neutral
    # Class order: [entailment, neutral, contradiction]
    logits = torch.tensor([
        [-2.0, 5.0, -3.0],   # sent 0: neutral
        [6.0, -2.0, -4.0],   # sent 1: strong entailment
        [-1.0, 4.0, -2.0],   # sent 2: neutral
    ])
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = SimpleNamespace(logits=logits)
    mock_sentence_engine.model = mock_model

    result = mock_sentence_engine.predict_document_sentences(claim, abstract, document_id="doc_1")

    assert isinstance(result, DocumentNLIPoolResult)
    assert result.num_sentences_evaluated == 3
    assert result.winning_label == NLILabel.ENTAILMENT
    assert result.p_entailment_max > 0.99
    assert result.winning_sentence is not None
    assert result.winning_sentence.sentence_idx == 1
    assert "ADAR1 was found to bind directly to Dicer" in result.winning_sentence.sentence_text
    assert result.winning_sentence.doc_id == "doc_1"


def test_sentence_nli_contradiction_max_pooling(mock_sentence_engine):
    """Test that max-pooling across sentences picks the single decisive refuting sentence."""
    claim = "Gene expression does not vary across identical cells."
    abstract = (
        "Cellular models provide insight into molecular dynamics. "
        "Surprisingly, genetically identical animals exhibited a high degree of variation in aggregate formation. "
        "This suggests stochastic fluctuations dominate expression."
    )

    # Mock model output: sentence 0 neutral, sentence 1 contradiction, sentence 2 neutral
    logits = torch.tensor([
        [-1.0, 4.0, -2.0],   # sent 0: neutral
        [-4.0, -2.0, 6.0],   # sent 1: strong contradiction
        [-1.5, 3.5, -1.0],   # sent 2: neutral
    ])
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = SimpleNamespace(logits=logits)
    mock_sentence_engine.model = mock_model

    result = mock_sentence_engine.predict_document_sentences(claim, abstract, document_id="doc_2")

    assert result.winning_label == NLILabel.CONTRADICTION
    assert result.p_contradiction_max > 0.99
    assert result.winning_sentence is not None
    assert result.winning_sentence.sentence_idx == 1
    assert "genetically identical animals exhibited a high degree of variation" in result.winning_sentence.sentence_text


def test_sentence_nli_all_neutral_defaults_to_neutral(mock_sentence_engine):
    """Test that when all sentences are neutral, the max-pool result is neutral."""
    claim = "Aspirin cures lung cancer."
    abstract = (
        "Cardiovascular diseases represent a leading cause of mortality. "
        "Physical exercise improves vascular endothelial function in adults."
    )

    logits = torch.tensor([
        [-3.0, 5.0, -3.0],   # sent 0: neutral
        [-2.0, 4.0, -2.0],   # sent 1: neutral
    ])
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = SimpleNamespace(logits=logits)
    mock_sentence_engine.model = mock_model

    result = mock_sentence_engine.predict_document_sentences(claim, abstract)

    assert result.winning_label == NLILabel.NEUTRAL
    assert result.p_entailment_max < 0.1
    assert result.p_contradiction_max < 0.1
    assert result.winning_sentence is None


def test_sentence_nli_multi_document_pooling(mock_sentence_engine):
    """Test max-pooling across multiple retrieved documents (e.g. top-3 or top-5)."""
    claim = "Metformin resolves lung fibrosis."
    doc_1 = "Metformin is a common anti-diabetic medication with well-characterized pharmacokinetics."
    doc_2 = "In mice with bleomycin-induced lung fibrosis, metformin accelerated the resolution of established fibrosis."
    doc_3 = "Renal function must be monitored during prolonged metformin therapy."

    # doc 1: neutral (1 sent), doc 2: entailment (1 sent), doc 3: neutral (1 sent)
    logits = torch.tensor([
        [-2.0, 4.0, -2.0],   # doc 1 sent 0
        [5.5, -2.0, -3.0],   # doc 2 sent 0 (entailment)
        [-1.0, 3.0, -1.0],   # doc 3 sent 0
    ])
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = SimpleNamespace(logits=logits)
    mock_sentence_engine.model = mock_model

    docs = [("scifact_101", doc_1), ("scifact_102", doc_2), ("scifact_103", doc_3)]
    result = mock_sentence_engine.predict_multi_document_sentences(claim, docs)

    assert result.num_documents == 3
    assert result.num_sentences_evaluated == 3
    assert result.winning_label == NLILabel.ENTAILMENT
    assert result.winning_sentence is not None
    assert result.winning_sentence.doc_id == "scifact_102"
    assert "accelerated the resolution of established fibrosis" in result.winning_sentence.sentence_text


def test_sentence_nli_empty_inputs(mock_sentence_engine):
    """Test edge cases with empty text."""
    with pytest.raises(NLIInputError):
        mock_sentence_engine.predict_document_sentences("", "Valid abstract.")

    res = mock_sentence_engine.predict_document_sentences("Valid claim.", "")
    assert res.num_sentences_evaluated == 0
    assert res.winning_label == NLILabel.NEUTRAL
    assert res.winning_confidence == 1.0
