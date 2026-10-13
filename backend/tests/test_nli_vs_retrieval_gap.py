"""Unit tests for NLI vs Retrieval Error Ablation harness."""

from pathlib import Path
import pytest

from app.services.corpus_schema import CorpusDocument
from scripts.evaluate_nli_vs_retrieval_gap import (
    extract_gold_evidence_text,
    load_dataset_and_corpus,
)


def test_extract_gold_evidence_text():
    """Verify extracting exact annotated sentences from corpus map."""
    doc = CorpusDocument(
        id="scifact_100",
        title="Sample Study",
        text="Sentence zero is background. Sentence one has the key finding. Sentence two discusses methodology.",
        source="SciFact",
    )
    corpus_map = {"scifact_100": doc}

    claim = {
        "claim_id": "scifact_1",
        "claim_text": "Key finding was reported.",
        "evidence_doc_ids": ["scifact_100"],
        "raw_evidence": {
            "100": [{"sentences": [1], "label": "SUPPORT"}]
        },
    }

    gold_text = extract_gold_evidence_text(claim, corpus_map)
    assert "Sentence one has the key finding." in gold_text
    assert "Sentence zero" not in gold_text
    assert "Sentence two" not in gold_text


def test_extract_gold_evidence_text_multiple_sentences():
    """Verify extracting multiple non-contiguous annotated sentences."""
    doc = CorpusDocument(
        id="scifact_200",
        title="Multi-sentence Study",
        text="Intro statement. Key fact alpha. Method description. Key fact beta. Concluding remark.",
        source="SciFact",
    )
    corpus_map = {"scifact_200": doc}

    claim = {
        "claim_id": "scifact_2",
        "claim_text": "Alpha and beta occurred.",
        "evidence_doc_ids": ["scifact_200"],
        "raw_evidence": {
            "200": [{"sentences": [1, 3], "label": "SUPPORT"}]
        },
    }

    gold_text = extract_gold_evidence_text(claim, corpus_map)
    assert "Key fact alpha." in gold_text
    assert "Key fact beta." in gold_text
    assert "Method description." not in gold_text


def test_load_dataset_and_corpus_balanced_selection(tmp_path: Path):
    """Verify loading real SciFact files returns balanced classes and valid indices."""
    claims_path = Path("data/scifact_claims.jsonl")
    corpus_path = Path("data/scifact_corpus.jsonl")

    if not claims_path.exists() or not corpus_path.exists():
        pytest.skip("SciFact benchmark files not present in data directory.")

    sample_size = 20
    claims, corpus_map, bm25 = load_dataset_and_corpus(
        claims_path=claims_path,
        corpus_path=corpus_path,
        sample_size=sample_size,
        balance_classes=True,
    )

    assert len(claims) == sample_size
    supports = sum(1 for c in claims if c["gold_label"] == "SUPPORT")
    contradicts = sum(1 for c in claims if c["gold_label"] == "CONTRADICT")
    assert supports == sample_size // 2
    assert contradicts == sample_size // 2

    # Verify corpus map and BM25 index integrity
    assert len(corpus_map) > 5000
    assert len(bm25.documents) == len(corpus_map)


def test_error_partition_taxonomy():
    """Verify exact 4-way error taxonomy classification logic."""
    cases = [
        # (gold_correct, ret_correct, expected_category)
        (True, True, "BOTH_CORRECT"),
        (True, False, "RETRIEVAL_FAILURE"),
        (False, False, "MODEL_FAILURE"),
        (False, True, "FORTUITOUS"),
    ]

    for g_ok, r_ok, expected in cases:
        if g_ok and r_ok:
            cat = "BOTH_CORRECT"
        elif g_ok and not r_ok:
            cat = "RETRIEVAL_FAILURE"
        elif not g_ok and not r_ok:
            cat = "MODEL_FAILURE"
        else:
            cat = "FORTUITOUS"
        assert cat == expected

