"""Unit tests for BM25Retriever service."""

from pathlib import Path
import pytest

from app.services.bm25_retriever import BM25Retriever, tokenize
from app.services.corpus_loader import CorpusLoader


@pytest.fixture
def corpus_docs():
    """Load default reference corpus documents."""
    loader = CorpusLoader.from_file("data/corpus.json")
    return loader.documents


def test_tokenize_deterministic():
    """Verify tokenizer extracts clean lowercase alphanumeric tokens."""
    tokens = tokenize("Paris is the Capital of France! (2023)")
    assert tokens == ["paris", "is", "the", "capital", "of", "france", "2023"]
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_bm25_index_builds(corpus_docs):
    """Verify BM25 index initializes from corpus documents."""
    retriever = BM25Retriever.from_documents(corpus_docs)
    assert len(retriever) == len(corpus_docs)
    assert len(retriever.documents) == len(corpus_docs)


def test_bm25_retrieval_exact_fact(corpus_docs):
    """Verify BM25 retrieves Paris document for 'capital of France'."""
    retriever = BM25Retriever.from_documents(corpus_docs)
    results = retriever.retrieve("capital of France", top_k=3)

    assert len(results) == 3
    top_doc = results[0]
    assert "paris" in top_doc.text.lower()
    assert top_doc.retrieval_method == "BM25"
    assert top_doc.rank == 1
    assert top_doc.score > 0.0


def test_bm25_top_k_variations(corpus_docs):
    """Verify top_k parameter bounds and handling."""
    retriever = BM25Retriever.from_documents(corpus_docs)

    res_1 = retriever.retrieve("speed of light", top_k=1)
    assert len(res_1) == 1

    res_3 = retriever.retrieve("speed of light", top_k=3)
    assert len(res_3) == 3

    res_large = retriever.retrieve("speed of light", top_k=1000)
    assert len(res_large) == len(corpus_docs)

    with pytest.raises(ValueError) as excinfo:
        retriever.retrieve("speed of light", top_k=0)
    assert "top_k must be a positive integer" in str(excinfo.value)

    with pytest.raises(ValueError):
        retriever.retrieve("speed of light", top_k=-5)


def test_bm25_empty_query_rejected(corpus_docs):
    """Verify empty or whitespace-only query is rejected with ValueError."""
    retriever = BM25Retriever.from_documents(corpus_docs)

    with pytest.raises(ValueError):
        retriever.retrieve("")

    with pytest.raises(ValueError):
        retriever.retrieve("    ")


def test_bm25_persistence(corpus_docs, tmp_path):
    """Verify save and load cycle produces identical retrieval results."""
    retriever = BM25Retriever.from_documents(corpus_docs)
    query = "Apollo 11 moon landing"
    original_results = retriever.retrieve(query, top_k=3)

    # Save index to temp directory
    retriever.save(tmp_path / "bm25")

    # Load from saved index
    loaded_retriever = BM25Retriever.load(tmp_path / "bm25")
    loaded_results = loaded_retriever.retrieve(query, top_k=3)

    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.document_id == loaded.document_id
        assert orig.rank == loaded.rank
        assert abs(orig.score - loaded.score) < 1e-5
