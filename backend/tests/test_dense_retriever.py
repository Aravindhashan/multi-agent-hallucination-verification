"""Unit tests for DenseRetriever service using SentenceTransformers and FAISS."""

from pathlib import Path
import pytest

from app.services.corpus_loader import CorpusLoader
from app.services.dense_retriever import DenseRetriever


@pytest.fixture(scope="module")
def shared_corpus_docs():
    """Load default reference corpus documents."""
    loader = CorpusLoader.from_file("data/corpus.json")
    return loader.documents


@pytest.fixture(scope="module")
def shared_dense_retriever(shared_corpus_docs):
    """Build shared DenseRetriever to reuse across tests and avoid re-encoding."""
    return DenseRetriever.from_documents(shared_corpus_docs)


def test_dense_index_builds(shared_dense_retriever, shared_corpus_docs):
    """Verify DenseRetriever builds FAISS index and stores document map."""
    assert len(shared_dense_retriever) == len(shared_corpus_docs)
    assert len(shared_dense_retriever.documents) == len(shared_corpus_docs)


def test_dense_retrieval_semantic_variation(shared_dense_retriever):
    """Verify dense retrieval finds relevant document even when query wording differs."""
    # Query has different wording from "Paris is the official capital and most populous city of France..."
    results = shared_dense_retriever.retrieve("Which city is France's capital?", top_k=3)

    assert len(results) == 3
    top_doc = results[0]
    assert "paris" in top_doc.text.lower()
    assert top_doc.retrieval_method == "DENSE"
    assert top_doc.rank == 1
    assert top_doc.score > 0.0


def test_dense_document_id_mapping(shared_dense_retriever):
    """Verify returned FAISS positions map to valid CorpusDocument IDs."""
    results = shared_dense_retriever.retrieve("photosynthesis in green plants", top_k=5)
    valid_ids = {doc.id for doc in shared_dense_retriever.documents}

    for res in results:
        assert res.document_id in valid_ids
        assert len(res.text) > 0
        assert res.source is not None


def test_dense_top_k_bounds(shared_dense_retriever, shared_corpus_docs):
    """Verify top_k bounds validation and behavior."""
    res_1 = shared_dense_retriever.retrieve("Jupiter largest planet", top_k=1)
    assert len(res_1) == 1

    res_large = shared_dense_retriever.retrieve("Jupiter largest planet", top_k=500)
    assert len(res_large) == len(shared_corpus_docs)

    with pytest.raises(ValueError):
        shared_dense_retriever.retrieve("Jupiter largest planet", top_k=0)

    with pytest.raises(ValueError):
        shared_dense_retriever.retrieve("Jupiter largest planet", top_k=-1)


def test_dense_empty_query_rejected(shared_dense_retriever):
    """Verify empty/whitespace query is rejected."""
    with pytest.raises(ValueError):
        shared_dense_retriever.retrieve("")

    with pytest.raises(ValueError):
        shared_dense_retriever.retrieve("   \t\n  ")


def test_dense_persistence(shared_dense_retriever, tmp_path):
    """Verify FAISS index save and load preserves retrieval functionality."""
    query = "penicillin antibacterial properties"
    original_results = shared_dense_retriever.retrieve(query, top_k=3)

    # Save to tmp directory
    shared_dense_retriever.save(tmp_path / "dense")

    # Load from tmp directory
    loaded_retriever = DenseRetriever.load(
        tmp_path / "dense",
        model=shared_dense_retriever._model,
    )
    loaded_results = loaded_retriever.retrieve(query, top_k=3)

    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.document_id == loaded.document_id
        assert orig.rank == loaded.rank
        assert abs(orig.score - loaded.score) < 1e-4
