"""Unit tests and factual sanity checks for HybridRetriever service."""

from pathlib import Path
import pytest

from app.services.corpus_loader import CorpusLoader
from app.services.dense_retriever import DenseRetriever
from app.services.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def shared_corpus_docs():
    """Load default reference corpus documents."""
    loader = CorpusLoader.from_file("data/corpus.json")
    return loader.documents


@pytest.fixture(scope="module")
def shared_hybrid_retriever(shared_corpus_docs):
    """Build shared HybridRetriever to reuse across tests."""
    import app.services.hybrid_retriever as hr
    retriever = HybridRetriever.from_documents(shared_corpus_docs)
    hr._hybrid_retriever_instance = retriever
    return retriever


def test_hybrid_candidate_merging(shared_hybrid_retriever):
    """Verify hybrid candidate set combines BM25 and Dense matches with deduplication."""
    results = shared_hybrid_retriever.retrieve_hybrid("capital of France", top_k=5)

    assert len(results) <= 5
    assert len(results) > 0

    # Ensure document IDs are unique in the returned candidate set
    doc_ids = [res.document_id for res in results]
    assert len(doc_ids) == len(set(doc_ids))

    # Check that the top candidate is a Paris document and was found by both methods
    top_cand = results[0]
    assert "paris" in top_cand.text.lower()
    assert "BM25" in top_cand.matched_methods
    assert "DENSE" in top_cand.matched_methods
    assert top_cand.bm25_score is not None
    assert top_cand.dense_score is not None
    assert top_cand.combined_rank == 1


def test_duplicate_documents_preservation(shared_hybrid_retriever):
    """Verify near-duplicate documents with different IDs remain distinct in retrieval.

    This ensures downstream evidence-independence modules receive all distinct documents.
    """
    # In corpus, doc_geo_001, doc_geo_002, and doc_geo_003_dup cover Paris/France
    results = shared_hybrid_retriever.retrieve_hybrid("Paris is the official capital of France", top_k=10)
    retrieved_ids = {res.document_id for res in results}

    # Verify that distinct document IDs covering the same fact are retrieved as separate candidates
    france_ids = {"doc_geo_001", "doc_geo_002", "doc_geo_003_dup"}
    overlap = france_ids.intersection(retrieved_ids)
    assert len(overlap) >= 2


def test_hybrid_independent_methods(shared_hybrid_retriever):
    """Verify BM25-only and Dense-only retrieval can be called independently."""
    bm25_res = shared_hybrid_retriever.retrieve_bm25("speed of light", top_k=3)
    dense_res = shared_hybrid_retriever.retrieve_dense("speed of light", top_k=3)

    assert len(bm25_res) == 3
    assert len(dense_res) == 3
    assert all(r.retrieval_method == "BM25" for r in bm25_res)
    assert all(r.retrieval_method == "DENSE" for r in dense_res)


def test_hybrid_base_evidence_retriever_adapter(shared_hybrid_retriever):
    """Verify retrieve() implementation for BaseEvidenceRetriever compatibility."""
    evidence_items = shared_hybrid_retriever.retrieve(
        claim_text="Python was created by Guido van Rossum",
        stance="supporting",
    )
    assert len(evidence_items) > 0
    top_item = evidence_items[0]
    assert "python" in top_item.text.lower()
    assert top_item.relevance_score >= 0.0 and top_item.relevance_score <= 1.0


def test_hybrid_persistence(shared_hybrid_retriever, tmp_path):
    """Verify saving and loading entire hybrid index directory."""
    query = "Tim Berners-Lee World Wide Web 1989"
    original_results = shared_hybrid_retriever.retrieve_hybrid(query, top_k=3)

    # Save to temp directory
    shared_hybrid_retriever.save(tmp_path / "indexes")

    # Load from temp directory
    loaded_hybrid = HybridRetriever.load(tmp_path / "indexes")
    loaded_results = loaded_hybrid.retrieve_hybrid(query, top_k=3)

    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.document_id == loaded.document_id
        assert orig.combined_rank == loaded.combined_rank
        assert orig.matched_methods == loaded.matched_methods


@pytest.mark.parametrize(
    "query,expected_keyword",
    [
        ("capital of France", "paris"),
        ("speed of light in vacuum", "299,792,458"),
        ("Apollo 11 moon landing", "armstrong"),
        ("Python programming language", "rossum"),
        ("discovery of penicillin", "fleming"),
        ("Jupiter largest planet", "jupiter"),
    ],
)
def test_factual_retrieval_sanity_check(shared_hybrid_retriever, query, expected_keyword):
    """Deterministic sanity check: verify relevant document appears in top-3 candidates."""
    candidates = shared_hybrid_retriever.retrieve_hybrid(query, top_k=3)
    assert len(candidates) > 0

    combined_text = " ".join([c.text.lower() + " " + c.title.lower() for c in candidates])
    assert expected_keyword.lower() in combined_text, (
        f"Expected keyword '{expected_keyword}' not found in top-3 retrieved candidates for query '{query}'"
    )

