"""Unit and integration tests for the FederatedEvidenceRetriever service."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.schemas import EvidenceItem
from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_schema import CorpusDocument
from app.services.federated_evidence_retriever import (
    FederatedEvidenceRetriever,
    get_federated_evidence_retriever,
)
from app.services.free_web_retriever import FreeWebRetriever


@pytest.fixture
def mock_scifact_retriever() -> BM25Retriever:
    """Provide a small in-memory BM25 index simulating SciFact."""
    docs = [
        CorpusDocument(
            id="scifact_1",
            title="Aspirin reduces myocardial infarction and vascular events",
            text="Secondary prevention trials indicate low-dose aspirin reduces vascular mortality.",
            source="PubMed / AllenAI SciFact",
            source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
            domain="ncbi.nlm.nih.gov",
        ),
        CorpusDocument(
            id="scifact_2",
            title="Statins for primary prevention of cardiovascular disease",
            text="HMG-CoA reductase inhibitors reduce LDL cholesterol levels and clinical outcomes.",
            source="PubMed / AllenAI SciFact",
            source_url="https://pubmed.ncbi.nlm.nih.gov/2/",
            domain="ncbi.nlm.nih.gov",
        ),
    ]
    return BM25Retriever.from_documents(docs)


@pytest.fixture
def mock_fever_retriever() -> BM25Retriever:
    """Provide a small in-memory BM25 index simulating FEVER."""
    docs = [
        CorpusDocument(
            id="fever_wiki_tim_berners_lee",
            title="Tim Berners-Lee",
            text="Sir Timothy John Berners-Lee is an English computer scientist best known as the inventor of the World Wide Web.",
            source="FEVER / Wikipedia",
            source_url="https://en.wikipedia.org/wiki/Tim_Berners-Lee",
            domain="en.wikipedia.org",
        )
    ]
    return BM25Retriever.from_documents(docs)


def test_federated_toy_mode():
    """Verify that 'toy' mode dispatches to MockEvidenceRetriever."""
    retriever = FederatedEvidenceRetriever(mode="toy")
    items = retriever.retrieve("Any factual claim", stance="supporting")
    assert len(items) == 2
    assert items[0].retrieval_method == "mock"
    assert "MOCK SUPPORTING EVIDENCE" in items[0].text


def test_federated_scifact_benchmark_mode(mock_scifact_retriever: BM25Retriever):
    """Verify exclusive SciFact benchmark retrieval and normalization."""
    retriever = FederatedEvidenceRetriever(
        mode="benchmark_scifact",
        scifact_retriever=mock_scifact_retriever,
    )
    items = retriever.retrieve("aspirin vascular mortality", stance="supporting")
    assert len(items) >= 1
    top = items[0]
    assert top.retrieval_method == "benchmark_scifact"
    assert top.domain == "ncbi.nlm.nih.gov"
    assert "Aspirin reduces myocardial infarction" in top.source_title
    assert top.relevance_score >= 0.50


def test_federated_fever_benchmark_mode(mock_fever_retriever: BM25Retriever):
    """Verify exclusive FEVER benchmark retrieval and normalization."""
    retriever = FederatedEvidenceRetriever(
        mode="benchmark_fever",
        fever_retriever=mock_fever_retriever,
    )
    items = retriever.retrieve("Tim Berners-Lee inventor World Wide Web", stance="supporting")
    assert len(items) == 1
    top = items[0]
    assert top.retrieval_method == "benchmark_fever"
    assert top.domain == "en.wikipedia.org"
    assert "Tim Berners-Lee" in top.source_title


def test_federated_hybrid_mode_biomedical_hit(
    mock_scifact_retriever: BM25Retriever,
    mock_fever_retriever: BM25Retriever,
):
    """Verify hybrid mode routes biomedical claim to SciFact and avoids web escalation on confident match."""
    mock_web = MagicMock(spec=FreeWebRetriever)

    retriever = FederatedEvidenceRetriever(
        mode="hybrid",
        web_retriever=mock_web,
        scifact_retriever=mock_scifact_retriever,
        fever_retriever=mock_fever_retriever,
        escalation_threshold=0.30,
        top_k=2,
    )

    # Claim containing biomedical keywords ('aspirin', 'mortality', 'vascular')
    items = retriever.retrieve("aspirin reduces vascular mortality", stance="supporting")

    assert len(items) >= 1
    assert items[0].retrieval_method == "benchmark_scifact"
    # Web retriever should NOT have been called because SciFact provided confident evidence
    mock_web.retrieve.assert_not_called()


def test_federated_hybrid_mode_escalation_to_web(
    mock_scifact_retriever: BM25Retriever,
    mock_fever_retriever: BM25Retriever,
):
    """Verify hybrid mode escalates to Live Web when benchmark corpora lack relevant evidence."""
    mock_web = MagicMock(spec=FreeWebRetriever)
    mock_web.retrieve.return_value = [
        EvidenceItem(
            evidence_id="11111111-1111-1111-1111-111111111111",
            text="JWST discovered carbon dioxide on WASP-39b.",
            source_url="https://science.nasa.gov/jwst-wasp39b",
            domain="science.nasa.gov",
            source_title="NASA JWST",
            retrieval_method="free_web_duckduckgo",
            relevance_score=0.95,
        )
    ]

    retriever = FederatedEvidenceRetriever(
        mode="hybrid",
        web_retriever=mock_web,
        scifact_retriever=mock_scifact_retriever,
        fever_retriever=mock_fever_retriever,
        escalation_threshold=0.40,
        top_k=2,
    )

    # Contemporary astronomy claim not present in biomedical SciFact or FEVER toy mock
    items = retriever.retrieve("JWST WASP-39b exoplanet atmosphere 2023", stance="supporting")

    # Should have triggered web escalation
    assert mock_web.retrieve.call_count == 1
    assert len(items) == 1
    assert items[0].retrieval_method == "free_web_duckduckgo"
    assert items[0].domain == "science.nasa.gov"


def test_federated_stance_handling(mock_scifact_retriever: BM25Retriever):
    """Verify supporting vs contradicting stance adaptations and invalid stance rejection."""
    retriever = FederatedEvidenceRetriever(
        mode="benchmark_scifact",
        scifact_retriever=mock_scifact_retriever,
    )

    supp = retriever.retrieve("aspirin mortality", stance="supporting")
    opp = retriever.retrieve("aspirin mortality", stance="contradicting")

    assert len(supp) >= 1
    assert len(opp) >= 1

    with pytest.raises(ValueError, match="Stance must be 'supporting' or 'contradicting'"):
        retriever.retrieve("claim", stance="neutral")


def test_get_federated_evidence_retriever_singleton():
    """Verify singleton accessor returns consistent instance."""
    inst1 = get_federated_evidence_retriever(force_reload=True)
    inst2 = get_federated_evidence_retriever()
    assert inst1 is inst2


def test_federated_real_disk_corpora_and_escalation():
    """Integration test verifying retrieval over real disk JSONL files (SciFact, FEVER) and escalation to Web."""
    retriever = FederatedEvidenceRetriever(mode="hybrid", top_k=3)

    # 1. Real SciFact disk query (biomedical domain)
    scifact_items = retriever.retrieve(
        "Aspirin reduces cardiovascular mortality in randomized clinical trials",
        stance="supporting",
    )
    assert len(scifact_items) >= 1
    assert scifact_items[0].retrieval_method == "benchmark_scifact"
    assert scifact_items[0].domain == "ncbi.nlm.nih.gov"
    assert "Aspirin" in scifact_items[0].source_title or "vascular" in scifact_items[0].text.lower()

    # 2. Real FEVER disk query (general factual domain)
    fever_items = retriever.retrieve(
        "Colin Kaepernick starting quarterback",
        stance="supporting",
    )
    assert len(fever_items) >= 1
    assert fever_items[0].retrieval_method == "benchmark_fever"
    assert fever_items[0].domain == "en.wikipedia.org"
    assert "Colin Kaepernick" in fever_items[0].source_title

