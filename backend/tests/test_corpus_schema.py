"""Unit tests for explicit reference corpus schema and validation constraints."""

from datetime import date
import pytest
from pydantic import ValidationError

from app.services.corpus_schema import Corpus, CorpusDocument


def test_valid_corpus_document_full_metadata():
    """Verify valid document with all optional metadata succeeds."""
    doc = CorpusDocument(
        id="doc_001",
        title="Earth Orbit",
        text="The Earth orbits around the Sun once every 365.25 days.",
        source="Astronomy Reference",
        source_url="https://astronomy.example.org/earth-orbit",
        domain="astronomy.example.org",
        published_date=date(2023, 5, 10),
    )
    assert doc.id == "doc_001"
    assert doc.title == "Earth Orbit"
    assert doc.published_date == date(2023, 5, 10)
    assert doc.domain == "astronomy.example.org"


def test_valid_corpus_document_optional_omitted():
    """Verify valid document succeeds when optional fields are omitted."""
    doc = CorpusDocument(
        id="doc_002",
        title="Water Boiling Point",
        text="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        source="Chemistry Handbook",
    )
    assert doc.id == "doc_002"
    assert doc.source_url is None
    assert doc.domain is None
    assert doc.published_date is None


def test_valid_corpus_model():
    """Verify valid Corpus model instantiation succeeds."""
    corpus = Corpus(
        version="1.0",
        documents=[
            CorpusDocument(
                id="doc_001",
                title="Doc 1",
                text="Text 1",
                source="Source 1",
            ),
            CorpusDocument(
                id="doc_002",
                title="Doc 2",
                text="Text 2",
                source="Source 2",
            ),
        ],
    )
    assert corpus.version == "1.0"
    assert len(corpus.documents) == 2


def test_missing_version_fails():
    """Verify missing version fails validation."""
    with pytest.raises(ValidationError):
        Corpus.model_validate({
            "documents": [
                {"id": "d1", "title": "T", "text": "X", "source": "S"}
            ]
        })


def test_empty_version_fails():
    """Verify empty or whitespace-only version fails validation."""
    with pytest.raises(ValidationError):
        Corpus.model_validate({
            "version": "   ",
            "documents": [
                {"id": "d1", "title": "T", "text": "X", "source": "S"}
            ]
        })


def test_missing_documents_fails():
    """Verify missing documents fails validation."""
    with pytest.raises(ValidationError):
        Corpus.model_validate({"version": "1.0"})


def test_documents_not_a_list_fails():
    """Verify documents as non-list fails validation."""
    with pytest.raises(ValidationError):
        Corpus.model_validate({"version": "1.0", "documents": "not-a-list"})


def test_missing_document_id_fails():
    """Verify document with missing ID fails validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "title": "Title",
            "text": "Text",
            "source": "Source",
        })


def test_empty_document_id_fails():
    """Verify whitespace-only document ID fails validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "id": "   ",
            "title": "Title",
            "text": "Text",
            "source": "Source",
        })


def test_empty_document_title_fails():
    """Verify whitespace-only document title fails validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "id": "doc_01",
            "title": "   ",
            "text": "Text",
            "source": "Source",
        })


def test_empty_document_text_fails():
    """Verify empty or whitespace-only document text fails validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "id": "doc_01",
            "title": "Title",
            "text": "   ",
            "source": "Source",
        })


def test_missing_source_fails():
    """Verify missing or empty source fails validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "id": "doc_01",
            "title": "Title",
            "text": "Text",
            "source": "  ",
        })


def test_duplicate_document_ids_fails():
    """Verify duplicate document IDs fail corpus validation."""
    with pytest.raises(ValidationError) as excinfo:
        Corpus.model_validate({
            "version": "1.0",
            "documents": [
                {"id": "dup_id", "title": "Doc A", "text": "Text A", "source": "Src A"},
                {"id": "dup_id", "title": "Doc B", "text": "Text B", "source": "Src B"},
            ]
        })
    assert "Duplicate document ID" in str(excinfo.value)


def test_invalid_published_date_fails():
    """Verify invalid date strings fail validation."""
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate({
            "id": "doc_01",
            "title": "Title",
            "text": "Text",
            "source": "Source",
            "published_date": "2024-99-99",
        })


def test_duplicate_text_with_different_ids_allowed():
    """Verify exact or near duplicate text with different IDs passes validation."""
    corpus = Corpus.model_validate({
        "version": "1.0",
        "documents": [
            {
                "id": "doc_wire_01",
                "title": "Syndicated Article 1",
                "text": "Breaking news: Major scientific breakthrough announced today.",
                "source": "News Wire A",
            },
            {
                "id": "doc_wire_02",
                "title": "Syndicated Article 2",
                "text": "Breaking news: Major scientific breakthrough announced today.",
                "source": "News Wire B",
            },
        ]
    })
    assert len(corpus.documents) == 2
    assert corpus.documents[0].text == corpus.documents[1].text
