"""Unit tests for CorpusLoader service."""

import json
from pathlib import Path
import pytest

from app.services.corpus_loader import CorpusLoader, CorpusLoadError, CorpusValidationError


def test_corpus_loader_from_dict():
    """Verify loading from a valid dictionary."""
    data = {
        "version": "1.0",
        "documents": [
            {
                "id": "doc_01",
                "title": "Title 1",
                "text": "Text 1",
                "source": "Source 1",
            }
        ]
    }
    loader = CorpusLoader.from_dict(data)
    assert len(loader) == 1
    assert loader.version == "1.0"
    doc = loader.get_document_by_id("doc_01")
    assert doc is not None
    assert doc.title == "Title 1"


def test_corpus_loader_from_json():
    """Verify loading from a valid JSON string."""
    json_str = json.dumps({
        "version": "1.0",
        "documents": [
            {
                "id": "doc_json_01",
                "title": "JSON Title",
                "text": "JSON Text content",
                "source": "JSON Source",
            }
        ]
    })
    loader = CorpusLoader.from_json(json_str)
    assert len(loader) == 1
    assert loader.get_document_by_id("doc_json_01") is not None


def test_corpus_loader_from_file_existing():
    """Verify loading the repository reference corpus file."""
    loader = CorpusLoader.from_file("data/corpus.json")
    assert len(loader) >= 15
    assert loader.version == "1.0"
    geo_doc = loader.get_document_by_id("doc_geo_001")
    assert geo_doc is not None
    assert "Paris" in geo_doc.text


def test_corpus_loader_nonexistent_file():
    """Verify error on non-existent file."""
    with pytest.raises(CorpusLoadError) as excinfo:
        CorpusLoader.from_file("data/non_existent_file.json")
    assert "not found" in str(excinfo.value)


def test_corpus_loader_invalid_json_string():
    """Verify error on malformed JSON string."""
    with pytest.raises(CorpusLoadError) as excinfo:
        CorpusLoader.from_json("invalid json { [")
    assert "Failed to parse JSON" in str(excinfo.value)


def test_corpus_loader_invalid_schema():
    """Verify validation error on schema mismatch."""
    with pytest.raises(CorpusValidationError) as excinfo:
        CorpusLoader.from_dict({"version": "1.0", "documents": [{"id": "d1"}]})
    assert "Corpus schema validation failed" in str(excinfo.value)
