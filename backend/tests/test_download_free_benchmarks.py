"""Unit tests for the free benchmark downloader script."""

import io
import json
from pathlib import Path
import tarfile
from unittest.mock import MagicMock, patch
import pytest

from app.services.corpus_schema import CorpusDocument
from scripts.download_free_benchmarks import (
    download_fever,
    download_scifact,
    normalize_scifact_document,
)


def test_normalize_scifact_document():
    """Verify raw SciFact dictionary conversion into a validated CorpusDocument."""
    raw = {
        "doc_id": 12345,
        "title": "Aspirin reduces cardiovascular mortality",
        "abstract": [
            "Cardiovascular disease is the leading cause of death worldwide.",
            "Randomized controlled trials demonstrate that low-dose aspirin reduces vascular events.",
        ],
    }

    doc = normalize_scifact_document(raw)
    assert isinstance(doc, CorpusDocument)
    assert doc.id == "scifact_12345"
    assert doc.title == "Aspirin reduces cardiovascular mortality"
    assert "Randomized controlled trials" in doc.text
    assert doc.domain == "ncbi.nlm.nih.gov"
    assert doc.source == "PubMed / AllenAI SciFact"
    assert doc.source_url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_download_scifact_mocked(tmp_path: Path):
    """Verify downloading and extracting SciFact files using a mocked tarball."""
    # Create an in-memory tarball mimicking SciFact release
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        # Add corpus.jsonl
        corpus_data = json.dumps(
            {
                "doc_id": 101,
                "title": "CRISPR gene editing",
                "abstract": ["CRISPR Cas9 enables targeted DNA modifications."],
            }
        ).encode("utf-8") + b"\n"
        info = tarfile.TarInfo("data/corpus.jsonl")
        info.size = len(corpus_data)
        tar.addfile(info, io.BytesIO(corpus_data))

        # Add claims_dev.jsonl
        claims_data = json.dumps(
            {
                "id": 1,
                "claim": "CRISPR Cas9 can modify human genomic DNA.",
                "evidence": {"101": [{"sentences": [0], "label": "SUPPORT"}]},
            }
        ).encode("utf-8") + b"\n"
        info_c = tarfile.TarInfo("data/claims_dev.jsonl")
        info_c.size = len(claims_data)
        tar.addfile(info_c, io.BytesIO(claims_data))

    tar_bytes.seek(0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = tar_bytes.getvalue()
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.get", return_value=mock_resp):
        docs, claims = download_scifact(target_dir=tmp_path, skip_if_exists=False)

    assert docs == 1
    assert claims == 1

    # Verify corpus file output
    corpus_file = tmp_path / "scifact_corpus.jsonl"
    assert corpus_file.exists()
    line = corpus_file.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["id"] == "scifact_101"
    assert "CRISPR gene editing" in data["title"]

    # Verify claims file output
    claims_file = tmp_path / "scifact_claims.jsonl"
    assert claims_file.exists()
    c_line = claims_file.read_text(encoding="utf-8").strip()
    c_data = json.loads(c_line)
    assert c_data["claim_id"] == "scifact_1"
    assert c_data["evidence_doc_ids"] == ["scifact_101"]


def test_download_fever_mocked(tmp_path: Path):
    """Verify streaming FEVER dev split and building reference Wikipedia corpus."""
    raw_lines = [
        json.dumps(
            {
                "id": 999,
                "claim": "Alan Turing was an English mathematician.",
                "label": "SUPPORTS",
                "evidence": [[[1, 2, "Alan_Turing", 0]]],
            }
        ),
        json.dumps(
            {
                "id": 1000,
                "claim": "Ada Lovelace wrote the first computer algorithm.",
                "label": "SUPPORTS",
                "evidence": [[[3, 4, "Ada_Lovelace", 1]]],
            }
        ),
    ]

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(raw_lines)
    mock_resp.raise_for_status.return_value = None

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = mock_resp
    stream_ctx.__exit__.return_value = None

    with patch("httpx.Client.stream", return_value=stream_ctx):
        docs, claims = download_fever(target_dir=tmp_path, max_claims=10)

    assert claims == 2
    assert docs == 2

    # Check claims file
    claims_file = tmp_path / "fever_claims.jsonl"
    assert claims_file.exists()
    c_lines = [json.loads(l) for l in claims_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(c_lines) == 2
    assert c_lines[0]["claim_id"] == "fever_999"
    assert "Alan_Turing" in c_lines[0]["cited_wikipedia_titles"]

    # Check corpus file
    corpus_file = tmp_path / "fever_corpus.jsonl"
    assert corpus_file.exists()
    doc_lines = [json.loads(l) for l in corpus_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(doc_lines) == 2
    titles = [d["title"] for d in doc_lines]
    assert "Alan Turing" in titles
    assert "Ada Lovelace" in titles
    assert all(d["domain"] == "en.wikipedia.org" for d in doc_lines)


def test_skip_if_exists(tmp_path: Path):
    """Verify that download functions skip network operations when target files exist."""
    corpus_file = tmp_path / "scifact_corpus.jsonl"
    claims_file = tmp_path / "scifact_claims.jsonl"

    corpus_file.write_text('{"id": "doc1"}\n', encoding="utf-8")
    claims_file.write_text('{"claim_id": "c1"}\n', encoding="utf-8")

    # With skip_if_exists=True, httpx.Client.get should never be called
    with patch("httpx.Client.get") as mock_get:
        docs, claims = download_scifact(target_dir=tmp_path, skip_if_exists=True)
        mock_get.assert_not_called()
        assert docs == 1
        assert claims == 1
