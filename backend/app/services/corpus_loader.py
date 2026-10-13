"""Corpus loader service for loading and validating reference evidence corpora."""

import json
from pathlib import Path
from pydantic import ValidationError

from app.services.corpus_schema import Corpus, CorpusDocument


class CorpusLoadError(Exception):
    """Raised when the reference corpus cannot be found, read, or parsed."""
    pass


class CorpusValidationError(Exception):
    """Raised when the reference corpus fails schema or constraint validation."""
    pass


class CorpusLoader:
    """Load, validate, and manage reference evidence corpus documents."""

    def __init__(self, corpus: Corpus) -> None:
        """Initialize with a pre-validated Corpus object."""
        self._corpus = corpus
        self._doc_map: dict[str, CorpusDocument] = {doc.id: doc for doc in corpus.documents}

    @property
    def corpus(self) -> Corpus:
        """Return the underlying Corpus model."""
        return self._corpus

    @property
    def version(self) -> str:
        """Return the corpus version string."""
        return self._corpus.version

    @property
    def documents(self) -> list[CorpusDocument]:
        """Return the list of validated documents in the corpus."""
        return self._corpus.documents

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self._corpus.documents)

    def get_document_by_id(self, doc_id: str) -> CorpusDocument | None:
        """Retrieve a document by its unique ID."""
        return self._doc_map.get(doc_id)

    @classmethod
    def from_dict(cls, data: object) -> "CorpusLoader":
        """Validate and load a corpus from a dictionary or raw JSON-compatible object."""
        if not isinstance(data, dict):
            raise CorpusValidationError("Top-level corpus data must be a JSON object (dict).")
        try:
            validated_corpus = Corpus.model_validate(data)
            return cls(validated_corpus)
        except ValidationError as e:
            raise CorpusValidationError(f"Corpus schema validation failed: {e}") from e

    @classmethod
    def from_json(cls, json_str: str) -> "CorpusLoader":
        """Parse and validate a corpus from a JSON string."""
        if not isinstance(json_str, str) or not json_str.strip():
            raise CorpusValidationError("JSON string must be non-empty.")
        try:
            raw_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise CorpusLoadError(f"Failed to parse JSON string: {e}") from e
        return cls.from_dict(raw_data)

    @classmethod
    def from_jsonl(cls, jsonl_str: str, version: str = "1.0.0") -> "CorpusLoader":
        """Parse and validate a corpus from a JSON Lines (JSONL) string."""
        if not isinstance(jsonl_str, str) or not jsonl_str.strip():
            raise CorpusValidationError("JSONL string must be non-empty.")
        docs: list[CorpusDocument] = []
        for line_num, line in enumerate(jsonl_str.strip().split("\n"), start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                doc_data = json.loads(line_str)
                docs.append(CorpusDocument.model_validate(doc_data))
            except Exception as e:
                raise CorpusLoadError(f"Failed to parse JSONL line {line_num}: {e}") from e
        corpus = Corpus(version=version, documents=docs)
        return cls(corpus)

    @classmethod
    def from_file(cls, file_path: Path | str) -> "CorpusLoader":
        """Load and validate a corpus from a local JSON or JSONL file path."""
        path = Path(file_path)
        if not path.exists():
            raise CorpusLoadError(f"Corpus file not found at: {path.resolve()}")
        if not path.is_file():
            raise CorpusLoadError(f"Corpus path is not a regular file: {path.resolve()}")
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise CorpusLoadError(f"Failed to read corpus file {path.resolve()}: {e}") from e

        if path.suffix.lower() == ".jsonl":
            return cls.from_jsonl(raw_text, version=f"{path.stem}-1.0.0")

        return cls.from_json(raw_text)

