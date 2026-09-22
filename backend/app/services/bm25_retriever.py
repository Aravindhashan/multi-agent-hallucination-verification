"""Lexical evidence retriever using rank-bm25."""

import json
from pathlib import Path
import re
from rank_bm25 import BM25Okapi

from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult

_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Extract deterministic lowercase word/number tokens from text."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Retrieve evidence passages using lexical BM25 matching."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        tokenized_corpus: list[list[str]] | None = None,
    ) -> None:
        """Initialize BM25 index over the provided corpus documents."""
        self._documents = documents
        self._doc_map: dict[str, CorpusDocument] = {doc.id: doc for doc in documents}

        if tokenized_corpus is not None:
            self._tokenized_corpus = tokenized_corpus
        else:
            self._tokenized_corpus = [
                tokenize(f"{doc.title} {doc.text}") for doc in documents
            ]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    @property
    def documents(self) -> list[CorpusDocument]:
        """Return the indexed documents."""
        return self._documents

    def __len__(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)

    @classmethod
    def from_documents(cls, documents: list[CorpusDocument]) -> "BM25Retriever":
        """Build a BM25 retriever instance from validated corpus documents."""
        return cls(documents=documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant evidence documents for the query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must contain at least one non-whitespace character.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not self._documents or self._bm25 is None:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self._bm25.get_scores(tokenized_query)
        # Pair index with score and sort descending
        scored_indices = sorted(
            enumerate(doc_scores),
            key=lambda item: item[1],
            reverse=True,
        )

        effective_top_k = min(top_k, len(scored_indices))
        results: list[RetrievalResult] = []

        for rank_idx, (doc_idx, score) in enumerate(scored_indices[:effective_top_k], start=1):
            doc = self._documents[doc_idx]
            results.append(
                RetrievalResult(
                    document_id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    source_url=doc.source_url,
                    domain=doc.domain,
                    published_date=doc.published_date,
                    score=float(score),
                    retrieval_method="BM25",
                    rank=rank_idx,
                )
            )

        return results

    def save(self, dir_path: Path | str) -> Path:
        """Persist BM25 tokenized corpus and document metadata to disk."""
        target_dir = Path(dir_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "documents": [doc.model_dump(mode="json") for doc in self._documents],
            "tokenized_corpus": self._tokenized_corpus,
        }

        save_file = target_dir / "bm25_index.json"
        save_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return save_file

    @classmethod
    def load(cls, dir_path: Path | str) -> "BM25Retriever":
        """Load a persisted BM25 retriever index from disk."""
        target_dir = Path(dir_path)
        save_file = target_dir / "bm25_index.json"

        if not save_file.exists():
            raise FileNotFoundError(f"BM25 index file not found at: {save_file.resolve()}")

        data = json.loads(save_file.read_text(encoding="utf-8"))
        documents = [CorpusDocument.model_validate(doc) for doc in data["documents"]]
        tokenized_corpus = data["tokenized_corpus"]

        return cls(documents=documents, tokenized_corpus=tokenized_corpus)
