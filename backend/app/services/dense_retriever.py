"""Dense semantic evidence retriever using SentenceTransformers and FAISS."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult


class DenseRetriever:
    """Retrieve evidence passages using dense vector embeddings and FAISS index."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        model_name: str = settings.EMBEDDING_MODEL,
        model: Any | None = None,
        index: Any | None = None,
        doc_id_map: list[str] | None = None,
    ) -> None:
        """Initialize Dense Retriever with documents and embedding index."""
        self._documents = documents
        self._doc_map: dict[str, CorpusDocument] = {doc.id: doc for doc in documents}
        self._model_name = model_name

        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)

        if index is not None and doc_id_map is not None:
            self._index = index
            self._doc_id_map = doc_id_map
        elif documents:
            self._index, self._doc_id_map = self._build_index(documents)
        else:
            self._index = None
            self._doc_id_map = []

    @property
    def documents(self) -> list[CorpusDocument]:
        """Return the indexed documents."""
        return self._documents

    @property
    def model_name(self) -> str:
        """Return the SentenceTransformer model name."""
        return self._model_name

    def __len__(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)

    def _build_index(
        self, documents: list[CorpusDocument]
    ) -> tuple[Any, list[str]]:
        """Compute normalized dense embeddings and index them in FAISS IndexFlatIP."""
        import faiss
        import numpy as np

        texts = [f"{doc.title} {doc.text}" for doc in documents]
        doc_id_map = [doc.id for doc in documents]

        # Generate normalized embeddings for cosine similarity with IndexFlatIP
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index, doc_id_map

    @classmethod
    def from_documents(
        cls,
        documents: list[CorpusDocument],
        model_name: str = settings.EMBEDDING_MODEL,
        model: Any | None = None,
    ) -> "DenseRetriever":
        """Build a Dense retriever instance from validated corpus documents."""
        return cls(documents=documents, model_name=model_name, model=model)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k semantically closest documents to the query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must contain at least one non-whitespace character.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not self._documents or self._index is None:
            return []

        import numpy as np

        effective_top_k = min(top_k, len(self._documents))

        # Encode normalized query vector
        query_vec = self._model.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, effective_top_k)

        results: list[RetrievalResult] = []
        for rank_idx, (doc_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if doc_idx < 0 or doc_idx >= len(self._doc_id_map):
                continue
            doc_id = self._doc_id_map[doc_idx]
            doc = self._doc_map[doc_id]

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
                    retrieval_method="DENSE",
                    rank=rank_idx,
                )
            )

        return results

    def save(self, dir_path: Path | str) -> tuple[Path, Path]:
        """Persist FAISS index binary and document mapping metadata to disk."""
        import faiss

        target_dir = Path(dir_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "faiss_index.bin"
        metadata_file = target_dir / "metadata.json"

        if self._index is not None:
            faiss.write_index(self._index, str(index_file))

        metadata = {
            "model_name": self._model_name,
            "doc_id_map": self._doc_id_map,
            "documents": [doc.model_dump(mode="json") for doc in self._documents],
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return index_file, metadata_file

    @classmethod
    def load(
        cls,
        dir_path: Path | str,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> "DenseRetriever":
        """Load a persisted FAISS index and document metadata from disk."""
        import faiss

        target_dir = Path(dir_path)
        index_file = target_dir / "faiss_index.bin"
        metadata_file = target_dir / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {index_file.resolve()}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_file.resolve()}")

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        resolved_model_name = model_name or metadata.get("model_name", settings.EMBEDDING_MODEL)

        documents = [CorpusDocument.model_validate(doc) for doc in metadata["documents"]]
        doc_id_map = metadata["doc_id_map"]

        index = faiss.read_index(str(index_file))

        return cls(
            documents=documents,
            model_name=resolved_model_name,
            model=model,
            index=index,
            doc_id_map=doc_id_map,
        )
