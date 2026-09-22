"""Hybrid evidence retriever combining BM25 lexical and Dense FAISS semantic search."""

from pathlib import Path
import threading
from uuid import uuid4

from app.config import settings
from app.schemas import EvidenceItem
from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_loader import CorpusLoader
from app.services.corpus_schema import CorpusDocument
from app.services.dense_retriever import DenseRetriever
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.retrieval_models import HybridCandidate, RetrievalResult


class HybridRetriever(BaseEvidenceRetriever):
    """Combine BM25 lexical search and Dense FAISS semantic search into a merged candidate set."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ) -> None:
        """Initialize Hybrid Retriever with BM25 and Dense sub-retrievers."""
        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Return the underlying BM25 retriever."""
        return self._bm25_retriever

    @property
    def dense_retriever(self) -> DenseRetriever:
        """Return the underlying Dense retriever."""
        return self._dense_retriever

    @classmethod
    def from_documents(
        cls,
        documents: list[CorpusDocument],
        model_name: str = settings.EMBEDDING_MODEL,
    ) -> "HybridRetriever":
        """Build hybrid retriever directly from a list of validated corpus documents."""
        bm25 = BM25Retriever.from_documents(documents)
        dense = DenseRetriever.from_documents(documents, model_name=model_name)
        return cls(bm25_retriever=bm25, dense_retriever=dense)

    @classmethod
    def from_corpus_file(
        cls,
        corpus_path: Path | str = settings.CORPUS_PATH,
        model_name: str = settings.EMBEDDING_MODEL,
    ) -> "HybridRetriever":
        """Load corpus from file and build both BM25 and Dense indexes."""
        loader = CorpusLoader.from_file(corpus_path)
        return cls.from_documents(loader.documents, model_name=model_name)

    def retrieve_bm25(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Perform lexical-only BM25 retrieval."""
        return self._bm25_retriever.retrieve(query=query, top_k=top_k)

    def retrieve_dense(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Perform dense semantic-only FAISS retrieval."""
        return self._dense_retriever.retrieve(query=query, top_k=top_k)

    def retrieve_hybrid(
        self, query: str, top_k: int = settings.DEFAULT_TOP_K
    ) -> list[HybridCandidate]:
        """Retrieve and merge candidate sets from both BM25 and Dense retrieval.

        Candidate aggregation merges matches, deduplicates by document ID, and
        preserves individual scores and ranks without applying premature RRF or
        cross-encoder reranking.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must contain at least one non-whitespace character.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        bm25_results = self.retrieve_bm25(query=query, top_k=top_k)
        dense_results = self.retrieve_dense(query=query, top_k=top_k)

        candidate_map: dict[str, dict] = {}

        # Process BM25 results
        for res in bm25_results:
            candidate_map[res.document_id] = {
                "document_id": res.document_id,
                "title": res.title,
                "text": res.text,
                "source": res.source,
                "source_url": res.source_url,
                "domain": res.domain,
                "published_date": res.published_date,
                "bm25_score": res.score,
                "bm25_rank": res.rank,
                "dense_score": None,
                "dense_rank": None,
                "matched_methods": ["BM25"],
            }

        # Process Dense results and merge
        for res in dense_results:
            if res.document_id in candidate_map:
                candidate_map[res.document_id]["dense_score"] = res.score
                candidate_map[res.document_id]["dense_rank"] = res.rank
                if "DENSE" not in candidate_map[res.document_id]["matched_methods"]:
                    candidate_map[res.document_id]["matched_methods"].append("DENSE")
            else:
                candidate_map[res.document_id] = {
                    "document_id": res.document_id,
                    "title": res.title,
                    "text": res.text,
                    "source": res.source,
                    "source_url": res.source_url,
                    "domain": res.domain,
                    "published_date": res.published_date,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "dense_score": res.score,
                    "dense_rank": res.rank,
                    "matched_methods": ["DENSE"],
                }

        # Stable ordering for candidate set:
        # 1. Documents found by both methods (sorted by best rank)
        # 2. Documents found by a single method (sorted by their individual rank)
        def candidate_sort_key(item: dict) -> tuple[int, int]:
            matched_count = len(item["matched_methods"])
            best_rank = min(
                r for r in [item["bm25_rank"], item["dense_rank"]] if r is not None
            )
            # Prioritize higher match count (2 before 1), then best individual rank
            return (-matched_count, best_rank)

        sorted_candidates = sorted(candidate_map.values(), key=candidate_sort_key)
        effective_top_k = min(top_k, len(sorted_candidates))

        results: list[HybridCandidate] = []
        for rank_idx, cand in enumerate(sorted_candidates[:effective_top_k], start=1):
            results.append(
                HybridCandidate(
                    document_id=cand["document_id"],
                    title=cand["title"],
                    text=cand["text"],
                    source=cand["source"],
                    source_url=cand["source_url"],
                    domain=cand["domain"],
                    published_date=cand["published_date"],
                    bm25_score=cand["bm25_score"],
                    bm25_rank=cand["bm25_rank"],
                    dense_score=cand["dense_score"],
                    dense_rank=cand["dense_rank"],
                    matched_methods=cand["matched_methods"],
                    combined_rank=rank_idx,
                )
            )

        return results

    def retrieve(self, claim_text: str, stance: str = "supporting") -> list[EvidenceItem]:
        """Implement BaseEvidenceRetriever interface for seamless backward compatibility.

        Converts hybrid candidates into EvidenceItem schemas with deterministic relevance scores.
        Adapts query based on stance (supporting vs. contradicting).
        """
        query_text = claim_text.strip()
        if str(stance).lower() == "contradicting":
            query_text = f"{query_text} controversy criticism contradiction disputed false refuted"

        candidates = self.retrieve_hybrid(query=query_text, top_k=settings.DEFAULT_TOP_K)
        evidence_items: list[EvidenceItem] = []

        for cand in candidates:
            # Construct a normalized relevance score from available signals
            if cand.dense_score is not None:
                # Dense cosine similarity with FlatIP is in [-1.0, 1.0], bounded to [0.0, 1.0]
                norm_score = max(0.0, min(1.0, (cand.dense_score + 1.0) / 2.0))
            elif cand.bm25_score is not None:
                # Bounded heuristic normalization for BM25 score
                norm_score = max(0.0, min(1.0, cand.bm25_score / (cand.bm25_score + 10.0)))
            else:
                norm_score = 0.50

            evidence_items.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    text=cand.text,
                    source_url=cand.source_url,
                    domain=cand.domain,
                    source_title=cand.title,
                    retrieval_method=f"hybrid:{'+'.join(cand.matched_methods)}",
                    relevance_score=round(norm_score, 4),
                )
            )

        return evidence_items

    def save(self, dir_path: Path | str = settings.INDEX_DIR) -> tuple[Path, tuple[Path, Path]]:
        """Save both BM25 and Dense FAISS indexes into the target index directory."""
        target_dir = Path(dir_path)
        bm25_dir = target_dir / "bm25"
        faiss_dir = target_dir / "faiss"

        bm25_saved = self._bm25_retriever.save(bm25_dir)
        dense_saved = self._dense_retriever.save(faiss_dir)
        return bm25_saved, dense_saved

    @classmethod
    def load(
        cls,
        dir_path: Path | str = settings.INDEX_DIR,
        model_name: str = settings.EMBEDDING_MODEL,
    ) -> "HybridRetriever":
        """Load both BM25 and Dense FAISS indexes from persisted index directory."""
        target_dir = Path(dir_path)
        bm25_dir = target_dir / "bm25"
        faiss_dir = target_dir / "faiss"

        bm25 = BM25Retriever.load(bm25_dir)
        dense = DenseRetriever.load(faiss_dir, model_name=model_name)
        return cls(bm25_retriever=bm25, dense_retriever=dense)


# ---------------------------------------------------------------------------
# Thread-safe Singleton Accessor
# ---------------------------------------------------------------------------

_hybrid_retriever_instance: HybridRetriever | None = None
_hybrid_retriever_lock = threading.Lock()


def get_hybrid_retriever(
    corpus_path: Path | str = settings.CORPUS_PATH,
    index_dir: Path | str = settings.INDEX_DIR,
    model_name: str = settings.EMBEDDING_MODEL,
    force_reload: bool = False,
) -> HybridRetriever:
    """Return a thread-safe singleton instance of HybridRetriever.

    If persisted indexes exist in index_dir, loads them; otherwise builds
    from corpus_path.
    """
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is not None and not force_reload:
        return _hybrid_retriever_instance

    with _hybrid_retriever_lock:
        if _hybrid_retriever_instance is not None and not force_reload:
            return _hybrid_retriever_instance

        idx_path = Path(index_dir)
        bm25_path = idx_path / "bm25"
        faiss_path = idx_path / "faiss"

        if bm25_path.exists() and faiss_path.exists():
            _hybrid_retriever_instance = HybridRetriever.load(idx_path, model_name=model_name)
        else:
            c_path = Path(corpus_path)
            if c_path.exists():
                _hybrid_retriever_instance = HybridRetriever.from_corpus_file(c_path, model_name=model_name)
            else:
                raise FileNotFoundError(
                    f"Neither index directory ({index_dir}) nor corpus file ({corpus_path}) found."
                )

        return _hybrid_retriever_instance
