"""Unified Federated Evidence Retriever routing across Live Web, SciFact, and FEVER."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

from app.config import settings
from app.schemas import EvidenceItem
from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_loader import CorpusLoader
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.free_web_retriever import FreeWebRetriever
from app.services.mock_evidence_retriever import MockEvidenceRetriever

logger = logging.getLogger("hallucination_verifier.federated_retriever")

# Biomedical domain heuristics for scientific routing
_BIOMEDICAL_KEYWORDS: frozenset[str] = frozenset(
    [
        "gene", "genes", "cancer", "cells", "cell", "protein", "proteins",
        "treatment", "therapy", "mortality", "virus", "viral", "syndrome",
        "disease", "diseases", "patients", "patient", "clinical", "rna",
        "dna", "mutation", "mutations", "drug", "drugs", "tumor", "tumors",
        "receptor", "receptors", "inhibitor", "antibiotic", "infection",
        "immune", "antibody", "antibodies", "cardiovascular", "myeloid",
        "vascular", "dosage", "aspirin", "statin", "in vivo", "in vitro",
        "pathogenesis", "hematopoiesis", "pathogenetic", "myelodysplastic",
    ]
)
_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_GENERIC_SCIENTIFIC_TOKENS: frozenset[str] = frozenset([
    "clinical", "patient", "patients", "study", "studies", "trial", "trials",
    "human", "humans", "effect", "effects", "risk", "risks", "rate", "rates",
    "level", "levels", "treatment", "therapy", "reduce", "reduces", "reduced",
    "lower", "lowers", "lowering", "increase", "increases", "increased",
    "associated", "significant", "significantly", "result", "results",
    "evidence", "group", "groups", "outcome", "outcomes", "disease", "diseases",
])
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
        "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
        "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
        "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
        "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
        "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
        "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
        "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
        "yourselves", "plan", "dates", "numbers", "locations", "verify",
    ]
)


class FederatedEvidenceRetriever(BaseEvidenceRetriever):
    """Unified evidence engine federating Live Web search and Curated Benchmarks.

    Supported Modes (`settings.EVIDENCE_MODE`):
    - 'live_web': DuckDuckGo Free HTML + Wikimedia REST API with SQLite caching.
    - 'benchmark_scifact': 5,183 peer-reviewed PubMed abstracts (AllenAI).
    - 'benchmark_fever': FEVER Wikipedia validation benchmark articles.
    - 'hybrid': Domain-aware dual-track routing. Queries local curated benchmarks
      first; if candidate relevance is below escalation threshold, seamlessly
      escalates to live web retrieval.
    - 'toy': Fast offline 15-document mock retrieval for unit tests.
    """

    def __init__(
        self,
        mode: str | None = None,
        web_retriever: FreeWebRetriever | None = None,
        scifact_retriever: BM25Retriever | None = None,
        fever_retriever: BM25Retriever | None = None,
        escalation_threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """Initialize federated retriever with optional overrides."""
        self.mode = (
            str(mode).strip().lower()
            if mode is not None
            else str(settings.EVIDENCE_MODE).strip().lower()
        )
        self.escalation_threshold = (
            float(escalation_threshold)
            if escalation_threshold is not None
            else float(settings.FEDERATED_ESCALATION_THRESHOLD)
        )
        self.top_k = int(top_k) if top_k is not None else int(settings.DEFAULT_TOP_K)

        self._web_retriever = web_retriever
        self._scifact_retriever = scifact_retriever
        self._fever_retriever = fever_retriever
        self._mock_retriever = MockEvidenceRetriever()

        self._scifact_lock = threading.Lock()
        self._fever_lock = threading.Lock()
        self._web_lock = threading.Lock()

    def _get_web_retriever(self) -> FreeWebRetriever:
        """Return initialized or shared FreeWebRetriever instance."""
        if self._web_retriever is None:
            with self._web_lock:
                if self._web_retriever is None:
                    self._web_retriever = FreeWebRetriever(max_results=self.top_k)
        return self._web_retriever

    def _get_scifact_retriever(self) -> BM25Retriever | None:
        """Lazy load and index SciFact PubMed corpus on first use."""
        if self._scifact_retriever is None:
            with self._scifact_lock:
                if self._scifact_retriever is None:
                    corpus_path = Path(settings.SCIFACT_CORPUS_PATH)
                    if not corpus_path.exists():
                        logger.warning("SciFact corpus file not found at %s", corpus_path)
                        return None
                    try:
                        loader = CorpusLoader.from_file(corpus_path)
                        self._scifact_retriever = BM25Retriever.from_documents(loader.documents)
                        logger.info("Indexed %d SciFact documents for federated retrieval.", len(loader.documents))
                    except Exception as exc:
                        logger.error("Failed to index SciFact corpus: %s", exc)
                        return None
        return self._scifact_retriever

    def _get_fever_retriever(self) -> BM25Retriever | None:
        """Lazy load and index FEVER Wikipedia corpus on first use."""
        if self._fever_retriever is None:
            with self._fever_lock:
                if self._fever_retriever is None:
                    corpus_path = Path(settings.FEVER_CORPUS_PATH)
                    if not corpus_path.exists():
                        logger.warning("FEVER corpus file not found at %s", corpus_path)
                        return None
                    try:
                        loader = CorpusLoader.from_file(corpus_path)
                        self._fever_retriever = BM25Retriever.from_documents(loader.documents)
                        logger.info("Indexed %d FEVER documents for federated retrieval.", len(loader.documents))
                    except Exception as exc:
                        logger.error("Failed to index FEVER corpus: %s", exc)
                        return None
        return self._fever_retriever

    @staticmethod
    def _is_biomedical_query(query: str) -> bool:
        """Detect whether query matches biomedical domain terminology."""
        tokens = set(_TOKEN_PATTERN.findall(query.lower()))
        return bool(tokens.intersection(_BIOMEDICAL_KEYWORDS))

    def _retrieve_from_bm25(
        self,
        retriever: BM25Retriever,
        query: str,
        corpus_tag: str,
        top_k: int,
    ) -> list[EvidenceItem]:
        """Convert raw BM25 search matches into standardized EvidenceItems."""
        raw_results = retriever.retrieve(query=query, top_k=top_k)
        items: list[EvidenceItem] = []
        query_tokens = set(_TOKEN_PATTERN.findall(query.lower()))
        meaningful_query_tokens = {
            t for t in query_tokens
            if t not in _ENGLISH_STOPWORDS and len(t) >= 2
        }

        query_words = _TOKEN_PATTERN.findall(query)
        entity_tokens = {
            w.lower() for w in query_words
            if any(c.isupper() for c in w) or any(c.isdigit() for c in w)
        }

        for rank_idx, r in enumerate(raw_results, start=1):
            doc_tokens = set(_TOKEN_PATTERN.findall(f"{r.title} {r.text}".lower()))
            overlap = len(query_tokens.intersection(doc_tokens))
            if r.score <= 0.0 and overlap == 0:
                continue

            content_overlap = (
                len(meaningful_query_tokens.intersection(doc_tokens)) / max(1, len(meaningful_query_tokens))
                if meaningful_query_tokens
                else (overlap / max(1, len(query_tokens)))
            )

            # Monotonic saturation normalization or rank-based fallback for zero-IDF small corpora
            if r.score > 0.0:
                base_rel = r.score / (r.score + 10.0)
                relevance = round(max(0.20, min(1.0, 0.5 * base_rel + 0.5 * content_overlap)), 3)
            else:
                if content_overlap < 0.25:
                    continue
                relevance = round(min(0.90, 0.40 + 0.50 * content_overlap), 3)

            # If query specifies subject entities/identifiers, penalize documents omitting them
            if entity_tokens and not any(e in doc_tokens for e in entity_tokens):
                relevance = round(relevance * 0.30, 3)
                if relevance < 0.35:
                    continue

            items.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    text=f"{r.title}: {r.text}" if not r.text.startswith(r.title) else r.text,
                    source_url=r.source_url,
                    domain=r.domain,
                    source_title=r.title,
                    retrieval_method=f"benchmark_{corpus_tag}",
                    relevance_score=relevance,
                )
            )

        return items

    def retrieve(self, claim_text: str, stance: str = "supporting") -> list[EvidenceItem]:
        """Retrieve stance-aware evidence according to the configured federation mode."""
        normalized_claim = FreeWebRetriever.strip_meta_instructions(claim_text.strip())
        if not normalized_claim:
            return []

        normalized_stance = stance.strip().lower()
        if normalized_stance not in ("supporting", "contradicting"):
            raise ValueError("Stance must be 'supporting' or 'contradicting'.")

        mode = self.mode

        # 1. Toy / Mock Mode
        if mode == "toy":
            return self._mock_retriever.retrieve(normalized_claim, stance=normalized_stance)

        # 2. Live Web Exclusive Mode
        if mode == "live_web":
            web = self._get_web_retriever()
            return web.retrieve(normalized_claim, stance=normalized_stance)

        # 3. SciFact Exclusive Benchmark Mode
        if mode == "benchmark_scifact":
            scifact = self._get_scifact_retriever()
            if scifact is None:
                logger.warning("SciFact retriever unavailable; falling back to web.")
                return self._get_web_retriever().retrieve(normalized_claim, stance=normalized_stance)
            query = (
                f"{normalized_claim} dispute controversy"
                if normalized_stance == "contradicting"
                else normalized_claim
            )
            return self._retrieve_from_bm25(scifact, query, "scifact", self.top_k)

        # 4. FEVER Exclusive Benchmark Mode
        if mode == "benchmark_fever":
            fever = self._get_fever_retriever()
            if fever is None:
                logger.warning("FEVER retriever unavailable; falling back to web.")
                return self._get_web_retriever().retrieve(normalized_claim, stance=normalized_stance)
            query = (
                f"{normalized_claim} dispute controversy"
                if normalized_stance == "contradicting"
                else normalized_claim
            )
            return self._retrieve_from_bm25(fever, query, "fever", self.top_k)

        # 5. Hybrid Dual-Track Federation Mode
        # Step A: Select primary local benchmark based on domain heuristics
        is_bio = self._is_biomedical_query(normalized_claim)
        primary_bm25 = self._get_scifact_retriever() if is_bio else self._get_fever_retriever()
        corpus_tag = "scifact" if is_bio else "fever"

        candidates: list[EvidenceItem] = []
        if primary_bm25 is not None:
            query = (
                f"{normalized_claim} dispute"
                if normalized_stance == "contradicting"
                else normalized_claim
            )
            candidates = self._retrieve_from_bm25(primary_bm25, query, corpus_tag, self.top_k)

        # Step B: Escalation Evaluation
        # If the claim contains specific named entities, proper nouns, acronyms, or alphanumeric identifiers,
        # verify that candidate documents actually ground them before accepting benchmark.
        claim_words = _TOKEN_PATTERN.findall(normalized_claim)
        entity_tokens = {
            w.lower() for w in claim_words
            if any(c.isupper() for c in w) or any(c.isdigit() for c in w)
        }
        content_tokens = {
            w.lower() for w in claim_words
            if w.lower() not in _ENGLISH_STOPWORDS and len(w) >= 3
        }
        topic_tokens = {w for w in content_tokens if w not in _GENERIC_SCIENTIFIC_TOKENS} or content_tokens

        # Individual candidate grounding check:
        # Rather than checking overlap across the union of all candidates (which can yield
        # false positives across fragmented multi-topic papers), at least one single candidate document
        # must ground the key entities AND achieve comprehensive topic coverage and relevance.
        def _candidate_grounds_claim(cand: EvidenceItem) -> bool:
            cand_tokens = set(_TOKEN_PATTERN.findall(cand.text.lower()))
            has_entities = (not entity_tokens) or (
                all(e in cand_tokens for e in entity_tokens)
                if len(entity_tokens) <= 2
                else (len([e for e in entity_tokens if e in cand_tokens]) / len(entity_tokens) >= 0.70)
            )
            content_overlap = (
                len(content_tokens.intersection(cand_tokens)) / max(1, len(content_tokens))
                if content_tokens
                else 1.0
            )
            topic_overlap = (
                len(topic_tokens.intersection(cand_tokens)) / max(1, len(topic_tokens))
                if topic_tokens
                else 1.0
            )
            return bool(
                has_entities
                and topic_overlap >= 0.75
                and content_overlap >= 0.50
                and cand.relevance_score >= self.escalation_threshold
            )

        has_confident_evidence = bool(candidates and any(_candidate_grounds_claim(c) for c in candidates))

        if has_confident_evidence:
            logger.debug(
                "Hybrid mode accepted %d %s benchmark candidates for '%s'",
                len(candidates),
                corpus_tag,
                normalized_claim,
            )
            return candidates

        # Step C: Escalate to Live Web Ingestion
        logger.info(
            "Benchmark candidates below threshold (count=%d, top=%s); escalating to Live Web for '%s'",
            len(candidates),
            candidates[0].relevance_score if candidates else "none",
            normalized_claim,
        )
        web = self._get_web_retriever()
        web_items = web.retrieve(normalized_claim, stance=normalized_stance)

        # Combine or fallback
        if web_items:
            # If we also had partial high-quality benchmark items, merge them without duplication
            seen_texts = {w.text.lower() for w in web_items}
            merged = list(web_items)
            for c in candidates:
                if c.text.lower() not in seen_texts and c.relevance_score >= self.escalation_threshold:
                    merged.append(c)
            # Ensure top benchmark candidate is represented for domain provenance
            if candidates and not any(c.domain and "ncbi" in c.domain for c in merged[: self.top_k]):
                merged.insert(1, candidates[0])
            return merged[: max(self.top_k, min(len(merged), 4))]

        return candidates


# Singleton instance management
_global_federated_instance: FederatedEvidenceRetriever | None = None
_global_federated_lock = threading.Lock()


def get_federated_evidence_retriever(
    mode: str | None = None,
    force_reload: bool = False,
) -> FederatedEvidenceRetriever:
    """Return thread-safe singleton instance of FederatedEvidenceRetriever."""
    global _global_federated_instance
    normalized_mode = mode.strip().lower() if mode is not None else None

    if _global_federated_instance is not None and not force_reload:
        if normalized_mode is None or _global_federated_instance.mode == normalized_mode:
            return _global_federated_instance

    with _global_federated_lock:
        if _global_federated_instance is not None and not force_reload:
            if normalized_mode is None or _global_federated_instance.mode == normalized_mode:
                return _global_federated_instance
        _global_federated_instance = FederatedEvidenceRetriever(mode=mode)
        return _global_federated_instance
