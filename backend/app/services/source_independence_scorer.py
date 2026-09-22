"""Source Independence & Redundancy Detection Engine (Step 4.3).

Evaluates whether retrieved evidence items represent genuinely independent sources
or whether multiple evidence items are substantially redundant, duplicated, or syndicated.

Operates as a deterministic, offline, pure lexical analysis engine without network calls,
LLMs, or ML models.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.services.source_independence_config import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DEFAULT_WEAK_RELATION_THRESHOLD,
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DISCLAIMER_PHRASE,
    SourceIndependenceConfigError,
    SourceIndependenceInputError,
    SourceIndependenceSettings,
)
from app.services.source_independence_models import (
    IndependenceCluster,
    IndependencePairResult,
    IndependenceSetResult,
    RedundancyClass,
)

logger = logging.getLogger("hallucination_verifier.source_independence")

# Regular expression matching word tokens with Unicode letter/digit support
_WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
# Control characters pattern (ASCII control characters excluding whitespace)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Whitespace normalization pattern
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text using Unicode NFKC, control character stripping, and whitespace collapse.

    Preserves numbers, punctuation, dates, and factual content (§10, §11).
    Does NOT mutate the original raw text.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)
    # Strip non-whitespace control characters
    normalized = _CONTROL_CHAR_PATTERN.sub("", normalized)
    # Collapse multiple whitespace characters into a single space and trim
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def tokenize(text: str) -> list[str]:
    """Deterministic tokenization pipeline (§14).

    Pipeline:
    1. Unicode NFKC normalization
    2. Lowercase conversion
    3. Strip control characters
    4. Split on word boundaries via regex
    5. Drop empty tokens
    """
    if not text:
        return []
    normalized = normalize_text(text).lower()
    return [tok for tok in _WORD_TOKEN_PATTERN.findall(normalized) if tok]


def extract_token_set(text: str) -> frozenset[str]:
    """Extract a frozenset of unique tokens from text for lexical Jaccard similarity."""
    return frozenset(tokenize(text))


def compute_jaccard_similarity(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    """Compute token Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| (§13).

    Rules:
    - Identical non-empty sets -> 1.0
    - Disjoint sets -> 0.0
    - Both sets empty -> 0.0 (explicitly guarded to prevent division by zero)
    - One set empty, other non-empty -> 0.0
    """
    if not tokens_a and not tokens_b:
        return 0.0
    union_len = len(tokens_a | tokens_b)
    if union_len == 0:
        return 0.0
    intersection_len = len(tokens_a & tokens_b)
    return intersection_len / union_len


def compute_char_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity as an alternative lexical metric (§50)."""
    norm_a = normalize_text(text_a).lower()
    norm_b = normalize_text(text_b).lower()
    if len(norm_a) < n or len(norm_b) < n:
        return 1.0 if norm_a == norm_b and norm_a else 0.0
    ngrams_a = frozenset(norm_a[i : i + n] for i in range(len(norm_a) - n + 1))
    ngrams_b = frozenset(norm_b[i : i + n] for i in range(len(norm_b) - n + 1))
    union_len = len(ngrams_a | ngrams_b)
    if union_len == 0:
        return 0.0
    return len(ngrams_a & ngrams_b) / union_len


def normalize_url(url: str | None) -> str | None:
    """Normalize a source URL deterministically for comparison (§17).

    Strips fragments, normalizes scheme and hostname to lowercase, removes standard ports,
    and strips trailing slashes from path components.
    """
    if not url or not url.strip():
        return None
    try:
        parsed = urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Remove standard port specifications
        if ":" in netloc:
            host, _, port = netloc.partition(":")
            if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                netloc = host
        path = parsed.path.rstrip("/")
        # Preserve root slash if path was only "/"
        if not path and parsed.path == "/":
            path = "/"
        query = parsed.query
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return url.strip().lower()


def normalize_domain(domain: str | None) -> str | None:
    """Normalize a domain string deterministically (§16)."""
    if not domain or not domain.strip():
        return None
    cleaned = domain.strip().lower()
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    return cleaned.rstrip(".")


def normalize_source(source: str | None) -> str | None:
    """Normalize a publisher or source string deterministically (§18)."""
    if not source or not source.strip():
        return None
    return source.strip().lower()


class _PreprocessedItem:
    """Internal lightweight structure caching pre-extracted metadata and token sets."""

    __slots__ = (
        "doc_id",
        "raw_text",
        "norm_text",
        "source",
        "source_url",
        "domain",
        "norm_source",
        "norm_url",
        "norm_domain",
        "token_set",
        "token_count",
    )

    def __init__(
        self,
        doc_id: str,
        raw_text: str,
        norm_text: str,
        source: str | None,
        source_url: str | None,
        domain: str | None,
        norm_source: str | None,
        norm_url: str | None,
        norm_domain: str | None,
        token_set: frozenset[str],
        token_count: int,
    ) -> None:
        self.doc_id = doc_id
        self.raw_text = raw_text
        self.norm_text = norm_text
        self.source = source
        self.source_url = source_url
        self.domain = domain
        self.norm_source = norm_source
        self.norm_url = norm_url
        self.norm_domain = norm_domain
        self.token_set = token_set
        self.token_count = token_count


class SourceIndependenceDetector:
    """Deterministic lexical source independence and redundancy detector (§9, §22).

    Attributes:
        settings: Configuration options and threshold values.
    """

    def __init__(self, settings: SourceIndependenceSettings | None = None) -> None:
        """Initialize detector with settings."""
        self.settings = settings or SourceIndependenceSettings()

    def _extract_document_fields(
        self, item: Any
    ) -> tuple[str, str, str | None, str | None, str | None]:
        """Extract (document_id, text, source, source_url, domain) from diverse input models."""
        if item is None:
            raise SourceIndependenceInputError("Evidence item cannot be None.")

        doc_id: str | None = None
        text: str | None = None
        source: str | None = None
        source_url: str | None = None
        domain: str | None = None

        if isinstance(item, dict):
            doc_id = item.get("document_id") or item.get("id")
            text = item.get("text")
            source = item.get("source")
            source_url = item.get("source_url")
            domain = item.get("domain")
        else:
            # Check attribute access (CorpusDocument, RetrievalResult, or generic object)
            doc_id = getattr(item, "document_id", None) or getattr(item, "id", None)
            text = getattr(item, "text", None)
            source = getattr(item, "source", None)
            source_url = getattr(item, "source_url", None)
            domain = getattr(item, "domain", None)

        if doc_id is None or not isinstance(doc_id, str) or not doc_id.strip():
            raise SourceIndependenceInputError(
                f"Evidence item missing valid string document_id: {item!r}"
            )
        if text is None or not isinstance(text, str):
            raise SourceIndependenceInputError(
                f"Evidence item '{doc_id}' missing valid string text: {item!r}"
            )

        return (
            doc_id.strip(),
            text,
            str(source) if source is not None else None,
            str(source_url) if source_url is not None else None,
            str(domain) if domain is not None else None,
        )

    def _preprocess_item(self, item: Any) -> _PreprocessedItem:
        """Preprocess and tokenize an evidence document once for efficient batching."""
        doc_id, raw_text, source, source_url, domain = self._extract_document_fields(item)
        norm_text = normalize_text(raw_text)
        tokens = extract_token_set(raw_text)
        norm_url = normalize_url(source_url)
        norm_source = normalize_source(source)
        norm_domain = normalize_domain(domain)
        # If domain was missing from metadata, attempt extraction from normalized URL netloc
        if not norm_domain and norm_url:
            try:
                norm_domain = normalize_domain(urlsplit(norm_url).netloc)
            except Exception:
                norm_domain = None

        return _PreprocessedItem(
            doc_id=doc_id,
            raw_text=raw_text,
            norm_text=norm_text,
            source=source,
            source_url=source_url,
            domain=domain,
            norm_source=norm_source,
            norm_url=norm_url,
            norm_domain=norm_domain,
            token_set=tokens,
            token_count=len(tokens),
        )

    def compare(self, evidence_a: Any, evidence_b: Any) -> IndependencePairResult:
        """Compare two evidence items and evaluate pairwise independence and redundancy (§19, §22).

        Args:
            evidence_a: First evidence document (CorpusDocument, RetrievalResult, or dict).
            evidence_b: Second evidence document (CorpusDocument, RetrievalResult, or dict).

        Returns:
            IndependencePairResult containing similarity score, flags, category, and explanation.
        """
        item_a = self._preprocess_item(evidence_a)
        item_b = self._preprocess_item(evidence_b)
        return self._compare_preprocessed(item_a, item_b)

    def _compare_preprocessed(
        self, a: _PreprocessedItem, b: _PreprocessedItem
    ) -> IndependencePairResult:
        """Compare two preprocessed evidence items following the hierarchical decision tree (§22)."""
        # §27 Self-Comparison policy
        if a.doc_id == b.doc_id:
            same_dom = bool(a.norm_domain and b.norm_domain and a.norm_domain == b.norm_domain)
            same_src = bool(a.norm_source and b.norm_source and a.norm_source == b.norm_source)
            same_u = bool(a.norm_url and b.norm_url and a.norm_url == b.norm_url)
            sim = 1.0
            explanation = (
                f"Self-comparison of document '{a.doc_id}': an evidence item is identically redundant "
                f"with itself (similarity={sim:.4f}, independence=0.0000). {DISCLAIMER_PHRASE}"
            )
            return IndependencePairResult(
                document_id_a=a.doc_id,
                document_id_b=b.doc_id,
                text_similarity=sim,
                is_exact_duplicate=True,
                is_near_duplicate=True,
                same_domain=same_dom,
                same_source=same_src,
                same_url=same_u,
                shared_metadata_signal=True,
                redundancy_class=RedundancyClass.EXACT_DUPLICATE,
                independence_score=0.0,
                explanation=explanation,
            )

        # Metadata relationship checks
        same_url = bool(a.norm_url is not None and b.norm_url is not None and a.norm_url == b.norm_url)
        same_domain = bool(
            a.norm_domain is not None and b.norm_domain is not None and a.norm_domain == b.norm_domain
        )
        same_source = bool(
            a.norm_source is not None and b.norm_source is not None and a.norm_source == b.norm_source
        )
        shared_metadata = same_url or same_domain or same_source

        # Exact textual match check
        exact_text_match = bool(
            a.norm_text and b.norm_text and a.norm_text.lower() == b.norm_text.lower()
        )

        # Token Jaccard similarity
        sim = compute_jaccard_similarity(a.token_set, b.token_set)

        # Meaningful tokens check (§26)
        min_tokens = self.settings.minimum_meaningful_tokens
        has_meaningful_tokens = a.token_count >= min_tokens and b.token_count >= min_tokens

        # Hierarchical Decision Tree (§22)
        # Layer 1: Exact Duplicate
        if same_url or (exact_text_match and has_meaningful_tokens):
            redundancy_class = RedundancyClass.EXACT_DUPLICATE
            is_exact = True
            is_near = True
            ind_score = 0.0
            trigger = "matching canonical source URL" if same_url else "identical normalized text"
            explanation = (
                f"Documents '{a.doc_id}' and '{b.doc_id}' share {trigger} (similarity={sim:.4f}); "
                f"classified as EXACT_DUPLICATE with independence=0.0000. {DISCLAIMER_PHRASE}"
            )

        # Layer 2: Near-Duplicate
        elif sim >= self.settings.near_duplicate_threshold:
            if not has_meaningful_tokens:
                # §26 Short text policy: cannot exceed WEAKLY_RELATED without hard URL match
                redundancy_class = RedundancyClass.WEAKLY_RELATED
                is_exact = False
                is_near = False
                ind_score = round(max(0.0, min(1.0, 1.0 - sim)), 4)
                explanation = (
                    f"Documents '{a.doc_id}' and '{b.doc_id}' have high token similarity ({sim:.4f}), "
                    f"but token count (min={min(a.token_count, b.token_count)}) is below the meaningful "
                    f"threshold ({min_tokens}); capped at WEAKLY_RELATED with independence={ind_score:.4f}. "
                    f"{DISCLAIMER_PHRASE}"
                )
            else:
                redundancy_class = RedundancyClass.REDUNDANT
                is_exact = False
                is_near = True
                ind_score = round(max(0.0, min(1.0, 1.0 - sim)), 4)
                explanation = (
                    f"Documents '{a.doc_id}' and '{b.doc_id}' have normalized token Jaccard similarity of "
                    f"{sim:.4f}, exceeding near-duplicate threshold {self.settings.near_duplicate_threshold:.2f}; "
                    f"classified as REDUNDANT with independence={ind_score:.4f}. {DISCLAIMER_PHRASE}"
                )

        # Layer 3: Weakly Related
        elif sim >= self.settings.weak_relation_threshold or (same_domain and same_source):
            redundancy_class = RedundancyClass.WEAKLY_RELATED
            is_exact = False
            is_near = False
            if sim >= self.settings.weak_relation_threshold:
                ind_score = round(max(0.0, min(1.0, 1.0 - sim)), 4)
                explanation = (
                    f"Documents '{a.doc_id}' and '{b.doc_id}' have moderate token similarity of {sim:.4f} "
                    f"(>= weak threshold {self.settings.weak_relation_threshold:.2f}); "
                    f"classified as WEAKLY_RELATED with independence={ind_score:.4f}. {DISCLAIMER_PHRASE}"
                )
            else:
                # Shared source/domain metadata with low text similarity (§16, §22)
                ind_score = round(max(0.0, min(1.0, (1.0 - sim) * 0.90)), 4)
                explanation = (
                    f"Documents '{a.doc_id}' and '{b.doc_id}' originate from the same source/domain "
                    f"('{a.norm_domain}') but have low text similarity ({sim:.4f}); "
                    f"shared origin is treated as contextual relation (WEAKLY_RELATED) rather than duplication "
                    f"with independence={ind_score:.4f}. {DISCLAIMER_PHRASE}"
                )

        # Layer 4: Independent
        else:
            redundancy_class = RedundancyClass.INDEPENDENT
            is_exact = False
            is_near = False
            ind_score = round(max(0.0, min(1.0, 1.0 - sim)), 4)
            explanation = (
                f"Documents '{a.doc_id}' and '{b.doc_id}' have token similarity of {sim:.4f} and "
                f"no matching source URL or shared publisher context; "
                f"classified as INDEPENDENT with independence={ind_score:.4f}. {DISCLAIMER_PHRASE}"
            )

        return IndependencePairResult(
            document_id_a=a.doc_id,
            document_id_b=b.doc_id,
            text_similarity=sim,
            is_exact_duplicate=is_exact,
            is_near_duplicate=is_near,
            same_domain=same_domain,
            same_source=same_source,
            same_url=same_url,
            shared_metadata_signal=shared_metadata,
            redundancy_class=redundancy_class,
            independence_score=ind_score,
            explanation=explanation,
        )

    def compare_batch(self, items: Sequence[Any]) -> list[IndependencePairResult]:
        """Compare all canonical pairs in an evidence sequence (§28).

        Produces canonical pair ordering: (0, 1), (0, 2), ..., (1, 2), ...
        Without (B, A) inversion or self-comparison entries. Does NOT mutate inputs.
        """
        if not isinstance(items, Sequence):
            raise SourceIndependenceInputError(
                f"Expected Sequence for items, got {type(items).__name__}"
            )
        n = len(items)
        if n < 2:
            return []

        # Preprocess each item once to avoid redundant tokenization across O(N^2) pairs
        preprocessed = [self._preprocess_item(it) for it in items]

        results: list[IndependencePairResult] = []
        for i in range(n):
            for j in range(i + 1, n):
                results.append(self._compare_preprocessed(preprocessed[i], preprocessed[j]))
        return results

    def analyze_set(self, items: Sequence[Any]) -> IndependenceSetResult:
        """Perform group-level complete-linkage redundancy clustering across an evidence set (§23, §24).

        Complete-linkage guarantees that all items in a cluster are mutually redundant,
        avoiding the chaining pathology of single-linkage clustering.
        """
        if not isinstance(items, Sequence):
            raise SourceIndependenceInputError(
                f"Expected Sequence for items, got {type(items).__name__}"
            )
        n = len(items)
        if n == 0:
            return IndependenceSetResult(
                total_documents=0,
                pairwise_comparisons=[],
                clusters=[],
                independent_document_ids=[],
                mean_pairwise_similarity=0.0,
            )

        pairwise = self.compare_batch(items)
        doc_ids = [self._extract_document_fields(it)[0] for it in items]

        if n == 1:
            return IndependenceSetResult(
                total_documents=1,
                pairwise_comparisons=[],
                clusters=[],
                independent_document_ids=doc_ids,
                mean_pairwise_similarity=0.0,
            )

        # Build adjacency matrix of redundant relationships (EXACT_DUPLICATE or REDUNDANT)
        redundant_pairs: set[tuple[str, str]] = set()
        total_sim = 0.0
        for p in pairwise:
            total_sim += p.text_similarity
            if p.redundancy_class in (RedundancyClass.EXACT_DUPLICATE, RedundancyClass.REDUNDANT):
                redundant_pairs.add((p.document_id_a, p.document_id_b))
                redundant_pairs.add((p.document_id_b, p.document_id_a))

        mean_sim = round(total_sim / len(pairwise), 4) if pairwise else 0.0

        # Complete-linkage clique discovery (greedy deterministic search)
        assigned: set[str] = set()
        clusters: list[IndependenceCluster] = []
        cluster_id = 0

        # Preserve canonical input order for stability
        for doc in doc_ids:
            if doc in assigned:
                continue
            # Potential candidates that are redundant with 'doc'
            candidates = [
                other
                for other in doc_ids
                if other not in assigned and other != doc and (doc, other) in redundant_pairs
            ]
            if not candidates:
                continue

            current_cluster = [doc]
            for cand in candidates:
                # Complete linkage check: candidate must be redundant with ALL members of current_cluster
                if all((cand, member) in redundant_pairs for member in current_cluster):
                    current_cluster.append(cand)

            if len(current_cluster) > 1:
                # Determine dominant cluster type
                is_all_exact = all(
                    any(
                        p.redundancy_class == RedundancyClass.EXACT_DUPLICATE
                        for p in pairwise
                        if set([p.document_id_a, p.document_id_b]) == set([m1, m2])
                    )
                    for i_idx, m1 in enumerate(current_cluster)
                    for m2 in current_cluster[i_idx + 1 :]
                )
                c_type = (
                    RedundancyClass.EXACT_DUPLICATE
                    if is_all_exact
                    else RedundancyClass.REDUNDANT
                )
                clusters.append(
                    IndependenceCluster(
                        cluster_id=cluster_id,
                        document_ids=sorted(current_cluster),
                        cluster_type=c_type,
                    )
                )
                cluster_id += 1
                assigned.update(current_cluster)

        independent_ids = [doc for doc in doc_ids if doc not in assigned]

        return IndependenceSetResult(
            total_documents=n,
            pairwise_comparisons=pairwise,
            clusters=clusters,
            independent_document_ids=sorted(independent_ids),
            mean_pairwise_similarity=mean_sim,
        )


# Module-level singleton management
_SINGLETON_DETECTOR: SourceIndependenceDetector | None = None


def get_source_independence_detector(
    settings: SourceIndependenceSettings | None = None,
) -> SourceIndependenceDetector:
    """Return or initialize the singleton SourceIndependenceDetector instance."""
    global _SINGLETON_DETECTOR
    if _SINGLETON_DETECTOR is None or settings is not None:
        _SINGLETON_DETECTOR = SourceIndependenceDetector(settings=settings)
    return _SINGLETON_DETECTOR


def reset_source_independence_detector() -> None:
    """Reset the singleton detector instance (primarily for testing)."""
    global _SINGLETON_DETECTOR
    _SINGLETON_DETECTOR = None
