"""Evidence Pollution & Circularity Defense Engine (Step 4.4).

Detects ungrounded evidence, exact duplicate / repeated evidence, and self-referential
generated content without ML models, network calls, or LLM inference.
Strictly honors the Anti-Fabrication Contract (§5) regarding absent citation graph metadata.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from app.services.evidence_pollution_config import (
    DEFAULT_BOILERPLATE_PATTERNS,
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DEFAULT_MINIMUM_TOKEN_LENGTH,
    DEFAULT_STOPWORDS,
    DEFAULT_UNGROUNDED_THRESHOLD,
    DEFAULT_WEAK_GROUNDING_THRESHOLD,
    DISCLAIMER_PHRASE,
    POLLUTION_BOILERPLATE_THRESHOLD,
    POLLUTION_CIRCULARITY_THRESHOLD,
    POLLUTION_CLEAN_RETENTION_TARGET,
    POLLUTION_CLEAN_THRESHOLD,
    POLLUTION_LOW_GROUNDING_THRESHOLD,
    EvidencePollutionConfigError,
    EvidencePollutionInputError,
    EvidencePollutionSettings,
)
from app.services.evidence_pollution_models import (
    BatchPollutionResult,
    EvidencePollutionResult,
    PollutionClass,
)
from app.services.source_independence_scorer import (
    normalize_domain,
    normalize_text,
    normalize_url,
    tokenize,
)

logger = logging.getLogger("hallucination_verifier.evidence_pollution")


def detect_boilerplate(
    text: str,
    patterns: Sequence[re.Pattern[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Detect common web scraper navigation headers, footers, cookie banners, or subscription templates.

    Returns:
        tuple of (is_boilerplate: bool, matched_phrases: list[str])
    """
    if not text:
        return False, []
    pats = patterns if patterns is not None else DEFAULT_BOILERPLATE_PATTERNS
    matched: list[str] = []
    for pat in pats:
        m = pat.search(text)
        if m:
            matched.append(m.group(0).strip())
    return (len(matched) > 0, matched)


def classify_pollution_gate(
    pollution_score: float,
    settings: EvidencePollutionSettings | None = None,
) -> str:
    """Classify pollution score into the 4-layer hierarchy (§8.4).

    Returns:
        One of: 'CIRCULARITY', 'LOW_GROUNDING', 'BOILERPLATE', 'WEAK_SIGNAL', 'CLEAN'.
    """
    cfg = settings or EvidencePollutionSettings()
    if pollution_score >= cfg.circularity_threshold:
        return "CIRCULARITY"
    if pollution_score >= cfg.low_grounding_threshold:
        return "LOW_GROUNDING"
    if pollution_score >= cfg.boilerplate_threshold:
        return "BOILERPLATE"
    if pollution_score >= cfg.clean_threshold:
        return "WEAK_SIGNAL"
    return "CLEAN"


def extract_meaningful_tokens(
    text: str,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    min_length: int = DEFAULT_MINIMUM_TOKEN_LENGTH,
) -> frozenset[str]:
    """Extract unique, non-stopword, non-trivial word tokens from text (§7).

    Excludes stopwords, tokens shorter than min_length, and purely numeric strings.
    """
    raw_tokens = tokenize(text)
    meaningful = [
        tok
        for tok in raw_tokens
        if tok not in stopwords and len(tok) >= min_length and not tok.isdigit()
    ]
    return frozenset(meaningful)


def compute_directional_overlap(tokens_c: frozenset[str], tokens_e: frozenset[str]) -> float:
    """Compute directional token overlap Overlap(C, E) = |C ∩ E| / |C| (§6.1).

    Guarded: |Tokens(C)| == 0 -> 0.0.
    Directional: Overlap(C, E) != Overlap(E, C) in general.
    """
    if not tokens_c:
        return 0.0
    return len(tokens_c & tokens_e) / len(tokens_c)


def compute_grounding_score(meaningful_c: frozenset[str], meaningful_e: frozenset[str]) -> float:
    """Compute directional lexical grounding score Grounding(C, E) = |C_m ∩ E_m| / |C_m| (§7).

    Guarded: |Meaningful(C)| == 0 -> 0.0.
    """
    if not meaningful_c:
        return 0.0
    return len(meaningful_c & meaningful_e) / len(meaningful_c)


class EvidencePollutionDetector:
    """Deterministic lexical evidence pollution and circularity defense engine (§4, §5).

    Attributes:
        settings: Heuristic thresholds and parameter settings.
    """

    def __init__(self, settings: EvidencePollutionSettings | None = None) -> None:
        """Initialize detector with settings."""
        self.settings = settings or EvidencePollutionSettings()

    def _extract_document_fields(
        self, item: Any
    ) -> tuple[str, str, str | None, str | None, str | None]:
        """Extract (document_id, text, source, source_url, domain) from input objects."""
        if item is None:
            raise EvidencePollutionInputError("Evidence item cannot be None.")

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
            doc_id = getattr(item, "document_id", None) or getattr(item, "id", None)
            text = getattr(item, "text", None)
            source = getattr(item, "source", None)
            source_url = getattr(item, "source_url", None)
            domain = getattr(item, "domain", None)

        if doc_id is None:
            raise EvidencePollutionInputError(
                f"Evidence item missing valid string document_id: {item!r}"
            )
        doc_id = str(doc_id).strip()
        if not doc_id:
            raise EvidencePollutionInputError(
                f"Evidence item missing valid string document_id: {item!r}"
            )
        if text is None or not isinstance(text, str):
            raise EvidencePollutionInputError(
                f"Evidence item '{doc_id}' missing valid string text: {item!r}"
            )

        return (
            doc_id,
            text,
            str(source) if source is not None else None,
            str(source_url) if source_url is not None else None,
            str(domain) if domain is not None else None,
        )

    def evaluate(
        self,
        claim: str,
        evidence: Any,
        claim_origin_url: str | None = None,
        seen_urls: set[str] | None = None,
        seen_texts: set[str] | None = None,
        seen_doc_ids: set[str] | None = None,
    ) -> EvidencePollutionResult:
        """Evaluate an individual claim-evidence pair for pollution and lexical grounding (§6, §7, §8).

        Args:
            claim: The natural language atomic claim string.
            evidence: Evidence document (CorpusDocument, RetrievalResult, or dict).
            claim_origin_url: Optional source URL of the claim/response for self-reference detection.
            seen_urls: Set of normalized URLs previously seen in the current evidence set.
            seen_texts: Set of normalized text strings previously seen in the current evidence set.
            seen_doc_ids: Set of document IDs previously seen in the current evidence set.

        Returns:
            EvidencePollutionResult with normalized scores, classification, flags, and explanation.
        """
        if claim is None or not isinstance(claim, str):
            raise EvidencePollutionInputError(f"Claim must be a valid string, got {type(claim).__name__}")

        doc_id, raw_text, source, source_url, domain = self._extract_document_fields(evidence)

        # 1. Normalization (raw texts remain unchanged §11)
        norm_claim = normalize_text(claim)
        norm_ev = normalize_text(raw_text)

        # 2. Token sets
        all_tokens_claim = frozenset(tokenize(claim))
        all_tokens_ev = frozenset(tokenize(raw_text))

        meaningful_claim = extract_meaningful_tokens(
            claim, self.settings.stopwords, self.settings.minimum_token_length
        )
        meaningful_ev = extract_meaningful_tokens(
            raw_text, self.settings.stopwords, self.settings.minimum_token_length
        )

        # 3. Directional Overlaps and Grounding (§6.1, §7)
        claim_ev_overlap = round(compute_directional_overlap(all_tokens_claim, all_tokens_ev), 4)
        ev_claim_overlap = round(compute_directional_overlap(all_tokens_ev, all_tokens_claim), 4)
        grounding = round(compute_grounding_score(meaningful_claim, meaningful_ev), 4)

        # 4. URL and Metadata comparisons
        norm_ev_url = normalize_url(source_url)
        norm_claim_url = normalize_url(claim_origin_url)

        is_exact_claim_text = bool(
            norm_claim and norm_ev and norm_claim.lower() == norm_ev.lower()
        )
        is_claim_origin_match = bool(
            norm_ev_url and norm_claim_url and norm_ev_url == norm_claim_url
        )

        is_repeated_url = bool(norm_ev_url and seen_urls and norm_ev_url in seen_urls)
        is_repeated_text = bool(norm_ev and seen_texts and norm_ev.lower() in seen_texts)
        is_repeated_doc_id = bool(seen_doc_ids and doc_id in seen_doc_ids)

        # 5. Meaningful token constraints (§10)
        min_tokens = self.settings.minimum_meaningful_tokens
        has_meaningful_tokens = (
            len(meaningful_claim) >= min_tokens and len(meaningful_ev) >= min_tokens
        )

        # 6. Boilerplate detection (§8.4)
        has_bp, bp_matches = detect_boilerplate(raw_text, self.settings.boilerplate_patterns)

        # 7. Precedence Decision Hierarchy (§8.1, §8.4, §9)
        # Precedence: SELF_REFERENTIAL -> EXACT_DUPLICATE -> UNGROUNDED -> BOILERPLATE -> WEAK_POLLUTION_SIGNAL -> CLEAN

        # Layer 1: Self-Referential Generated Content (P >= 0.90)
        if is_claim_origin_match or (is_exact_claim_text and has_meaningful_tokens):
            classification = PollutionClass.SELF_REFERENTIAL
            pollution_score = 1.0
            is_exact = is_exact_claim_text
            is_self = True
            trigger = (
                "matching claim origin URL"
                if is_claim_origin_match
                else "identical text repetition of the claim"
            )
            explanation = (
                f"Evidence '{doc_id}' appears self-referential due to {trigger} "
                f"(overlap={claim_ev_overlap:.4f}, grounding={grounding:.4f}); "
                f"classified as SELF_REFERENTIAL with pollution_score=1.0000. "
                f"Full citation-graph circularity detection is not supported by the current repository schema. "
                f"{DISCLAIMER_PHRASE}"
            )

        # Layer 2: Exact Duplicate / Repeated Evidence (P >= 0.90)
        elif is_repeated_url or is_repeated_text or is_repeated_doc_id:
            classification = PollutionClass.EXACT_DUPLICATE
            pollution_score = 1.0
            is_exact = True
            is_self = False
            reasons = []
            if is_repeated_url:
                reasons.append("repeated canonical URL")
            if is_repeated_text:
                reasons.append("repeated identical text")
            if is_repeated_doc_id:
                reasons.append("repeated document ID")
            trigger = " and ".join(reasons)
            explanation = (
                f"Evidence '{doc_id}' is duplicated within the retrieved set due to {trigger}; "
                f"classified as EXACT_DUPLICATE with pollution_score=1.0000. "
                f"Full citation-graph circularity detection is not supported by the current repository schema. "
                f"{DISCLAIMER_PHRASE}"
            )

        # Layer 3: Boilerplate / Scraped Template Header or Footer (P >= 0.60)
        elif has_bp and (grounding < 0.50 or len(bp_matches) >= 2):
            classification = PollutionClass.BOILERPLATE
            pollution_score = self.settings.boilerplate_threshold
            is_exact = False
            is_self = False
            bp_str = ", ".join(f"'{m}'" for m in bp_matches[:3])
            explanation = (
                f"Evidence '{doc_id}' contains web navigation template / scraper boilerplate "
                f"({bp_str}) (grounding={grounding:.4f}); classified as BOILERPLATE "
                f"with pollution_score={self.settings.boilerplate_threshold:.4f}. {DISCLAIMER_PHRASE}"
            )

        # Layer 4: Ungrounded Evidence (P >= 0.80)
        elif (
            grounding < self.settings.ungrounded_threshold
            and len(meaningful_claim) >= min_tokens
            and len(meaningful_ev) >= min_tokens
        ):
            classification = PollutionClass.UNGROUNDED
            pollution_score = self.settings.low_grounding_threshold
            is_exact = False
            is_self = False
            explanation = (
                f"Evidence '{doc_id}' exhibits low lexical grounding relative to the claim "
                f"(grounding={grounding:.4f} < ungrounded_threshold={self.settings.ungrounded_threshold:.2f}); "
                f"classified as UNGROUNDED with pollution_score={self.settings.low_grounding_threshold:.4f}. "
                f"Note: Lexical grounding is a lexical filter and not semantic entailment. "
                f"{DISCLAIMER_PHRASE}"
            )

        # Layer 5: Weak Pollution / Suspicious Signal / Short-Text Boundary (§10)
        elif (
            not has_meaningful_tokens
            or grounding < self.settings.weak_grounding_threshold
        ):
            classification = PollutionClass.WEAK_POLLUTION_SIGNAL
            pollution_score = 0.30
            is_exact = False
            is_self = False
            if not has_meaningful_tokens:
                min_found = min(len(meaningful_claim), len(meaningful_ev))
                explanation = (
                    f"Evidence '{doc_id}' or claim contains fewer meaningful tokens ({min_found}) "
                    f"than the required threshold ({min_tokens}); evaluated conservatively as "
                    f"WEAK_POLLUTION_SIGNAL with pollution_score=0.3000 (overlap={claim_ev_overlap:.4f}, "
                    f"grounding={grounding:.4f}). {DISCLAIMER_PHRASE}"
                )
            else:
                explanation = (
                    f"Evidence '{doc_id}' exhibits weak lexical grounding relative to the claim "
                    f"(grounding={grounding:.4f} < weak_threshold={self.settings.weak_grounding_threshold:.2f}); "
                    f"classified as WEAK_POLLUTION_SIGNAL with pollution_score=0.3000. {DISCLAIMER_PHRASE}"
                )

        # Layer 6: Clean Evidence (P < 0.20)
        else:
            classification = PollutionClass.CLEAN
            pollution_score = 0.0
            is_exact = False
            is_self = False
            explanation = (
                f"Evidence '{doc_id}' is well-grounded (grounding={grounding:.4f}) and exhibits no "
                f"detectable self-reference, duplicate, or boilerplate contamination; classified as CLEAN "
                f"with pollution_score=0.0000. {DISCLAIMER_PHRASE}"
            )

        return EvidencePollutionResult(
            document_id=doc_id,
            pollution_score=pollution_score,
            classification=classification,
            claim_token_count=len(meaningful_claim),
            evidence_token_count=len(meaningful_ev),
            claim_evidence_overlap=claim_ev_overlap,
            evidence_claim_overlap=ev_claim_overlap,
            grounding_score=grounding,
            is_exact_duplicate=is_exact,
            is_self_referential=is_self,
            is_boilerplate=has_bp,
            is_circular=False,  # Strictly False per Anti-Fabrication Contract §5
            circularity_supported=False,  # Strictly False per Anti-Fabrication Contract §5
            explanation=explanation,
        )

    def evaluate_batch(
        self,
        claim: str,
        evidence_items: Sequence[Any],
        claim_origin_url: str | None = None,
        batch_size: int = 10,
    ) -> BatchPollutionResult:
        """Evaluate a set of retrieved evidence items against a claim (§15, §8.4).

        Tracks seen URLs, normalized texts, and document IDs across the set to detect
        in-batch duplicates and circular repetition. Executes in memory-safe chunks
        bounded by batch_size (default: 10).
        """
        if not isinstance(evidence_items, Sequence):
            raise EvidencePollutionInputError(
                f"Expected Sequence for evidence_items, got {type(evidence_items).__name__}"
            )
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise EvidencePollutionInputError(
                f"batch_size must be an integer, got {type(batch_size).__name__}"
            )
        if batch_size < 1:
            raise EvidencePollutionInputError(
                f"batch_size must be >= 1, got {batch_size}"
            )

        seen_urls: set[str] = set()
        seen_texts: set[str] = set()
        seen_doc_ids: set[str] = set()

        evaluations: list[EvidencePollutionResult] = []
        polluted_ids: list[str] = []
        ungrounded_ids: list[str] = []
        boilerplate_ids: list[str] = []
        clean_ids: list[str] = []

        # Process in memory-safe chunks
        for i in range(0, len(evidence_items), batch_size):
            chunk = evidence_items[i : i + batch_size]
            for item in chunk:
                res = self.evaluate(
                    claim=claim,
                    evidence=item,
                    claim_origin_url=claim_origin_url,
                    seen_urls=seen_urls,
                    seen_texts=seen_texts,
                    seen_doc_ids=seen_doc_ids,
                )
                evaluations.append(res)

                # Record seen metadata for subsequent items in the batch
                doc_id, raw_text, _, source_url, _ = self._extract_document_fields(item)
                seen_doc_ids.add(doc_id)
                if raw_text:
                    seen_texts.add(normalize_text(raw_text).lower())
                norm_u = normalize_url(source_url)
                if norm_u:
                    seen_urls.add(norm_u)

                # Categorize
                if res.classification in (
                    PollutionClass.SELF_REFERENTIAL,
                    PollutionClass.EXACT_DUPLICATE,
                    PollutionClass.UNGROUNDED,
                    PollutionClass.BOILERPLATE,
                ):
                    polluted_ids.append(res.document_id)
                if res.classification == PollutionClass.UNGROUNDED:
                    ungrounded_ids.append(res.document_id)
                elif res.classification == PollutionClass.BOILERPLATE:
                    boilerplate_ids.append(res.document_id)
                elif res.classification == PollutionClass.CLEAN:
                    clean_ids.append(res.document_id)

        return BatchPollutionResult(
            total_items=len(evaluations),
            evaluations=evaluations,
            polluted_document_ids=polluted_ids,
            ungrounded_document_ids=ungrounded_ids,
            boilerplate_document_ids=boilerplate_ids,
            clean_document_ids=clean_ids,
        )


# Singleton accessors
_SINGLETON_DETECTOR: EvidencePollutionDetector | None = None


def get_evidence_pollution_detector(
    settings: EvidencePollutionSettings | None = None,
) -> EvidencePollutionDetector:
    """Return or initialize the singleton EvidencePollutionDetector instance."""
    global _SINGLETON_DETECTOR
    if _SINGLETON_DETECTOR is None or settings is not None:
        _SINGLETON_DETECTOR = EvidencePollutionDetector(settings=settings)
    return _SINGLETON_DETECTOR


def reset_evidence_pollution_detector() -> None:
    """Reset the singleton detector instance (for test isolation)."""
    global _SINGLETON_DETECTOR
    _SINGLETON_DETECTOR = None
