"""Service orchestration for claim analysis and verification planning."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.schemas import ClaimAnalysisResponse
from app.services.claim_analyzer import BaseClaimAnalyzer
from app.services.session_store import get_session_store
from app.services.simple_claim_analyzer import SimpleClaimAnalyzer


def _create_claim_analyzer() -> BaseClaimAnalyzer:
    """Return the claim analyzer configured for the current application.

    Module 4 always uses the transparent heuristic analyzer. A future settings
    option can select another ``BaseClaimAnalyzer`` implementation without
    changing the service's public function or the API route.
    """

    return SimpleClaimAnalyzer()


def analyze_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    atomic_claim: object = None,
    query_id: UUID | None = None,
) -> ClaimAnalysisResponse:
    """Normalize one claim, analyze it, and add response metadata.

    A caller-provided claim ID is preserved so Module 3 output can be correlated
    manually. If it is absent, UUID4 creates a standalone analysis identifier.
    """

    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain non-whitespace characters.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    # If atomic_claim not supplied, parse with NLP extractor to extract anchors
    resolved_atomic = atomic_claim
    if resolved_atomic is None:
        try:
            from app.services.nlp_claim_extractor import NLPClaimExtractor

            extractor = NLPClaimExtractor()
            parsed = extractor.extract_atomic_claims(normalized_claim_text)
            if parsed.claims:
                resolved_atomic = parsed.claims[0]
        except Exception:
            resolved_atomic = None

    analyzer = _create_claim_analyzer()
    result = analyzer.analyze(normalized_claim_text, atomic_claim=resolved_atomic)

    # If query_id provided, record into session store
    if query_id is not None:
        session_store = get_session_store()
        session_store.record_analysis(
            query_id=query_id,
            complexity_score=result.complexity_score,
            verification_plan=result.verification_plan,
        )
        session_store.record_claim_analysis(
            query_id=query_id,
            claim_id=resolved_claim_id,
            claim_text=normalized_claim_text,
            complexity_score=result.complexity_score,
            verification_plan=result.verification_plan,
            requires_deep_verification=result.requires_deep_verification,
            ablation_log=result.ablation_log.model_dump() if result.ablation_log else None,
        )
        if result.ablation_log is not None:
            session_store.update_metadata(
                query_id=query_id,
                key="planning_ablation_log",
                value=result.ablation_log.model_dump(),
            )

    return ClaimAnalysisResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=normalized_claim_text,
        complexity_score=result.complexity_score,
        verification_plan=result.verification_plan,
        requires_deep_verification=result.requires_deep_verification,
        ablation_log=result.ablation_log,
        analyzed_at=datetime.now(timezone.utc),
    )
