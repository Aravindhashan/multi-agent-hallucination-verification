"""Service orchestration for the Prosecutor stage of verification."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.schemas import ProsecutorResponse, ProsecutorResult
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_agent import BaseProsecutor
from app.services.simple_prosecutor import SimpleProsecutor


def prosecute_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
) -> ProsecutorResponse:
    """Normalize one claim and gather evidence intended to challenge it.

    Uses injected retriever (defaults to get_evidence_retriever()).
    """

    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain non-whitespace characters.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    if retriever is None:
        from app.dependencies import get_evidence_retriever
        retriever = get_evidence_retriever()

    prosecutor: BaseProsecutor = SimpleProsecutor(retriever=retriever)
    contradicting_evidence, query_used = (
        prosecutor.gather_contradicting_evidence(normalized_claim_text)
    )

    # Capture agent output separately before adding API correlation metadata.
    result = ProsecutorResult(
        claim_text=normalized_claim_text,
        contradicting_evidence=contradicting_evidence,
        query_used=query_used,
        challenged_at=datetime.now(timezone.utc),
    )

    if query_id is not None:
        try:
            from app.services.session_store import get_session_store
            session_store = get_session_store()
            if hasattr(session_store, "record_claim_evidence"):
                session_store.record_claim_evidence(
                    query_id=query_id,
                    claim_id=resolved_claim_id,
                    evidence_type="contradicting",
                    evidence=[e.model_dump() for e in contradicting_evidence],
                )
        except Exception:
            pass

    return ProsecutorResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=result.claim_text,
        contradicting_evidence=result.contradicting_evidence,
        query_used=result.query_used,
        challenged_at=result.challenged_at,
    )
