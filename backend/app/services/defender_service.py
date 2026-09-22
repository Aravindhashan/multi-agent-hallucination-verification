"""Service orchestration for the Defender stage of verification."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.schemas import DefenderResponse, DefenderResult
from app.services.defender_agent import BaseDefender
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.simple_defender import SimpleDefender


def defend_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
) -> DefenderResponse:
    """Normalize one claim and gather evidence intended to support it.

    Uses injected retriever (defaults to get_evidence_retriever()).
    """

    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain non-whitespace characters.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    if retriever is None:
        from app.dependencies import get_evidence_retriever
        retriever = get_evidence_retriever()

    defender: BaseDefender = SimpleDefender(retriever=retriever)
    supporting_evidence, query_used = defender.gather_supporting_evidence(
        normalized_claim_text
    )

    # The internal result captures provider-independent Defender output before
    # the correlation ID is added to the public API response.
    result = DefenderResult(
        claim_text=normalized_claim_text,
        supporting_evidence=supporting_evidence,
        query_used=query_used,
        defended_at=datetime.now(timezone.utc),
    )

    if query_id is not None:
        try:
            from app.services.session_store import get_session_store
            session_store = get_session_store()
            if hasattr(session_store, "record_claim_evidence"):
                session_store.record_claim_evidence(
                    query_id=query_id,
                    claim_id=resolved_claim_id,
                    evidence_type="supporting",
                    evidence=[e.model_dump() for e in supporting_evidence],
                )
        except Exception:
            pass

    return DefenderResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=result.claim_text,
        supporting_evidence=result.supporting_evidence,
        query_used=result.query_used,
        defended_at=result.defended_at,
    )
