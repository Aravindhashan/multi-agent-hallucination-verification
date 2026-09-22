"""API route for decomposing an LLM response into candidate claims."""

from uuid import uuid4

from fastapi import APIRouter

from app.schemas import ClaimExtractionRequest, ClaimExtractionResponse
from app.services.claim_extraction_service import extract_claims
from app.services.session_store import get_session_store

router = APIRouter(prefix="/api/v1", tags=["claims"])


@router.post("/extract-claims", response_model=ClaimExtractionResponse)
async def extract_response_claims(
    request: ClaimExtractionRequest,
) -> ClaimExtractionResponse:
    """Extract ordered atomic claims with full qualifiers from an LLM response.

    Preserves caller-supplied query_id when available and updates the pipeline
    session store with extracted claims.
    """

    query_id = request.query_id if request.query_id is not None else uuid4()
    method = getattr(request, "method", "nlp") or "nlp"

    result = extract_claims(
        query_id=query_id,
        response_text=request.response_text,
        query=request.query,
        method=method,
    )

    # Record extracted claims into the pipeline session store
    session_store = get_session_store()
    session_store.record_claims(
        query_id=query_id,
        claims=[claim.model_dump(mode="json") for claim in result.claims],
        extracted_at=result.extracted_at,
    )
    if result.decomposition_metrics is not None:
        session_store.update_metadata(
            query_id=query_id,
            key="decomposition_metrics",
            value=result.decomposition_metrics.model_dump(mode="json"),
        )

    return result
