"""API route for analyzing one claim and planning its verification."""

from fastapi import APIRouter

from app.schemas import ClaimAnalysisRequest, ClaimAnalysisResponse
from app.services.claim_analysis_service import analyze_claim

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze-claim", response_model=ClaimAnalysisResponse)
async def analyze_extracted_claim(
    request: ClaimAnalysisRequest,
) -> ClaimAnalysisResponse:
    """Assess one extracted claim and return a verification plan.

    This endpoint is the fourth pipeline stage. It estimates verification
    complexity based on semantic qualifiers (dates, numbers, entities, scopes)
    and recommends retrieval and challenge actions.
    """

    return analyze_claim(
        claim_text=request.claim_text,
        claim_id=request.claim_id,
        atomic_claim=request,
        query_id=request.query_id,
    )
