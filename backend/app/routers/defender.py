"""API route for gathering evidence that supports one claim."""

from fastapi import APIRouter, Depends

from app.dependencies import get_evidence_retriever
from app.schemas import DefenderRequest, DefenderResponse
from app.services.defender_service import defend_claim
from app.services.evidence_retriever import BaseEvidenceRetriever

router = APIRouter(prefix="/api/v1", tags=["defender"])


@router.post("/defend-claim", response_model=DefenderResponse)
async def defend_extracted_claim(
    request: DefenderRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> DefenderResponse:
    """Run the Defender stage for a single extracted claim with injected retriever."""

    return defend_claim(
        claim_text=request.claim_text,
        claim_id=request.claim_id,
        query_id=request.query_id,
        retriever=retriever,
    )
