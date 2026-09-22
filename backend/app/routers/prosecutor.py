"""API route for gathering evidence that challenges one claim."""

from fastapi import APIRouter, Depends

from app.dependencies import get_evidence_retriever
from app.schemas import ProsecutorRequest, ProsecutorResponse
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_service import prosecute_claim

router = APIRouter(prefix="/api/v1", tags=["prosecutor"])


@router.post("/prosecute-claim", response_model=ProsecutorResponse)
async def prosecute_extracted_claim(
    request: ProsecutorRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> ProsecutorResponse:
    """Run the Prosecutor stage for a single extracted claim with injected retriever."""

    return prosecute_claim(
        claim_text=request.claim_text,
        claim_id=request.claim_id,
        query_id=request.query_id,
        retriever=retriever,
    )
