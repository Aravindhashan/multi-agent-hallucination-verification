"""API route for gathering evidence that supports one claim."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_evidence_retriever
from app.schemas import DefenderRequest, DefenderResponse
from app.services.defender_service import defend_claim, defend_claims_batch
from app.services.evidence_retriever import BaseEvidenceRetriever

router = APIRouter(prefix="/api/v1", tags=["defender"])


@router.post("/defend-claim", response_model=DefenderResponse)
async def defend_extracted_claim(
    request: DefenderRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> DefenderResponse:
    """Run the Defender stage for a single extracted claim with injected retriever."""

    try:
        return defend_claim(
            claim_text=request.claim_text,
            claim_id=request.claim_id,
            query_id=request.query_id,
            retriever=retriever,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Evidence retrieval failed: {exc}",
        )


@router.post("/defend-claims-batch", response_model=list[DefenderResponse])
async def defend_extracted_claims_batch(
    requests: list[DefenderRequest],
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> list[DefenderResponse]:
    """Run the Defender stage for a batch of extracted claims with memory-safe slicing."""

    try:
        return defend_claims_batch(
            claims=requests,
            retriever=retriever,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Batch evidence retrieval failed: {exc}",
        )

