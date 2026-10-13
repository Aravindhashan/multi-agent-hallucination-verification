"""API routes for multi-turn evidentiary debate orchestration."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_evidence_retriever
from app.schemas import DebateClaimRequest, DebateClaimResponse
from app.services.debate_service import debate_claim, debate_claims_batch
from app.services.evidence_retriever import BaseEvidenceRetriever

router = APIRouter(prefix="/api/v1", tags=["debate"])


@router.post("/debate-claim", response_model=DebateClaimResponse)
async def debate_extracted_claim(
    request: DebateClaimRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> DebateClaimResponse:
    """Run the multi-turn debate orchestrator for a single extracted claim with injected retriever."""
    try:
        return debate_claim(
            claim_text=request.claim_text,
            claim_id=request.claim_id,
            query_id=request.query_id,
            retriever=retriever,
            max_rounds=request.max_rounds,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Debate evidence retrieval failed: {exc}",
        )


@router.post("/debate-claims-batch", response_model=list[DebateClaimResponse])
async def debate_extracted_claims_batch(
    requests: list[DebateClaimRequest],
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> list[DebateClaimResponse]:
    """Run the multi-turn debate orchestrator for a batch of extracted claims with memory-safe slicing."""
    try:
        return debate_claims_batch(
            claims=requests,
            retriever=retriever,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Batch debate evidence retrieval failed: {exc}",
        )
