"""API routes for Module 9 Natural Language Inference (NLI) claim adjudication."""

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    AdjudicationBatchRequest,
    AdjudicationBatchResponse,
    AdjudicationRequest,
    AdjudicationResponse,
)
from app.services.adjudication_service import (
    AdjudicationService,
    get_adjudication_service,
)

router = APIRouter(prefix="/api/v1", tags=["adjudication"])


def get_adjudication_service_dep() -> AdjudicationService:
    """FastAPI dependency yielding the singleton AdjudicationService."""
    return get_adjudication_service()


@router.post("/adjudicate", response_model=AdjudicationResponse)
async def adjudicate_claim_endpoint(
    request: AdjudicationRequest,
    service: AdjudicationService = Depends(get_adjudication_service_dep),
) -> AdjudicationResponse:
    """Run DeBERTa cross-encoder NLI claim adjudication over federated evidence."""
    try:
        return service.adjudicate_claim(
            claim_text=request.claim_text,
            claim_id=request.claim_id,
            query_id=request.query_id,
            supporting_evidence=request.supporting_evidence or None,
            contradicting_evidence=request.contradicting_evidence or None,
            run_debate=request.run_debate,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Claim adjudication inference failed: {exc}",
        )


@router.post("/adjudicate-batch", response_model=AdjudicationBatchResponse)
async def adjudicate_claims_batch_endpoint(
    request: AdjudicationBatchRequest,
    service: AdjudicationService = Depends(get_adjudication_service_dep),
) -> AdjudicationBatchResponse:
    """Run batch claim adjudication with memory-safe chunking."""
    try:
        responses = service.adjudicate_claims_batch(
            claims=request.requests,
            batch_size=2,
        )
        mean_conf = (
            sum(r.confidence_score for r in responses) / len(responses)
            if responses
            else 0.0
        )
        return AdjudicationBatchResponse(
            responses=responses,
            total_claims_processed=len(responses),
            mean_confidence=round(mean_conf, 4),
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Batch claim adjudication inference failed: {exc}",
        )
