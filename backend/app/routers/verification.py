"""API routes for Module 11 Master End-to-End Pipeline Verification."""

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    VerificationBatchRequest,
    VerificationBatchResponse,
    VerificationRequest,
    VerificationResponse,
)
from app.services.verification_service import (
    VerificationService,
    get_verification_service,
)

router = APIRouter(prefix="/api/v1", tags=["verification"])


def get_verification_service_dep() -> VerificationService:
    """FastAPI dependency yielding the singleton VerificationService."""
    return get_verification_service()


@router.post("/verify", response_model=VerificationResponse)
async def verify_endpoint(
    request: VerificationRequest,
    service: VerificationService = Depends(get_verification_service_dep),
) -> VerificationResponse:
    """Execute end-to-end factual verification on a text statement or multi-claim passage."""
    try:
        return service.verify_text(
            text=request.text,
            query_id=request.query_id,
            evidence_mode=request.evidence_mode,
            extraction_method=request.extraction_method,
            max_debate_rounds=request.max_debate_rounds,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Master verification pipeline failed: {exc}",
        )


@router.post("/verify-batch", response_model=VerificationBatchResponse)
async def verify_batch_endpoint(
    request: VerificationBatchRequest,
    service: VerificationService = Depends(get_verification_service_dep),
) -> VerificationBatchResponse:
    """Execute batch end-to-end factual verification with memory-safe chunking."""
    try:
        return service.verify_batch(
            requests=request.requests,
            batch_size=2,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Master verification batch pipeline failed: {exc}",
        )
