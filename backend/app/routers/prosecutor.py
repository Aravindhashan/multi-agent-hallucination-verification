"""API route for gathering evidence that challenges one claim."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_evidence_retriever
from app.schemas import (
    ProsecutorRequest,
    ProsecutorResponse,
    RecallAblationRequest,
    RecallAblationSummary,
)
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_service import prosecute_claim, prosecute_claims_batch
from app.services.recall_ablation import PerspectiveRecallAblationHarness

router = APIRouter(prefix="/api/v1", tags=["prosecutor"])


@router.post("/prosecute-claim", response_model=ProsecutorResponse)
async def prosecute_extracted_claim(
    request: ProsecutorRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> ProsecutorResponse:
    """Run the Prosecutor stage for a single extracted claim with injected retriever."""

    try:
        return prosecute_claim(
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


@router.post("/prosecute-claims-batch", response_model=list[ProsecutorResponse])
async def prosecute_extracted_claims_batch(
    requests: list[ProsecutorRequest],
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> list[ProsecutorResponse]:
    """Run the Prosecutor stage for a batch of extracted claims with memory-safe slicing."""

    try:
        return prosecute_claims_batch(
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


@router.post("/prosecutor/ablation-recall", response_model=RecallAblationSummary)
async def evaluate_perspective_recall_ablation(
    request: RecallAblationRequest,
    retriever: BaseEvidenceRetriever = Depends(get_evidence_retriever),
) -> RecallAblationSummary:
    """Evaluate evidence recall lift and Jaccard complementarity across perspectives."""

    try:
        harness = PerspectiveRecallAblationHarness(retriever=retriever)
        return harness.evaluate_claims_batch(
            claims=request.claims,
            gold_evidence_by_claim=request.gold_evidence_by_claim,
            gold_contradictions_by_claim=request.gold_contradictions_by_claim,
            top_k=request.top_k,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Recall ablation evaluation failed: {exc}",
        )

