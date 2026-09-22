"""API route for evaluating evidence quality, credibility, freshness, and pollution."""

from datetime import datetime, timezone
from fastapi import APIRouter

from app.schemas import EvidenceQualityRequest, EvidenceQualityResponse
from app.services.evidence_quality_service import get_evidence_quality_service
from app.services.session_store import get_session_store

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence-quality"])


@router.post("/evaluate-quality", response_model=EvidenceQualityResponse)
async def evaluate_evidence_quality(
    request: EvidenceQualityRequest,
) -> EvidenceQualityResponse:
    """Evaluate candidate evidence items across all Research Module 4 defense layers.

    Applies:
    1. Source Credibility & Domain Authority Tiering (Step 4.1)
    2. Temporal Freshness & Exponential Age Decay (Step 4.2)
    3. Source Independence & Complete-Linkage Redundancy Clustering (Step 4.3)
    4. Evidence Pollution & Directional Lexical Grounding Defense (Step 4.4)
    5. Kish Effective Sample Size (n_eff) calculation
    """
    quality_service = get_evidence_quality_service()

    report = quality_service.evaluate_evidence(
        claim=request.claim_text,
        evidence_items=request.evidence_items,
        reference_date=request.reference_date,
        claim_origin_url=request.claim_origin_url,
    )

    evaluated_dicts = [item.model_dump(mode="json") for item in report.evaluated_items]

    # If query_id was provided, record metadata into session store
    if request.query_id is not None:
        session_store = get_session_store()
        session_store.update_metadata(
            request.query_id,
            "evidence_quality_evaluation",
            {
                "total_evidence_count": report.total_evidence_count,
                "admissible_count": report.admissible_count,
                "kish_effective_count": report.kish_effective_count,
                "cluster_count": report.cluster_count,
            },
        )

    return EvidenceQualityResponse(
        claim_id=request.claim_id,
        query_id=request.query_id,
        claim_text=report.claim_text,
        total_evidence_count=report.total_evidence_count,
        admissible_count=report.admissible_count,
        polluted_count=report.polluted_count,
        cluster_count=report.cluster_count,
        kish_effective_count=report.kish_effective_count,
        evaluated_items=evaluated_dicts,
        disclaimer=report.disclaimer,
        evaluated_at=report.evaluated_at,
    )
