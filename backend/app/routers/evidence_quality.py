"""API route for evaluating evidence quality, credibility, freshness, and pollution."""

from datetime import datetime, timezone
from fastapi import APIRouter

from app.schemas import (
    DebateQualityBatchRequest,
    DebateQualityRequest,
    DebateQualityResponse,
    EvidenceQualityBatchRequest,
    EvidenceQualityRequest,
    EvidenceQualityResponse,
    PooledClusterBatchRequest,
    PooledClusterReport,
)
from app.services.cluster_pooling import get_cluster_pooling_service
from app.services.evidence_quality_service import get_evidence_quality_service
from app.services.session_store import get_session_store

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence-quality"])


@router.post("/evaluate-quality", response_model=EvidenceQualityResponse)
async def evaluate_evidence_quality(
    request: EvidenceQualityRequest,
) -> EvidenceQualityResponse:
    """Evaluate candidate evidence items across all Research Module 8 defense layers.

    Applies:
    1. Source Credibility & Domain Authority Tiering (Step 8.1)
    2. Anchor-Aware Temporal Freshness & Continuous Envelope (Step 8.2)
    3. Source Independence & Corrected Kish's n_eff Clustering (Step 8.3)
    4. 4-Layer Pollution & Scraper Boilerplate Defense (Step 8.4)
    5. Admissibility Gating & Multi-Dimensional Reliability Weighting (Step 8.5)
    """
    quality_service = get_evidence_quality_service()

    report = quality_service.evaluate_evidence(
        claim=request.claim_text,
        evidence_items=request.evidence_items,
        reference_date=request.reference_date,
        claim_origin_url=request.claim_origin_url,
        temporal_anchor=request.temporal_anchor,
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
        if request.claim_id is not None:
            session_store.record_claim_quality_audit(
                query_id=request.query_id,
                claim_id=request.claim_id,
                quality_data={
                    "total_evidence_count": report.total_evidence_count,
                    "admissible_count": report.admissible_count,
                    "polluted_count": report.polluted_count,
                    "cluster_count": report.cluster_count,
                    "kish_effective_count": report.kish_effective_count,
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


@router.post("/evaluate-quality-batch", response_model=list[EvidenceQualityResponse])
async def evaluate_evidence_quality_batch(
    payload: list[EvidenceQualityRequest] | EvidenceQualityBatchRequest,
) -> list[EvidenceQualityResponse]:
    """Evaluate candidate evidence items across all quality layers for a batch of claims."""
    quality_service = get_evidence_quality_service()
    req_list = payload.requests if isinstance(payload, EvidenceQualityBatchRequest) else payload
    return quality_service.evaluate_claims_batch(req_list)


@router.post("/pool-clusters", response_model=PooledClusterReport)
async def pool_evidence_clusters(
    request: EvidenceQualityRequest,
) -> PooledClusterReport:
    """Pool redundant evidence items into cluster representatives with log-odds reliability weights (Step 8.7)."""
    pooling_service = get_cluster_pooling_service()
    return pooling_service.pool_clusters(
        claim=request.claim_text,
        evidence_items=request.evidence_items,
        reference_date=request.reference_date,
        claim_origin_url=request.claim_origin_url,
        temporal_anchor=request.temporal_anchor,
        claim_id=request.claim_id,
        query_id=request.query_id,
    )


@router.post("/pool-clusters-batch", response_model=list[PooledClusterReport])
async def pool_evidence_clusters_batch(
    payload: list[EvidenceQualityRequest] | PooledClusterBatchRequest,
) -> list[PooledClusterReport]:
    """Pool redundant evidence items for a batch of claims with memory-safe chunking (Step 8.7)."""
    pooling_service = get_cluster_pooling_service()
    req_list = payload.requests if isinstance(payload, PooledClusterBatchRequest) else payload
    return pooling_service.pool_clusters_batch(req_list)


@router.post("/evaluate-debate-quality", response_model=DebateQualityResponse)
async def evaluate_debate_evidence_quality(
    request: DebateQualityRequest,
) -> DebateQualityResponse:
    """Evaluate dual-perspective debate evidence pools with independent and joint Kish n_eff (Step 8.8)."""
    quality_service = get_evidence_quality_service()
    return quality_service.evaluate_debate_quality(
        claim=request.claim_text,
        supporting_evidence=request.supporting_evidence,
        contradicting_evidence=request.contradicting_evidence,
        reference_date=request.reference_date,
        claim_origin_url=request.claim_origin_url,
        temporal_anchor=request.temporal_anchor,
        claim_id=request.claim_id,
        query_id=request.query_id,
    )


@router.post("/evaluate-debate-quality-batch", response_model=list[DebateQualityResponse])
async def evaluate_debate_evidence_quality_batch(
    payload: list[DebateQualityRequest] | DebateQualityBatchRequest,
) -> list[DebateQualityResponse]:
    """Evaluate dual-perspective debate evidence quality across a batch of claims with memory-safe chunking (Step 8.8)."""
    quality_service = get_evidence_quality_service()
    req_list = payload.requests if isinstance(payload, DebateQualityBatchRequest) else payload
    return quality_service.evaluate_debate_quality_batch(req_list)


