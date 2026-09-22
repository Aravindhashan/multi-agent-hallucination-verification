"""Service orchestration for extracting structured atomic claims from LLM text."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.schemas import Claim, ClaimExtractionResponse, DecompositionMetrics
from app.services.claim_extractor import BaseClaimExtractor
from app.services.claim_models import ClaimExtractionResult
from app.services.llm_client import BaseLLMClient
from app.services.llm_claim_extractor import LLMClaimExtractor
from app.services.nlp_claim_extractor import NLPClaimExtractor


def get_claim_extractor(
    method: str = "nlp", llm_client: BaseLLMClient | None = None
) -> BaseClaimExtractor:
    """Return the configured claim extractor strategy ('nlp' or 'llm')."""
    strategy = method.lower().strip()
    if strategy == "llm":
        return LLMClaimExtractor(llm_client=llm_client)
    return NLPClaimExtractor()


def extract_atomic_claims(
    response_text: str,
    query: str = "",
    method: str = "nlp",
    llm_client: BaseLLMClient | None = None,
) -> ClaimExtractionResult:
    """Extract structured AtomicClaim objects using the specified extraction method."""
    extractor = get_claim_extractor(method=method, llm_client=llm_client)
    return extractor.extract_atomic_claims(response_text=response_text, query=query)


def extract_claims(
    query_id: UUID,
    response_text: str,
    query: str = "",
    method: str = "nlp",
    llm_client: BaseLLMClient | None = None,
) -> ClaimExtractionResponse:
    """Extract, identify, and order claims for the API endpoint with full semantic qualifier preservation."""
    extraction_result = extract_atomic_claims(
        response_text=response_text,
        query=query,
        method=method,
        llm_client=llm_client,
    )

    claims = [
        Claim(
            claim_id=atomic.claim_id,
            text=atomic.claim_text,
            order=index,
            subject=atomic.subject,
            predicate=atomic.predicate,
            object=atomic.object,
            temporal_anchor=atomic.temporal_anchor,
            numerical_value=atomic.numerical_value,
            spatial_anchor=atomic.spatial_anchor,
            temporal_anchors=atomic.temporal_anchors,
            numerical_values=atomic.numerical_values,
            spatial_anchors=atomic.spatial_anchors,
            qualifiers=atomic.qualifiers,
            original_clause=atomic.original_clause,
            is_self_contained=atomic.is_self_contained,
        )
        for index, atomic in enumerate(extraction_result.claims)
    ]

    raw_metrics = getattr(extraction_result, "decomposition_metrics", None)
    decomp_metrics = None
    if raw_metrics is not None:
        if isinstance(raw_metrics, DecompositionMetrics):
            decomp_metrics = raw_metrics
        elif hasattr(raw_metrics, "model_dump"):
            decomp_metrics = DecompositionMetrics(**raw_metrics.model_dump())
        elif isinstance(raw_metrics, dict):
            decomp_metrics = DecompositionMetrics(**raw_metrics)

    return ClaimExtractionResponse(
        query_id=query_id,
        claims=claims,
        decomposition_metrics=decomp_metrics,
        extracted_at=extraction_result.extracted_at if getattr(extraction_result, "extracted_at", None) else datetime.now(timezone.utc),
    )
