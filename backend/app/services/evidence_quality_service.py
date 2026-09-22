"""Unified Evidence Quality and Defense Service (Research Module 4).

Orchestrates Source Credibility, Temporal Freshness, Source Independence,
and Evidence Pollution Defense into a cohesive evidence evaluation pipeline.
"""

from datetime import date, datetime, timezone
import math
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field

from app.services.evidence_pollution_detector import get_evidence_pollution_detector
from app.services.source_credibility_scorer import get_source_credibility_scorer
from app.services.source_independence_config import DISCLAIMER_PHRASE as INDEPENDENCE_DISCLAIMER
from app.services.source_independence_scorer import get_source_independence_detector
from app.services.temporal_freshness_scorer import get_temporal_freshness_scorer


class EvidenceQualityItemReport(BaseModel):
    """Detailed quality assessment for an individual evidence document."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="Unique identifier of the evidence item.")
    text: str = Field(description="Text snippet of the evidence.")
    credibility_score: float = Field(ge=0.0, le=1.0, description="Source credibility score.")
    domain_tier: int = Field(ge=1, le=4, description="Domain authority tier (1 to 4).")
    domain_tier_label: str = Field(description="Domain authority label.")
    freshness_score: float = Field(ge=0.0, le=1.0, description="Temporal freshness decay score.")
    age_days: int | None = Field(default=None, description="Age in days relative to reference date.")
    pollution_score: float = Field(ge=0.0, le=1.0, description="Evidence pollution score.")
    pollution_class: str = Field(description="Pollution classification category.")
    directional_grounding_score: float = Field(ge=0.0, le=1.0, description="Lexical grounding overlap.")
    cluster_id: int = Field(description="Complete-linkage redundancy cluster ID.")
    cluster_size: int = Field(ge=1, description="Number of items in this redundancy cluster.")
    composite_reliability_weight: float = Field(
        ge=0.0, le=1.0, description="Composite reliability weight accounting for all dimensions."
    )
    is_admissible: bool = Field(description="Whether evidence passes pollution and credibility gates.")
    admissibility_reason: str = Field(description="Explanation of admissibility decision.")


class EvidenceQualityReport(BaseModel):
    """Set-level quality and defense report across all candidate evidence items."""

    model_config = ConfigDict(frozen=True)

    claim_text: str = Field(description="The factual claim evaluated.")
    total_evidence_count: int = Field(ge=0, description="Total evidence passages evaluated.")
    admissible_count: int = Field(ge=0, description="Number of admissible passages.")
    polluted_count: int = Field(ge=0, description="Number of passages rejected due to pollution.")
    cluster_count: int = Field(ge=0, description="Number of redundancy clusters identified.")
    kish_effective_count: float = Field(
        ge=0.0, description="Kish effective sample size (n_eff) discounting redundancy."
    )
    evaluated_items: list[EvidenceQualityItemReport] = Field(
        description="Detailed quality reports for each evidence item."
    )
    disclaimer: str = Field(
        default=INDEPENDENCE_DISCLAIMER, description="Standard research heuristic boundary notice."
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when evaluation completed.",
    )


class EvidenceQualityService:
    """Orchestrates Submodules 4.1, 4.2, 4.3, and 4.4 into a unified evidence evaluation pipeline."""

    def __init__(self) -> None:
        self._credibility_scorer = get_source_credibility_scorer()
        self._freshness_scorer = get_temporal_freshness_scorer()
        self._independence_detector = get_source_independence_detector()
        self._pollution_detector = get_evidence_pollution_detector()

    def _normalize_item_to_doc(self, item: Any, fallback_index: int) -> dict[str, Any]:
        """Convert input evidence objects into uniform dictionary format."""
        if isinstance(item, dict):
            doc_id = str(item.get("evidence_id") or item.get("document_id") or item.get("id") or f"doc_{fallback_index}")
            text = str(item.get("text", "")).strip()
            source_url = item.get("source_url")
            domain = item.get("domain")
            source = item.get("source_title") or item.get("source", "unknown")
            pub_date = item.get("published_date")
        else:
            doc_id = str(
                getattr(item, "evidence_id", None)
                or getattr(item, "document_id", None)
                or getattr(item, "id", None)
                or f"doc_{fallback_index}"
            )
            text = str(getattr(item, "text", "")).strip()
            source_url = getattr(item, "source_url", None)
            domain = getattr(item, "domain", None)
            source = getattr(item, "source_title", None) or getattr(item, "source", None) or "unknown"
            pub_date = getattr(item, "published_date", None)

        return {
            "document_id": doc_id,
            "id": doc_id,
            "evidence_id": doc_id,
            "text": text,
            "source_url": str(source_url) if source_url else None,
            "domain": str(domain) if domain else None,
            "source": str(source) if source else "unknown",
            "published_date": pub_date,
        }

    def evaluate_evidence(
        self,
        claim: str,
        evidence_items: Sequence[Any],
        reference_date: date | None = None,
        claim_origin_url: str | None = None,
    ) -> EvidenceQualityReport:
        """Evaluate a set of candidate evidence items against a factual claim."""
        normalized_claim = claim.strip()
        ref_date = reference_date if reference_date is not None else date.today()

        if not evidence_items:
            return EvidenceQualityReport(
                claim_text=normalized_claim,
                total_evidence_count=0,
                admissible_count=0,
                polluted_count=0,
                cluster_count=0,
                kish_effective_count=0.0,
                evaluated_items=[],
            )

        # 1. Standardize documents
        docs = [self._normalize_item_to_doc(it, idx) for idx, it in enumerate(evidence_items)]

        # 2. Submodule 4.3: Source Independence & Redundancy complete-linkage clustering
        independence_res = self._independence_detector.analyze_set(docs)
        doc_to_cluster: dict[str, tuple[int, int]] = {}
        for cluster in independence_res.clusters:
            c_size = len(cluster.document_ids)
            for d_id in cluster.document_ids:
                doc_to_cluster[d_id] = (cluster.cluster_id, c_size)

        next_cluster_id = len(independence_res.clusters)
        for d_id in independence_res.independent_document_ids:
            if d_id not in doc_to_cluster:
                doc_to_cluster[d_id] = (next_cluster_id, 1)
                next_cluster_id += 1

        # 3. Submodule 4.4: Evidence Pollution & Circularity batch evaluation
        pollution_batch = self._pollution_detector.evaluate_batch(
            claim=normalized_claim,
            evidence_items=docs,
            claim_origin_url=claim_origin_url,
        )
        pollution_map = {res.document_id: res for res in pollution_batch.evaluations}

        # 4. Evaluate individual items (Credibility + Freshness + Pollution + Composite)
        evaluated_items: list[EvidenceQualityItemReport] = []
        cluster_weights: list[float] = []

        admissible_count = 0
        polluted_count = 0

        for doc in docs:
            d_id = doc["document_id"]

            # Submodule 4.1: Source Credibility
            cred_res = self._credibility_scorer.score(doc)

            # Submodule 4.2: Temporal Freshness
            fresh_res = self._freshness_scorer.score_document(doc, reference_date=ref_date)

            # Submodule 4.4: Pollution
            poll_res = pollution_map.get(d_id)
            p_score = poll_res.pollution_score if poll_res else 0.0
            p_class = poll_res.classification.value if poll_res else "CLEAN"
            p_grounding = poll_res.grounding_score if poll_res else 1.0

            # Admissibility gate: Reject if severe pollution or zero credibility
            cluster_id, cluster_size = doc_to_cluster.get(d_id, (0, 1))

            if p_score >= 0.80:
                is_admissible = False
                reason = f"Rejected: High evidence pollution ({p_class}, score={p_score:.2f})"
                polluted_count += 1
            elif cred_res.credibility_score < 0.20:
                is_admissible = False
                reason = f"Rejected: Low source credibility ({cred_res.credibility_score:.2f})"
            else:
                is_admissible = True
                reason = "Admissible: Passed credibility and pollution thresholds."
                admissible_count += 1

            # Composite weight calculation:
            # W_i = C_cred * F_fresh * (1 - P_poll) * (1 / cluster_size)
            redundancy_factor = 1.0 / float(cluster_size) if cluster_size > 0 else 1.0
            composite = cred_res.credibility_score * fresh_res.freshness_score * (1.0 - p_score) * redundancy_factor
            composite_weight = round(max(0.0, min(1.0, float(composite))), 4) if is_admissible else 0.0

            if is_admissible:
                cluster_weights.append(redundancy_factor)

            evaluated_items.append(
                EvidenceQualityItemReport(
                    evidence_id=d_id,
                    text=doc["text"],
                    credibility_score=cred_res.credibility_score,
                    domain_tier=cred_res.domain_tier,
                    domain_tier_label=cred_res.domain_tier_label,
                    freshness_score=fresh_res.freshness_score,
                    age_days=fresh_res.age_days,
                    pollution_score=p_score,
                    pollution_class=p_class,
                    directional_grounding_score=p_grounding,
                    cluster_id=cluster_id,
                    cluster_size=cluster_size,
                    composite_reliability_weight=composite_weight,
                    is_admissible=is_admissible,
                    admissibility_reason=reason,
                )
            )

        # 5. Compute Kish Effective Sample Size (n_eff):
        # n_eff = (sum w_i)^2 / sum (w_i^2)
        if cluster_weights:
            sum_w = sum(cluster_weights)
            sum_w_sq = sum(w * w for w in cluster_weights)
            kish_n_eff = round(float((sum_w ** 2) / sum_w_sq), 2) if sum_w_sq > 0.0 else 0.0
        else:
            kish_n_eff = 0.0

        return EvidenceQualityReport(
            claim_text=normalized_claim,
            total_evidence_count=len(docs),
            admissible_count=admissible_count,
            polluted_count=polluted_count,
            cluster_count=len(independence_res.clusters),
            kish_effective_count=kish_n_eff,
            evaluated_items=evaluated_items,
        )


_global_quality_service: EvidenceQualityService | None = None


def get_evidence_quality_service() -> EvidenceQualityService:
    """Return the global singleton EvidenceQualityService."""
    global _global_quality_service
    if _global_quality_service is None:
        _global_quality_service = EvidenceQualityService()
    return _global_quality_service
