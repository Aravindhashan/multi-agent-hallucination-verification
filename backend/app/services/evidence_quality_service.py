"""Unified Evidence Quality and Defense Service (Research Module 4).

Orchestrates Source Credibility, Temporal Freshness, Source Independence,
and Evidence Pollution Defense into a cohesive evidence evaluation pipeline.
"""

from datetime import date, datetime, timezone
import math
from typing import Any, Sequence
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

from app.schemas import (
    DebateQualityRequest,
    DebateQualityResponse,
    EvidenceQualityRequest,
    EvidenceQualityResponse,
)
from app.services.evidence_pollution_detector import get_evidence_pollution_detector
from app.services.session_store import get_session_store
from app.services.source_credibility_scorer import get_source_credibility_scorer
from app.services.source_independence_config import DISCLAIMER_PHRASE as INDEPENDENCE_DISCLAIMER
from app.services.source_independence_scorer import (
    compute_kish_effective_sample_size,
    get_source_independence_detector,
)
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
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Normalized estimated relevance to the claim."
    )
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

    def _normalize_item_to_doc(
        self, item: Any, fallback_index: int, default_temporal_anchor: str | None = None
    ) -> dict[str, Any]:
        """Convert input evidence objects into uniform dictionary format."""
        if isinstance(item, dict):
            doc_id = str(item.get("evidence_id") or item.get("document_id") or item.get("id") or f"doc_{fallback_index}")
            text = str(item.get("text", "")).strip()
            source_url = item.get("source_url")
            domain = item.get("domain")
            source = item.get("source_title") or item.get("source", "unknown")
            pub_date = item.get("published_date")
            rel_score = float(item.get("relevance_score", 1.0))
            t_anchor = item.get("temporal_anchor") or (
                item.get("temporal_anchors")[0] if item.get("temporal_anchors") else default_temporal_anchor
            )
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
            rel_score = float(getattr(item, "relevance_score", 1.0))
            t_anchor = getattr(item, "temporal_anchor", None)
            if not t_anchor and hasattr(item, "temporal_anchors") and getattr(item, "temporal_anchors", None):
                t_anchor = getattr(item, "temporal_anchors")[0]
            if not t_anchor:
                t_anchor = default_temporal_anchor

        return {
            "document_id": doc_id,
            "id": doc_id,
            "evidence_id": doc_id,
            "text": text,
            "source_url": str(source_url) if source_url else None,
            "domain": str(domain) if domain else None,
            "source": str(source) if source else "unknown",
            "published_date": pub_date,
            "relevance_score": rel_score,
            "temporal_anchor": str(t_anchor) if t_anchor else None,
        }

    def evaluate_evidence(
        self,
        claim: str,
        evidence_items: Sequence[Any],
        reference_date: date | None = None,
        claim_origin_url: str | None = None,
        temporal_anchor: str | None = None,
        batch_size: int = 10,
    ) -> EvidenceQualityReport:
        """Evaluate a set of candidate evidence items against a factual claim."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

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

        # 1. Standardize documents with anchor awareness
        docs = [
            self._normalize_item_to_doc(it, idx, default_temporal_anchor=temporal_anchor)
            for idx, it in enumerate(evidence_items)
        ]

        # 2. Submodule 8.3: Source Independence & Redundancy complete-linkage clustering
        independence_res = self._independence_detector.analyze_set(docs, batch_size=batch_size)
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

        # 3. Submodule 8.4: Evidence Pollution & Circularity batch evaluation
        pollution_batch = self._pollution_detector.evaluate_batch(
            claim=normalized_claim,
            evidence_items=docs,
            claim_origin_url=claim_origin_url,
            batch_size=batch_size,
        )
        pollution_map = {res.document_id: res for res in pollution_batch.evaluations}

        # 4. Submodules 8.1 & 8.2: Credibility and Freshness batch evaluation
        cred_results = self._credibility_scorer.score_batch(docs, batch_size=batch_size)
        cred_map = {res.document_id: res for res in cred_results}

        fresh_results = self._freshness_scorer.score_batch(
            docs, reference_date=ref_date, temporal_anchor=temporal_anchor, batch_size=batch_size
        )
        fresh_map = {doc["document_id"]: res for doc, res in zip(docs, fresh_results)}

        # 5. Evaluate individual items (Credibility + Freshness + Pollution + Composite)
        evaluated_items: list[EvidenceQualityItemReport] = []
        cluster_weights: list[float] = []

        admissible_count = 0
        polluted_count = 0

        for doc in docs:
            d_id = doc["document_id"]

            cred_res = cred_map[d_id]
            fresh_res = fresh_map[d_id]

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
                    relevance_score=float(doc.get("relevance_score", 1.0)),
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

        # 6. Compute Kish Effective Sample Size (n_eff):
        # n_eff = M^2 / sum(s_k^2) over admissible clusters
        admissible_cluster_counts: dict[int, int] = {}
        for item in evaluated_items:
            if item.is_admissible:
                c_id = item.cluster_id
                admissible_cluster_counts[c_id] = admissible_cluster_counts.get(c_id, 0) + 1

        if admissible_cluster_counts:
            kish_n_eff = round(
                compute_kish_effective_sample_size(list(admissible_cluster_counts.values())),
                2,
            )
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

    def evaluate_claims_batch(
        self,
        requests: Sequence[EvidenceQualityRequest | dict[str, Any]],
        batch_size: int = 10,
    ) -> list[EvidenceQualityResponse]:
        """Evaluate a batch of claim-evidence quality requests with memory-safe chunking."""
        if not isinstance(requests, Sequence):
            raise ValueError(f"Expected Sequence for requests, got {type(requests).__name__}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not requests:
            return []

        results: list[EvidenceQualityResponse] = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            for req in chunk:
                if isinstance(req, dict):
                    claim_text = req.get("claim_text", "")
                    evidence_items = req.get("evidence_items", [])
                    claim_id = req.get("claim_id")
                    query_id = req.get("query_id")
                    ref_date = req.get("reference_date")
                    origin_url = req.get("claim_origin_url")
                    t_anchor = req.get("temporal_anchor")
                else:
                    claim_text = req.claim_text
                    evidence_items = req.evidence_items
                    claim_id = req.claim_id
                    query_id = req.query_id
                    ref_date = req.reference_date
                    origin_url = req.claim_origin_url
                    t_anchor = getattr(req, "temporal_anchor", None)

                report = self.evaluate_evidence(
                    claim=claim_text,
                    evidence_items=evidence_items,
                    reference_date=ref_date,
                    claim_origin_url=origin_url,
                    temporal_anchor=t_anchor,
                    batch_size=batch_size,
                )

                evaluated_dicts = [item.model_dump(mode="json") for item in report.evaluated_items]

                # Update session store if query_id is provided
                if query_id is not None:
                    session_store = get_session_store()
                    session_store.update_metadata(
                        query_id,
                        "evidence_quality_evaluation",
                        {
                            "total_evidence_count": report.total_evidence_count,
                            "admissible_count": report.admissible_count,
                            "kish_effective_count": report.kish_effective_count,
                            "cluster_count": report.cluster_count,
                        },
                    )
                    if claim_id is not None:
                        session_store.record_claim_quality_audit(
                            query_id=query_id,
                            claim_id=claim_id,
                            quality_data={
                                "total_evidence_count": report.total_evidence_count,
                                "admissible_count": report.admissible_count,
                                "polluted_count": report.polluted_count,
                                "cluster_count": report.cluster_count,
                                "kish_effective_count": report.kish_effective_count,
                            },
                        )

                results.append(
                    EvidenceQualityResponse(
                        claim_id=claim_id,
                        query_id=query_id,
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
                )

        return results

    def evaluate_debate_quality(
        self,
        claim: str,
        supporting_evidence: Sequence[Any],
        contradicting_evidence: Sequence[Any],
        reference_date: date | None = None,
        claim_origin_url: str | None = None,
        temporal_anchor: str | None = None,
        claim_id: UUID | str | None = None,
        query_id: UUID | None = None,
        batch_size: int = 10,
    ) -> DebateQualityResponse:
        """Evaluate dual-perspective debate evidence pools with independent and joint Kish n_eff (Step 8.8).

        Computes:
        1. Supporting quality & cluster pooling: e_hat_k,sup and n_eff,sup.
        2. Contradicting quality & cluster pooling: e_hat_k,con and n_eff,con.
        3. Cross-perspective joint analysis: n_eff,joint and overlap discount.
        4. Overall downstream NLI call reduction.
        5. SQLite session_claims audit persistence under 'debate_quality'.
        """
        from app.services.cluster_pooling import get_cluster_pooling_service

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        normalized_claim = claim.strip()
        ref_date = reference_date if reference_date is not None else date.today()
        pooling_service = get_cluster_pooling_service(quality_service=self)

        # 1. Pool supporting evidence
        sup_report = pooling_service.pool_clusters(
            claim=normalized_claim,
            evidence_items=supporting_evidence,
            reference_date=ref_date,
            claim_origin_url=claim_origin_url,
            temporal_anchor=temporal_anchor,
            batch_size=batch_size,
            claim_id=claim_id,
            query_id=query_id,
        )

        # 2. Pool contradicting evidence
        con_report = pooling_service.pool_clusters(
            claim=normalized_claim,
            evidence_items=contradicting_evidence,
            reference_date=ref_date,
            claim_origin_url=claim_origin_url,
            temporal_anchor=temporal_anchor,
            batch_size=batch_size,
            claim_id=claim_id,
            query_id=query_id,
        )

        n_eff_sup = sup_report.kish_effective_count
        n_eff_con = con_report.kish_effective_count

        # 3. Cross-perspective joint analysis across all evidence
        all_raw_evidence = list(supporting_evidence) + list(contradicting_evidence)
        if all_raw_evidence:
            joint_quality = self.evaluate_evidence(
                claim=normalized_claim,
                evidence_items=all_raw_evidence,
                reference_date=ref_date,
                claim_origin_url=claim_origin_url,
                temporal_anchor=temporal_anchor,
                batch_size=batch_size,
            )
            joint_n_eff = joint_quality.kish_effective_count
        else:
            joint_n_eff = 0.0

        # Overlap discount measures redundancy across the two perspectives
        sum_separate = n_eff_sup + n_eff_con
        if sum_separate > 0.0:
            cross_overlap_discount = round(
                max(0.0, min(1.0, (sum_separate - joint_n_eff) / sum_separate)), 4
            )
        else:
            cross_overlap_discount = 0.0

        total_raw = sup_report.total_raw_evidence_count + con_report.total_raw_evidence_count
        total_adm = sup_report.admissible_clusters + con_report.admissible_clusters
        total_pooled = len(sup_report.pooled_representatives) + len(con_report.pooled_representatives)

        if total_raw > 0:
            overall_nli_reduction = round(((total_raw - total_pooled) / total_raw) * 100.0, 2)
        else:
            overall_nli_reduction = 0.0

        response = DebateQualityResponse(
            claim_id=claim_id,
            query_id=query_id,
            claim_text=normalized_claim,
            supporting_report=sup_report,
            contradicting_report=con_report,
            supporting_n_eff=n_eff_sup,
            contradicting_n_eff=n_eff_con,
            joint_n_eff=joint_n_eff,
            cross_perspective_overlap_discount=cross_overlap_discount,
            total_raw_evidence_count=total_raw,
            total_admissible_count=total_adm,
            total_pooled_representatives_count=total_pooled,
            overall_nli_call_reduction_percent=overall_nli_reduction,
            pooled_supporting_evidence=sup_report.pooled_representatives,
            pooled_contradicting_evidence=con_report.pooled_representatives,
            disclaimer=INDEPENDENCE_DISCLAIMER,
        )

        # 4. SessionStore persistence under session_claims.metadata_json['debate_quality']
        if query_id is not None:
            session_store = get_session_store()
            session_store.update_metadata(
                query_id,
                "debate_quality_evaluation",
                {
                    "supporting_raw_count": sup_report.total_raw_evidence_count,
                    "supporting_n_eff": n_eff_sup,
                    "contradicting_raw_count": con_report.total_raw_evidence_count,
                    "contradicting_n_eff": n_eff_con,
                    "joint_n_eff": joint_n_eff,
                    "cross_perspective_overlap_discount": cross_overlap_discount,
                    "overall_nli_call_reduction_percent": overall_nli_reduction,
                },
            )
            if claim_id is not None:
                session_store.record_debate_quality_audit(
                    query_id=query_id,
                    claim_id=claim_id,
                    debate_quality_data={
                        "supporting_raw_count": sup_report.total_raw_evidence_count,
                        "supporting_admissible_count": sup_report.admissible_clusters,
                        "supporting_n_eff": n_eff_sup,
                        "contradicting_raw_count": con_report.total_raw_evidence_count,
                        "contradicting_admissible_count": con_report.admissible_clusters,
                        "contradicting_n_eff": n_eff_con,
                        "joint_n_eff": joint_n_eff,
                        "cross_perspective_overlap_discount": cross_overlap_discount,
                        "overall_nli_call_reduction_percent": overall_nli_reduction,
                    },
                )

        return response

    def evaluate_debate_quality_batch(
        self,
        requests: Sequence[DebateQualityRequest | dict[str, Any]],
        batch_size: int = 10,
    ) -> list[DebateQualityResponse]:
        """Evaluate dual-perspective debate evidence quality across a batch of claims."""
        if not isinstance(requests, Sequence):
            raise ValueError(f"Expected Sequence for requests, got {type(requests).__name__}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not requests:
            return []

        results: list[DebateQualityResponse] = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            for req in chunk:
                if isinstance(req, dict):
                    claim_text = str(req.get("claim_text", "")).strip()
                    sup_ev = req.get("supporting_evidence", [])
                    con_ev = req.get("contradicting_evidence", [])
                    ref_date = req.get("reference_date")
                    origin_url = req.get("claim_origin_url")
                    t_anchor = req.get("temporal_anchor")
                    c_id = req.get("claim_id")
                    q_id = req.get("query_id")
                else:
                    claim_text = req.claim_text.strip()
                    sup_ev = req.supporting_evidence
                    con_ev = req.contradicting_evidence
                    ref_date = req.reference_date
                    origin_url = req.claim_origin_url
                    t_anchor = req.temporal_anchor
                    c_id = req.claim_id
                    q_id = req.query_id

                res = self.evaluate_debate_quality(
                    claim=claim_text,
                    supporting_evidence=sup_ev,
                    contradicting_evidence=con_ev,
                    reference_date=ref_date,
                    claim_origin_url=origin_url,
                    temporal_anchor=t_anchor,
                    claim_id=c_id,
                    query_id=q_id,
                    batch_size=batch_size,
                )
                results.append(res)

        return results

    def evaluate_debate_claim_response(
        self,
        debate_response: Any,
        reference_date: date | None = None,
        claim_origin_url: str | None = None,
        temporal_anchor: str | None = None,
        batch_size: int = 10,
    ) -> DebateQualityResponse:
        """Convenience method directly evaluating the output of Module 7 Debate."""
        c_text = getattr(debate_response, "claim_text", "")
        sup_ev = getattr(debate_response, "final_supporting_evidence", [])
        con_ev = getattr(debate_response, "final_contradicting_evidence", [])
        c_id = getattr(debate_response, "claim_id", None)
        q_id = getattr(debate_response, "query_id", None)

        return self.evaluate_debate_quality(
            claim=c_text,
            supporting_evidence=sup_ev,
            contradicting_evidence=con_ev,
            reference_date=reference_date,
            claim_origin_url=claim_origin_url,
            temporal_anchor=temporal_anchor,
            claim_id=c_id,
            query_id=q_id,
            batch_size=batch_size,
        )



_global_quality_service: EvidenceQualityService | None = None


def get_evidence_quality_service() -> EvidenceQualityService:
    """Return the global singleton EvidenceQualityService."""
    global _global_quality_service
    if _global_quality_service is None:
        _global_quality_service = EvidenceQualityService()
    return _global_quality_service
