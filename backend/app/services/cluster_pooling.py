"""Cluster-Level Log-Odds Pooling Service (Submodule 8.7).

Collapses complete-linkage redundancy clusters into highest-quality representatives
and computes reliability-weighted log-odds for downstream Module 9 (NLI Adjudication)
and Module 10 (Calibrated Verdict Engine).

Architectural Invariants & Reviewer §4 Reconciliation:
------------------------------------------------------
1. NO Corroboration Bonus: The arbitrary (1.0 + 0.10 * ln|C_k|) bonus is strictly
   dropped per Reviewer §4. Clusters of redundant news wire articles (J >= 0.85)
   represent mechanical syndication rather than independent epistemic corroboration.
   Boosting larger clusters would contradict Kish's n_eff sample-size discount.
2. Representative Selection: e_hat_k = argmax_{e in C_k, adm} S_comp(e), where
   S_comp(e) = Credibility(e) * Freshness(e) * (1.0 - Pollution(e)).
3. Cluster Reliability Weight: W(C_k) = S_comp(e_hat_k) in [0.0, 1.0].
   If all items in a cluster are rejected by pollution/credibility gates, W(C_k) = 0.0.
4. Downstream NLI Call Reduction: Downstream NLI evaluates only e_hat_k per admissible
   cluster, reducing inference volume from M to K_adm.
5. Log-Odds Primitives: Mathematically sound transformations between probability
   and log-odds, with epsilon-clamping and overflow protection, enabling Bayesian
   evidence aggregation.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import math
from typing import Any, Sequence
from uuid import UUID

from app.schemas import (
    EvidenceItem,
    EvidenceQualityRequest,
    PooledClusterReport,
    PooledEvidenceCluster,
)
from app.services.evidence_quality_service import (
    EvidenceQualityItemReport,
    EvidenceQualityReport,
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.session_store import get_session_store
from app.services.source_independence_config import DISCLAIMER_PHRASE

logger = logging.getLogger("hallucination_verifier.cluster_pooling")


# ============================================================================
# 1. Log-Odds & Probability Primitives
# ============================================================================


def prob_to_log_odds(p: float, eps: float = 1e-6) -> float:
    """Convert probability p in [0.0, 1.0] to log-odds ln(p / (1 - p)).

    Args:
        p: Probability value in [0.0, 1.0].
        eps: Epsilon clamping threshold to avoid log(0) or division by zero.

    Returns:
        Log-odds value in float.

    Raises:
        ValueError: If p is not a finite float or outside [0.0, 1.0].
    """
    if not isinstance(p, (int, float)) or isinstance(p, bool) or math.isnan(p) or math.isinf(p):
        raise ValueError(f"Expected finite numeric probability, got {p}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Probability must be in [0.0, 1.0], got {p}")

    p_clamped = min(max(float(p), eps), 1.0 - eps)
    return math.log(p_clamped / (1.0 - p_clamped))


def log_odds_to_prob(l: float, eps: float = 1e-6) -> float:
    """Convert log-odds l to probability via sigmoid function 1 / (1 + exp(-l)).

    Clamps extreme log-odds to [-35.0, 35.0] to completely avoid OverflowError.

    Args:
        l: Log-odds value.
        eps: Small epsilon bound to ensure result strictly in [0.0, 1.0].

    Returns:
        Probability value in [0.0, 1.0].

    Raises:
        ValueError: If l is not a finite float.
    """
    if not isinstance(l, (int, float)) or isinstance(l, bool) or math.isnan(l) or math.isinf(l):
        raise ValueError(f"Expected finite numeric log-odds, got {l}")

    val = float(l)
    if val >= 35.0:
        return 1.0 - eps
    if val <= -35.0:
        return eps

    return 1.0 / (1.0 + math.exp(-val))


def combine_log_odds(
    log_odds_list: Sequence[float],
    weights: Sequence[float],
    prior_log_odds: float = 0.0,
) -> float:
    """Combine multiple independent log-odds updates using cluster reliability weights.

    Bayesian update rule for independent evidence:
    L_posterior = L_0 + sum_{k=1}^K w_k * (L_k - L_0)

    When prior is uninformative (50/50 prior, L_0 = 0.0):
    L_posterior = sum_{k=1}^K w_k * L_k

    Args:
        log_odds_list: Sequence of log-odds updates from independent clusters.
        weights: Cluster reliability weights W(C_k) in [0.0, 1.0].
        prior_log_odds: Baseline prior log-odds L_0 (default 0.0 for 50/50 prior).

    Returns:
        Combined posterior log-odds.

    Raises:
        ValueError: If lengths do not match or weights are negative.
    """
    if len(log_odds_list) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(log_odds_list)} log-odds vs {len(weights)} weights."
        )

    for idx, w in enumerate(weights):
        if not isinstance(w, (int, float)) or isinstance(w, bool) or math.isnan(w) or w < 0.0:
            raise ValueError(f"Weight at index {idx} must be non-negative finite numeric, got {w}")

    for idx, lo in enumerate(log_odds_list):
        if not isinstance(lo, (int, float)) or isinstance(lo, bool) or math.isnan(lo) or math.isinf(lo):
            raise ValueError(f"Log-odds at index {idx} must be finite numeric, got {lo}")

    if not log_odds_list:
        return float(prior_log_odds)

    total_update = sum(
        float(w) * (float(lo) - float(prior_log_odds))
        for lo, w in zip(log_odds_list, weights)
    )
    combined = float(prior_log_odds) + total_update

    # Bound extreme output to [-50.0, 50.0] for downstream safety
    return max(-50.0, min(50.0, combined))


def combine_nli_probabilities(
    probs_list: Sequence[dict[str, float] | tuple[float, float, float]],
    weights: Sequence[float],
    prior_prob_support: float = 0.5,
) -> dict[str, Any]:
    """Combine cluster-level NLI entailment and contradiction distributions with weights.

    Args:
        probs_list: Sequence of (p_ent, p_con, p_neu) or {"entailment": p, "contradiction": p}.
        weights: Cluster reliability weights W(C_k) in [0.0, 1.0].
        prior_prob_support: Prior probability of claim support (default 0.5).

    Returns:
        Dictionary with combined support log-odds, contradiction log-odds,
        posterior probabilities, and net evidence direction.
    """
    if len(probs_list) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(probs_list)} distributions vs {len(weights)} weights."
        )

    prior_lo_sup = prob_to_log_odds(prior_prob_support)
    prior_lo_con = prob_to_log_odds(1.0 - prior_prob_support)

    lo_sup_list: list[float] = []
    lo_con_list: list[float] = []
    effective_weights: list[float] = []

    for dist, w in zip(probs_list, weights):
        if isinstance(dist, dict):
            p_ent = float(dist.get("entailment", dist.get("p_ent", 0.0)))
            p_con = float(dist.get("contradiction", dist.get("p_con", 0.0)))
            p_neu = float(dist.get("neutral", dist.get("p_neu", 0.0)))
        elif isinstance(dist, (tuple, list)) and len(dist) >= 2:
            p_ent = float(dist[0])
            p_con = float(dist[1])
            p_neu = float(dist[2]) if len(dist) >= 3 else 0.0
        else:
            raise ValueError(f"Unsupported NLI probability format: {dist}")

        # Informativeness gating: a neutral document provides zero Bayes factor evidence
        # for or against the claim. Scale effective weight by (1 - p_neu).
        inf_scale = max(0.01, min(1.0, 1.0 - p_neu)) if p_neu > 0.0 else 1.0
        effective_weights.append(w * inf_scale)

        lo_sup_list.append(prob_to_log_odds(p_ent))
        lo_con_list.append(prob_to_log_odds(p_con))

    combined_lo_sup = combine_log_odds(lo_sup_list, effective_weights, prior_log_odds=prior_lo_sup)
    combined_lo_con = combine_log_odds(lo_con_list, effective_weights, prior_log_odds=prior_lo_con)

    post_p_sup = log_odds_to_prob(combined_lo_sup)
    post_p_con = log_odds_to_prob(combined_lo_con)

    # Determine net evidence stance
    if post_p_sup > 0.55 and post_p_sup > post_p_con:
        direction = "SUPPORTING"
    elif post_p_con > 0.55 and post_p_con > post_p_sup:
        direction = "CONTRADICTING"
    else:
        direction = "NEUTRAL"

    return {
        "combined_log_odds_support": round(combined_lo_sup, 4),
        "combined_log_odds_contradiction": round(combined_lo_con, 4),
        "posterior_support_probability": round(post_p_sup, 4),
        "posterior_contradiction_probability": round(post_p_con, 4),
        "net_direction": direction,
    }


# ============================================================================
# 2. Intrinsic Composite Scoring & Cluster Representative Selection
# ============================================================================


def compute_intrinsic_composite_score(
    credibility_score: float,
    freshness_score: float,
    pollution_score: float,
) -> float:
    """Compute intrinsic epistemic channel quality score S_comp(e) of an evidence item.

    Architectural Formula Reconciliation (Reviewer Assessment §3):
    --------------------------------------------------------------
    The pipeline utilizes two intentionally distinct composite scoring formulations
    serving two fundamentally different statistical objectives:

    1. Retrieval Ranking Heuristic (Modules 5.3, 6.3, 7.1, 7.2):
       S_retrieval = 0.5 * relevance + 0.3 * credibility + 0.2 * freshness
       - Objective: Open-corpus candidate selection.
       - Role: Relevance dominates (0.50) to ensure retrieved candidates address
         the specific search query. Linear additive form prevents edge cases from
         prematurely excluding documents from downstream debate consideration.

    2. Bayesian Epistemic Channel Reliability W(C_k) (Modules 8.7, 9, 10):
       W(C_k) = S_intrinsic(e_hat_k) = Credibility * Freshness * (1.0 - Pollution)
       - Objective: Channel reliability weighting for Bayesian log-odds updates:
         L_comb = L_0 + sum_k W(C_k) * (L_k - L_0).
       - Deliberate Omission of Relevance: In Module 9, DeBERTa NLI cross-encoder
         directly evaluates the representative passage against the claim to produce
         the log-likelihood update L_k. Multiplying W(C_k) by lexical retrieval
         relevance would double-count semantic relevance (once crudely via BM25/Dense
         query keywords, and a second time via cross-encoder NLI).
       - Multiplicative Formulation Rationale: In epistemic evidence aggregation,
         channel reliability is an AND condition: a completely fabricated source
         (Credibility = 0) or circular hallucination loop (Pollution = 1.0) possesses
         ZERO channel reliability, regardless of freshness. Multiplicative composition
         guarantees that ungrounded or fabricated items cannot update belief odds.
       - Historical Safeguard: Submodule 8.2 upper-envelope guarantees historical
         scholarship never zeroes out (historical half-life = 3650d, F >= 0.10).

    Args:
        credibility_score: Source credibility score in [0.0, 1.0].
        freshness_score: Temporal freshness score in [0.0, 1.0].
        pollution_score: Evidence pollution score in [0.0, 1.0].

    Returns:
        Intrinsic composite score rounded to 4 decimals in [0.0, 1.0].
    """
    c = max(0.0, min(1.0, float(credibility_score)))
    f = max(0.0, min(1.0, float(freshness_score)))
    p = max(0.0, min(1.0, float(pollution_score)))

    raw_comp = c * f * (1.0 - p)
    return round(max(0.0, min(1.0, raw_comp)), 4)


def select_cluster_representative(
    cluster_items: Sequence[EvidenceQualityItemReport | dict[str, Any]],
) -> tuple[EvidenceQualityItemReport | dict[str, Any], float, bool, str]:
    """Select highest-quality representative e_hat_k and compute W(C_k).

    Architectural Invariant (Reviewer §4 Reconciliation):
    -----------------------------------------------------
    The cluster reliability weight is strictly:
    W(C_k) = S_comp(e_hat_k) in [0.0, 1.0]

    No arbitrary (1.0 + 0.10 * ln|C_k|) corroboration bonus is applied.
    Redundancy discounting is handled cleanly by Kish's n_eff at the set level.

    Deterministic Tie-Breaking Sort Key:
    1. Admissibility priority (admissible items strictly preferred).
    2. S_comp(e) descending (intrinsic channel reliability).
    3. Credibility score descending.
    4. Freshness score descending.
    5. Pollution score ascending (1.0 - p descending).
    6. Retrieval relevance score descending (prefers most query-relevant copy).
    7. Lexicographical evidence_id ascending for 100% determinism.

    Args:
        cluster_items: All candidate evidence items belonging to cluster C_k.

    Returns:
        Tuple of (representative_item, cluster_reliability_weight, is_admissible, reason).
    """
    if not cluster_items:
        raise ValueError("Cannot select representative from empty cluster.")

    def _get_field(item: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field_name, default)
        return getattr(item, field_name, default)

    # Inspect admissibility
    admissible_items: list[Any] = []
    for it in cluster_items:
        is_adm = bool(_get_field(it, "is_admissible", True))
        cred = float(_get_field(it, "credibility_score", 1.0))
        poll = float(_get_field(it, "pollution_score", 0.0))
        # Enforce gates
        if is_adm and poll < 0.80 and cred >= 0.20:
            admissible_items.append(it)

    candidate_pool = admissible_items if admissible_items else list(cluster_items)
    is_cluster_admissible = len(admissible_items) > 0

    def _sort_key(item: Any) -> tuple[int, float, float, float, float, float, str]:
        item_is_adm = 1 if item in admissible_items else 0
        c = float(_get_field(item, "credibility_score", 1.0))
        f = float(_get_field(item, "freshness_score", 1.0))
        p = float(_get_field(item, "pollution_score", 0.0))
        rel = float(_get_field(item, "relevance_score", 1.0))
        s_comp = compute_intrinsic_composite_score(c, f, p)
        e_id = str(_get_field(item, "evidence_id", _get_field(item, "document_id", "")))

        return (
            item_is_adm,
            round(s_comp, 6),
            round(c, 6),
            round(f, 6),
            round(1.0 - p, 6),
            round(rel, 6),  # Relevance tie-breaker before lexicographical fallback
            e_id,  # lexicographical tie-break
        )

    sorted_candidates = sorted(candidate_pool, key=_sort_key)
    representative = sorted_candidates[-1]

    rep_cred = float(_get_field(representative, "credibility_score", 1.0))
    rep_fresh = float(_get_field(representative, "freshness_score", 1.0))
    rep_poll = float(_get_field(representative, "pollution_score", 0.0))
    s_comp_hat = compute_intrinsic_composite_score(rep_cred, rep_fresh, rep_poll)

    if is_cluster_admissible:
        # Reviewer §4 Reconciliation: W(C_k) = S_comp(e_hat_k) without log-bonus!
        cluster_weight = round(s_comp_hat, 4)
        reason = "Admissible cluster: representative selected by argmax S_comp."
    else:
        # Fully rejected cluster receives 0.0 weight
        cluster_weight = 0.0
        reason = "Rejected: All cluster members failed admissibility gates."

    return representative, cluster_weight, is_cluster_admissible, reason


# ============================================================================
# 3. Cluster Pooling Service
# ============================================================================


class ClusterPoolingService:
    """Service that collapses redundancy clusters into highest-quality representatives.

    Prepares clean, deduplicated, reliability-weighted evidence for downstream
    Module 9 NLI inference and Module 10 Bayesian verdict adjudication.
    """

    def __init__(self, quality_service: EvidenceQualityService | None = None) -> None:
        """Initialize ClusterPoolingService with injected or singleton EvidenceQualityService."""
        self._quality_service = (
            quality_service if quality_service is not None else get_evidence_quality_service()
        )

    def pool_from_quality_report(
        self,
        quality_report: EvidenceQualityReport,
        original_items: Sequence[Any] | None = None,
        claim_id: UUID | str | None = None,
        query_id: UUID | None = None,
    ) -> PooledClusterReport:
        """Pool clusters directly from an existing EvidenceQualityReport.

        Args:
            quality_report: Pre-evaluated EvidenceQualityReport from Submodule 8.5.
            original_items: Optional original EvidenceItems to copy rich attributes.
            claim_id: Optional claim identifier.
            query_id: Optional pipeline query identifier.

        Returns:
            PooledClusterReport containing collapsed clusters and winning representatives.
        """
        if quality_report.total_evidence_count == 0 or not quality_report.evaluated_items:
            return PooledClusterReport(
                claim_text=quality_report.claim_text,
                claim_id=claim_id,
                query_id=query_id,
                total_raw_evidence_count=0,
                total_clusters=0,
                admissible_clusters=0,
                polluted_clusters=0,
                kish_effective_count=0.0,
                nli_call_reduction_percent=0.0,
                clusters=[],
                pooled_representatives=[],
                disclaimer=quality_report.disclaimer or DISCLAIMER_PHRASE,
            )

        # Build lookup for original items
        orig_lookup: dict[str, Any] = {}
        if original_items:
            for it in original_items:
                i_id = str(
                    getattr(it, "evidence_id", None)
                    or (it.get("evidence_id") if isinstance(it, dict) else None)
                    or getattr(it, "document_id", None)
                    or (it.get("document_id") if isinstance(it, dict) else None)
                    or getattr(it, "id", None)
                    or (it.get("id") if isinstance(it, dict) else None)
                    or ""
                )
                if i_id:
                    orig_lookup[i_id] = it

        # Group evaluated items by cluster_id
        cluster_groups: dict[int, list[EvidenceQualityItemReport]] = {}
        for item in quality_report.evaluated_items:
            c_id = item.cluster_id
            cluster_groups.setdefault(c_id, []).append(item)

        pooled_clusters: list[PooledEvidenceCluster] = []
        pooled_representatives: list[EvidenceItem] = []

        # Sort cluster IDs for 100% deterministic output order
        for c_id in sorted(cluster_groups.keys()):
            c_members = cluster_groups[c_id]
            rep_item, cluster_w, is_adm, reason = select_cluster_representative(c_members)

            rep_id = str(rep_item.evidence_id)
            orig_obj = orig_lookup.get(rep_id)

            # Resolve domain and URL from original or report
            domain = getattr(orig_obj, "domain", None) if orig_obj else None
            source_url = getattr(orig_obj, "source_url", None) if orig_obj else None
            source_title = getattr(orig_obj, "source_title", None) if orig_obj else None
            if isinstance(orig_obj, dict):
                domain = orig_obj.get("domain", domain)
                source_url = orig_obj.get("source_url", source_url)
                source_title = orig_obj.get("source_title", source_title)

            # Create structured EvidenceItem for downstream NLI
            evidence_item = EvidenceItem(
                evidence_id=rep_id,
                text=rep_item.text,
                source_url=str(source_url) if source_url else None,
                domain=str(domain) if domain else None,
                source_title=str(source_title) if source_title else None,
                retrieval_method="cluster_pooled",
                relevance_score=1.0,
                credibility_score=rep_item.credibility_score,
                freshness_score=rep_item.freshness_score,
                composite_score=round(0.5 * 1.0 + 0.3 * rep_item.credibility_score + 0.2 * rep_item.freshness_score, 4),
                cluster_reliability_weight=cluster_w,
            )

            pooled_cluster = PooledEvidenceCluster(
                cluster_id=c_id,
                representative_id=rep_id,
                representative_text=rep_item.text,
                cluster_size=len(c_members),
                member_ids=[str(m.evidence_id) for m in c_members],
                cluster_reliability_weight=cluster_w,
                is_admissible=is_adm,
                admissibility_reason=reason,
                domain=str(domain) if domain else None,
                source_url=str(source_url) if source_url else None,
                credibility_score=rep_item.credibility_score,
                freshness_score=rep_item.freshness_score,
                pollution_score=rep_item.pollution_score,
                pollution_class=rep_item.pollution_class,
                directional_grounding_score=rep_item.directional_grounding_score,
                representative_item=evidence_item,
            )
            pooled_clusters.append(pooled_cluster)

            if is_adm:
                pooled_representatives.append(evidence_item)

        total_raw = quality_report.total_evidence_count
        total_clusters = len(pooled_clusters)
        admissible_clusters = sum(1 for c in pooled_clusters if c.is_admissible)
        polluted_clusters = total_clusters - admissible_clusters

        # NLI call reduction: (M - K_adm) / M * 100%
        if total_raw > 0:
            nli_reduction = round(((total_raw - admissible_clusters) / total_raw) * 100.0, 2)
        else:
            nli_reduction = 0.0

        report = PooledClusterReport(
            claim_text=quality_report.claim_text,
            claim_id=claim_id,
            query_id=query_id,
            total_raw_evidence_count=total_raw,
            total_clusters=total_clusters,
            admissible_clusters=admissible_clusters,
            polluted_clusters=polluted_clusters,
            kish_effective_count=quality_report.kish_effective_count,
            nli_call_reduction_percent=nli_reduction,
            clusters=pooled_clusters,
            pooled_representatives=pooled_representatives,
            disclaimer=quality_report.disclaimer or DISCLAIMER_PHRASE,
        )

        # Audit persistence if session IDs provided
        if query_id is not None and claim_id is not None:
            try:
                session_store = get_session_store()
                session_store.record_cluster_pooling_audit(
                    query_id=query_id,
                    claim_id=claim_id,
                    pooling_data={
                        "total_raw_evidence_count": total_raw,
                        "total_clusters": total_clusters,
                        "admissible_clusters": admissible_clusters,
                        "polluted_clusters": polluted_clusters,
                        "kish_effective_count": quality_report.kish_effective_count,
                        "nli_call_reduction_percent": nli_reduction,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record cluster pooling audit for query_id=%s, claim_id=%s: %s",
                    query_id,
                    claim_id,
                    exc,
                )

        return report

    def pool_clusters(
        self,
        claim: str,
        evidence_items: Sequence[Any],
        reference_date: date | None = None,
        claim_origin_url: str | None = None,
        temporal_anchor: str | None = None,
        batch_size: int = 10,
        claim_id: UUID | str | None = None,
        query_id: UUID | None = None,
    ) -> PooledClusterReport:
        """Evaluate evidence through Module 8 quality pipeline and pool clusters.

        Args:
            claim: The factual claim string.
            evidence_items: Candidate evidence passages.
            reference_date: Reference date for freshness decay.
            claim_origin_url: Optional origin URL of the claim.
            temporal_anchor: Optional historical anchor year/event.
            batch_size: Memory-safe chunk size for sub-services (default 10).
            claim_id: Optional claim identifier.
            query_id: Optional pipeline query identifier.

        Returns:
            PooledClusterReport containing collapsed clusters and representatives.
        """
        quality_report = self._quality_service.evaluate_evidence(
            claim=claim,
            evidence_items=evidence_items,
            reference_date=reference_date,
            claim_origin_url=claim_origin_url,
            temporal_anchor=temporal_anchor,
            batch_size=batch_size,
        )
        return self.pool_from_quality_report(
            quality_report=quality_report,
            original_items=evidence_items,
            claim_id=claim_id,
            query_id=query_id,
        )

    def pool_clusters_batch(
        self,
        requests: Sequence[EvidenceQualityRequest | dict[str, Any]],
        batch_size: int = 10,
    ) -> list[PooledClusterReport]:
        """Pool clusters across a batch of claims with memory-safe chunking.

        Enforces strict chunking: processes in blocks of <= batch_size items.
        Preserves 1-to-1 input sequence order.

        Args:
            requests: Sequence of EvidenceQualityRequest or dicts.
            batch_size: Maximum chunk size (default 10).

        Returns:
            List of PooledClusterReport matching input sequence order.

        Raises:
            ValueError: If batch_size < 1 or not an integer.
        """
        if not isinstance(requests, Sequence):
            raise ValueError(f"Expected Sequence for requests, got {type(requests).__name__}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not requests:
            return []

        results: list[PooledClusterReport] = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            for req in chunk:
                if isinstance(req, dict):
                    claim_text = str(req.get("claim_text", "")).strip()
                    evidence_items = req.get("evidence_items", [])
                    ref_date = req.get("reference_date")
                    origin_url = req.get("claim_origin_url")
                    t_anchor = req.get("temporal_anchor")
                    c_id = req.get("claim_id")
                    q_id = req.get("query_id")
                else:
                    claim_text = req.claim_text.strip()
                    evidence_items = req.evidence_items
                    ref_date = req.reference_date
                    origin_url = req.claim_origin_url
                    t_anchor = req.temporal_anchor
                    c_id = req.claim_id
                    q_id = req.query_id

                pooled_res = self.pool_clusters(
                    claim=claim_text,
                    evidence_items=evidence_items,
                    reference_date=ref_date,
                    claim_origin_url=origin_url,
                    temporal_anchor=t_anchor,
                    batch_size=batch_size,
                    claim_id=c_id,
                    query_id=q_id,
                )
                results.append(pooled_res)

        return results


# Module-level singleton
_cluster_pooling_service: ClusterPoolingService | None = None


def get_cluster_pooling_service(
    quality_service: EvidenceQualityService | None = None,
) -> ClusterPoolingService:
    """Retrieve singleton instance of ClusterPoolingService."""
    global _cluster_pooling_service
    if _cluster_pooling_service is None or quality_service is not None:
        _cluster_pooling_service = ClusterPoolingService(quality_service=quality_service)
    return _cluster_pooling_service
