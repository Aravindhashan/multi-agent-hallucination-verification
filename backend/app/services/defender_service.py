'''Service orchestration for the Defender stage of verification.'''

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.schemas import DefenderRequest, DefenderResponse, DefenderResult, PlanningAblationLog, EvidenceItem
from app.services.defender_agent import BaseDefender
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.simple_defender import SimpleDefender, PlanGuidedDefender
from app.services.session_store import get_session_store

def compute_token_overlap(q1: str, q2: str) -> float:
    """Compute token Jaccard similarity between two query strings."""
    t1 = set(q1.lower().split())
    t2 = set(q2.lower().split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))


def defend_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
) -> DefenderResponse:
    """Normalize one claim and gather evidence intended to support it.

    Uses injected retriever (defaults to get_evidence_retriever()).
    """

    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain non-whitespace characters.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    # Resolve retriever dependency
    if retriever is None:
        from app.dependencies import get_evidence_retriever
        retriever = get_evidence_retriever()

    # Load verification plan & anchors for this claim (if available)
    plan: list[str] = []
    anchors: dict[str, list[str]] = {"temporal": [], "numerical": [], "spatial": []}
    if query_id is not None:
        store = get_session_store()
        claim_record = store.get_claim_record(query_id, resolved_claim_id)
        if claim_record:
            plan = claim_record.get("verification_plan", [])
            # Anchor fields are stored as lists in the claim record
            anchors["temporal"] = claim_record.get("temporal_anchors", [])
            anchors["numerical"] = claim_record.get("numerical_values", [])
            anchors["spatial"] = claim_record.get("spatial_anchors", [])

    # Fallback anchor extraction using NLP claim extractor if none found
    if not any(anchors.values()):
        try:
            from app.services.nlp_claim_extractor import NLPClaimExtractor

            extractor = NLPClaimExtractor()
            parsed = extractor.extract_atomic_claims(normalized_claim_text)
            if parsed.claims:
                first_claim = parsed.claims[0]
                anchors["temporal"] = list(first_claim.temporal_anchors)
                anchors["numerical"] = list(first_claim.numerical_values)
                anchors["spatial"] = list(first_claim.spatial_anchors)
        except Exception:
            pass

    # Instantiate the plan-guided defender agent
    defender = PlanGuidedDefender(
        retriever=retriever,
        anchors=anchors,
        verification_plan=plan,
    )

    from app.config import settings

    raw_query = defender.build_query(normalized_claim_text)
    baseline_query = normalized_claim_text
    overlap = compute_token_overlap(raw_query, baseline_query)
    diversity_threshold = getattr(settings, "DEFENDER_QUERY_DIVERSITY_THRESHOLD", 0.85)
    is_exact_match = (raw_query.strip().lower() == baseline_query.strip().lower())
    is_near_duplicate = is_exact_match or (overlap >= diversity_threshold)

    if is_near_duplicate and not is_exact_match:
        # Near-duplicate query rejected in favor of baseline to avoid issuing redundant queries
        query_used = baseline_query
        difference_rationale = (
            f"Candidate dynamic query had high token overlap ({overlap:.2f} >= {diversity_threshold}) with baseline; "
            "rejected divergent query and defaulted to baseline to prevent near-duplicate retrieval."
        )
        empirical_lift_expected = "No lift expected (query rejected as near-duplicate of baseline)."
    elif is_exact_match:
        query_used = baseline_query
        difference_rationale = "Dynamic query defaulted to baseline claim text because no anchors or verification directives were present."
        empirical_lift_expected = "No lift expected (query identical to baseline)."
    else:
        query_used = raw_query
        difference_rationale = "Dynamic query incorporates extracted anchors and verification plan to narrow retrieval."
        empirical_lift_expected = "Higher precision for date/number/location‑heavy claims."

    # Log queries in PlanningAblationLog (Submodule 5.1)
    ablation_log = PlanningAblationLog(
        dynamic_strategy="Plan‑guided",
        dynamic_queries=[query_used],
        fixed_baseline_strategy="Baseline claim text",
        fixed_baseline_queries=[baseline_query],
        difference_rationale=difference_rationale,
        empirical_lift_expected=empirical_lift_expected,
    )

    # Persist ablation log into session_claims for this specific claim (Rank 3 fix: prevent sibling claim overwrites)
    if query_id is not None:
        try:
            store = get_session_store()
            if hasattr(store, "record_claim_ablation_log"):
                store.record_claim_ablation_log(
                    query_id=query_id,
                    claim_id=resolved_claim_id,
                    ablation_log=ablation_log.model_dump(),
                )
            if hasattr(store, "update_metadata"):
                store.update_metadata(
                    query_id=query_id,
                    key="defender_ablation_log",
                    value=ablation_log.model_dump(),
                )
        except Exception:
            pass

    # Retrieve evidence using the defender retriever with stance='supporting'
    raw_evidence = retriever.retrieve(query_used, stance="supporting")

    min_rel = getattr(settings, "DEFENDER_MIN_RELEVANCE", 0.2)
    w_rel = getattr(settings, "DEFENDER_RELEVANCE_WEIGHT", 0.5)
    w_cred = getattr(settings, "DEFENDER_CREDIBILITY_WEIGHT", 0.3)
    w_fresh = getattr(settings, "DEFENDER_FRESHNESS_WEIGHT", 0.2)

    # Apply credibility & freshness weighting and calculate composite score
    def _calculate_composite(item: EvidenceItem) -> float:
        rel = getattr(item, "relevance_score", None)
        if rel is None:
            rel = 1.0
        cred = getattr(item, "credibility_score", None)
        if cred is None:
            cred = 1.0
        fresh = getattr(item, "freshness_score", None)
        if fresh is None:
            fresh = 1.0
        return round(w_rel * rel + w_cred * cred + w_fresh * fresh, 4)

    # Apply threshold gate: discard items with relevance < min_rel or composite < min_rel
    filtered: list[EvidenceItem] = []
    for item in raw_evidence:
        item.composite_score = _calculate_composite(item)
        rel = getattr(item, "relevance_score", None)
        rel_val = 1.0 if rel is None else rel
        if rel_val >= min_rel and (item.composite_score is None or item.composite_score >= min_rel):
            filtered.append(item)

    ranked = sorted(filtered, key=lambda item: item.composite_score or 0.0, reverse=True)

    # Persist evidence JSON and metadata for this claim
    if query_id is not None:
        try:
            session_store = get_session_store()
            if hasattr(session_store, "record_claim_evidence"):
                session_store.record_claim_evidence(
                    query_id=query_id,
                    claim_id=resolved_claim_id,
                    evidence_type="supporting",
                    evidence=[e.model_dump(mode="json") for e in ranked],
                    query_used=query_used,
                    baseline_query=baseline_query,
                    raw_evidence_count=len(raw_evidence),
                    retained_evidence_count=len(ranked),
                    ablation_log=ablation_log.model_dump(),
                )
        except Exception:
            pass

    # Build internal result model
    result = DefenderResult(
        claim_text=normalized_claim_text,
        supporting_evidence=ranked,
        query_used=query_used,
        defended_at=datetime.now(timezone.utc),
    )

    # Return API response model
    return DefenderResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=result.claim_text,
        supporting_evidence=result.supporting_evidence,
        query_used=result.query_used,
        defended_at=result.defended_at,
    )


def defend_claims_batch(
    claims: list[dict[str, Any] | DefenderRequest | Any],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    batch_size: int = 5,
) -> list[DefenderResponse]:
    """Defend a list of claims in isolated memory-safe batches (max batch_size per slice).

    Respects RAM limits by batching requests (default max 5 claims per slice)
    and aggregating responses sequentially.
    """
    if not claims:
        return []

    responses: list[DefenderResponse] = []
    # Process in chunks of batch_size (default 5 to respect RAM limits)
    for i in range(0, len(claims), batch_size):
        batch_slice = claims[i : i + batch_size]
        for item in batch_slice:
            if isinstance(item, DefenderRequest):
                c_text = item.claim_text
                c_id = item.claim_id
                q_id = item.query_id or query_id
            elif isinstance(item, dict):
                c_text = item.get("claim_text") or item.get("text") or ""
                c_id = item.get("claim_id") or item.get("id")
                q_id = item.get("query_id") or query_id
            else:
                c_text = getattr(item, "claim_text", getattr(item, "text", ""))
                c_id = getattr(item, "claim_id", getattr(item, "id", None))
                q_id = getattr(item, "query_id", query_id)

            resp = defend_claim(
                claim_text=c_text,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
            )
            responses.append(resp)

    return responses
