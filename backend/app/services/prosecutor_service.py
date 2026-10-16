"""Service orchestration for the Prosecutor stage of verification."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.schemas import (
    EvidenceItem,
    PlanningAblationLog,
    ProsecutorRequest,
    ProsecutorResponse,
    ProsecutorResult,
    StructuredChallenge,
)
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_agent import BaseProsecutor
from app.services.session_store import get_session_store
from app.services.simple_prosecutor import AdvCounterQueryProsecutor, SimpleProsecutor

REFUTATION_CUES = [
    "refuted", "disputed", "false", "misconception", "denied",
    "retracted", "never happened", "debunked", "untrue", "inaccurate",
    "contradicts", "conflict", "hoax", "myth", "incorrect",
]


def extract_structured_challenges(
    claim_text: str,
    evidence_items: list[EvidenceItem],
    anchors: dict[str, list[str]],
) -> list[StructuredChallenge]:
    """Identify structured factual challenges comparing claim anchors against retrieved evidence.

    Strictly structured — extracts anchor conflicts and dispute cues without conversational prose.
    """
    challenges: list[StructuredChallenge] = []

    for item in evidence_items:
        text_lower = item.text.lower()
        found_cues = [cue for cue in REFUTATION_CUES if cue in text_lower]

        # 1. Temporal anchor conflict
        matched_conflict = False
        for t_anchor in anchors.get("temporal", []):
            if found_cues or (t_anchor.lower() not in text_lower):
                ch = StructuredChallenge(
                    conflict_type="temporal",
                    claim_anchor=t_anchor,
                    evidence_snippet=item.text[:200],
                    refutation_cues=found_cues,
                    confidence_score=round(min(1.0, (item.composite_score or 0.8) + (0.05 if found_cues else 0.0)), 4),
                )
                item.challenge = ch
                challenges.append(ch)
                matched_conflict = True
                break

        if matched_conflict:
            continue

        # 2. Numerical anchor conflict
        for n_val in anchors.get("numerical", []):
            if found_cues or (n_val not in item.text):
                ch = StructuredChallenge(
                    conflict_type="numerical",
                    claim_anchor=n_val,
                    evidence_snippet=item.text[:200],
                    refutation_cues=found_cues,
                    confidence_score=round(min(1.0, (item.composite_score or 0.8) + (0.05 if found_cues else 0.0)), 4),
                )
                item.challenge = ch
                challenges.append(ch)
                matched_conflict = True
                break

        if matched_conflict:
            continue

        # 3. Spatial anchor conflict
        for s_anchor in anchors.get("spatial", []):
            if found_cues or (s_anchor.lower() not in text_lower):
                ch = StructuredChallenge(
                    conflict_type="spatial",
                    claim_anchor=s_anchor,
                    evidence_snippet=item.text[:200],
                    refutation_cues=found_cues,
                    confidence_score=round(min(1.0, (item.composite_score or 0.8) + (0.05 if found_cues else 0.0)), 4),
                )
                item.challenge = ch
                challenges.append(ch)
                matched_conflict = True
                break

        if matched_conflict:
            continue

        # 4. Refutation cue conflict or general predicate challenge
        if found_cues:
            ch = StructuredChallenge(
                conflict_type="refutation",
                claim_anchor=claim_text[:80],
                evidence_snippet=item.text[:200],
                refutation_cues=found_cues,
                confidence_score=round(item.composite_score or 0.8, 4),
            )
            item.challenge = ch
            challenges.append(ch)
        elif item.composite_score and item.composite_score >= 0.2:
            ch = StructuredChallenge(
                conflict_type="predicate",
                claim_anchor=claim_text[:80],
                evidence_snippet=item.text[:200],
                refutation_cues=["contradiction_stance"],
                confidence_score=round(item.composite_score, 4),
            )
            item.challenge = ch
            challenges.append(ch)

    return challenges



def compute_token_overlap(q1: str, q2: str) -> float:
    """Compute token Jaccard similarity between two query strings."""
    t1 = set(q1.lower().split())
    t2 = set(q2.lower().split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))


def prosecute_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
) -> ProsecutorResponse:
    """Normalize one claim and gather evidence intended to challenge it.

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

    # Load verification plan & anchors for this claim (if available in session store)
    plan: list[str] = []
    anchors: dict[str, list[str]] = {"temporal": [], "numerical": [], "spatial": []}
    if query_id is not None:
        store = get_session_store()
        claim_record = store.get_claim_record(query_id, resolved_claim_id)
        if claim_record:
            plan = claim_record.get("verification_plan", [])
            anchors["temporal"] = claim_record.get("temporal_anchors", [])
            anchors["numerical"] = claim_record.get("numerical_values", [])
            anchors["spatial"] = claim_record.get("spatial_anchors", [])

    # Fallback anchor & SPO extraction using NLP claim extractor
    spo: dict[str, str | None] = {"subject": None, "predicate": None, "object": None}
    try:
        from app.services.nlp_claim_extractor import NLPClaimExtractor

        extractor = NLPClaimExtractor()
        parsed = extractor.extract_atomic_claims(normalized_claim_text)
        if parsed.claims:
            first_claim = parsed.claims[0]
            if not any(anchors.values()):
                anchors["temporal"] = list(first_claim.temporal_anchors)
                anchors["numerical"] = list(first_claim.numerical_values)
                anchors["spatial"] = list(first_claim.spatial_anchors)
            spo["subject"] = first_claim.subject
            spo["predicate"] = first_claim.predicate
            spo["object"] = first_claim.object
    except Exception:
        pass

    # Instantiate the adversarial counter-query prosecutor agent (Pillar 2)
    prosecutor = AdvCounterQueryProsecutor(
        retriever=retriever,
        anchors=anchors,
        spo=spo,
        verification_plan=plan,
    )

    from app.config import settings

    raw_query = prosecutor.build_counter_query(normalized_claim_text)
    baseline_query = normalized_claim_text
    overlap = compute_token_overlap(raw_query, baseline_query)
    diversity_threshold = getattr(settings, "PROSECUTOR_QUERY_DIVERSITY_THRESHOLD", 0.85)
    is_exact_match = (raw_query.strip().lower() == baseline_query.strip().lower())
    is_near_duplicate = is_exact_match or (overlap >= diversity_threshold)

    if is_near_duplicate and not is_exact_match:
        # Near-duplicate query rejected in favor of baseline to avoid issuing redundant queries
        query_used = baseline_query
        difference_rationale = (
            f"Candidate dynamic counter-query had high token overlap ({overlap:.2f} >= {diversity_threshold}) with baseline; "
            "rejected divergent query and defaulted to baseline to prevent near-duplicate retrieval."
        )
        empirical_lift_expected = "No lift expected (query rejected as near-duplicate of baseline)."
    elif is_exact_match:
        query_used = baseline_query
        difference_rationale = "Dynamic counter-query defaulted to baseline claim text because no anchors or verification directives were present."
        empirical_lift_expected = "No lift expected (query identical to baseline)."
    else:
        query_used = raw_query
        difference_rationale = "Dynamic query incorporates adversarial anchor-shift probes (dispute dates/quantities/locations/predicates) to uncover targeted contradictions."
        empirical_lift_expected = "Higher recall for factual contradictions, chronological errors, and count discrepancies."

    # Construct PlanningAblationLog (Submodule 6.1)
    ablation_log = PlanningAblationLog(
        dynamic_strategy="Adversarial counter-query generation",
        dynamic_queries=[query_used],
        fixed_baseline_strategy="Static contradiction keyword suffix",
        fixed_baseline_queries=[f"{normalized_claim_text} controversy criticism contradiction disputed false refuted"],
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
                    key="prosecutor_ablation_log",
                    value=ablation_log.model_dump(),
                )
        except Exception:
            pass

    # Retrieve evidence using the prosecutor retriever with stance='contradicting'
    raw_evidence = retriever.retrieve(query_used, stance="contradicting")

    min_rel = getattr(settings, "PROSECUTOR_MIN_RELEVANCE", 0.2)
    w_rel = getattr(settings, "PROSECUTOR_RELEVANCE_WEIGHT", 0.5)
    w_cred = getattr(settings, "PROSECUTOR_CREDIBILITY_WEIGHT", 0.3)
    w_fresh = getattr(settings, "PROSECUTOR_FRESHNESS_WEIGHT", 0.2)

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

    # Extract structured challenges (Submodule 6.3)
    structured_challenges = extract_structured_challenges(
        claim_text=normalized_claim_text,
        evidence_items=ranked,
        anchors=anchors,
    )

    # Persist evidence JSON and metadata for this claim
    if query_id is not None:
        try:
            session_store = get_session_store()
            if hasattr(session_store, "record_claim_evidence"):
                session_store.record_claim_evidence(
                    query_id=query_id,
                    claim_id=resolved_claim_id,
                    evidence_type="contradicting",
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
    result = ProsecutorResult(
        claim_text=normalized_claim_text,
        contradicting_evidence=ranked,
        query_used=query_used,
        challenges=structured_challenges,
        challenged_at=datetime.now(timezone.utc),
    )

    # Return API response model
    return ProsecutorResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=result.claim_text,
        contradicting_evidence=result.contradicting_evidence,
        query_used=result.query_used,
        challenges=result.challenges,
        challenged_at=result.challenged_at,
    )



def prosecute_claims_batch(
    claims: list[dict[str, Any] | ProsecutorRequest | Any],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    batch_size: int = 5,
) -> list[ProsecutorResponse]:
    """Prosecute a list of claims in isolated memory-safe batches (max batch_size per slice).

    Respects RAM limits by batching requests (default max 5 claims per slice)
    and aggregating responses sequentially.
    """
    if not claims:
        return []

    responses: list[ProsecutorResponse] = []
    # Process in chunks of batch_size (default 5 to respect RAM limits)
    for i in range(0, len(claims), batch_size):
        batch_slice = claims[i : i + batch_size]
        for item in batch_slice:
            if isinstance(item, ProsecutorRequest):
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

            resp = prosecute_claim(
                claim_text=c_text,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
            )
            responses.append(resp)

    return responses
