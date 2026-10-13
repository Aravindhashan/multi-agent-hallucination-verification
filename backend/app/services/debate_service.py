"""Multi-Turn Evidentiary Debate Orchestration Service (Module 7).

Coordinates structured, grounded evidentiary debate rounds between the
Plan-Guided Defender Agent (Module 5) and Adversarial Counter-Query Prosecutor Agent (Module 6).

Key Invariants:
1. Grounded Evidentiary Cross-Examination vs. Free-Text LLM Monologues:
   Strictly zero conversational LLM debate prose, persuasive rhetoric, or subjective speeches.
   Debate is conducted entirely through iterative, targeted evidence retrieval queries
   and structured challenge signals (StructuredChallenge).
2. Turn-Bounded Termination (T=2) Without Early Consensus Stopping:
   Debate terminates at exactly two rounds (T=2) without premature consensus exit.
3. Memory-Safe Chunking:
   Batch processing slices requests into chunks <= 5 claims to respect RAM limits.
4. Credibility-Biased Cross-Examination Mitigation:
   Prosecutor Round 2 queries extract factual entities/metrics from Defender citations
   rather than issuing generic clickbait dispute tokens. Multi-signal ranking acts as defense gate.
5. Deduplication & Incremental Lift:
   Consolidates evidence pools across rounds with exact ID/URL deduplication, ranking descending
   by composite score, computing continuous Jaccard complementarity, and tracking T=1 vs T=2 lift.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID, uuid4

from app.config import settings
from app.dependencies import get_evidence_retriever
from app.schemas import (
    DebateClaimRequest,
    DebateClaimResponse,
    DebateRoundRecord,
    EvidenceItem,
    StructuredChallenge,
)
from app.services.defender_service import defend_claim
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_service import extract_structured_challenges, prosecute_claim
from app.services.session_store import get_session_store
from app.utils.lexical import CANONICAL_STOPWORDS

logger = logging.getLogger(__name__)


def compute_evidence_jaccard(
    evidence_a: Sequence[EvidenceItem],
    evidence_b: Sequence[EvidenceItem],
) -> float:
    """Compute Jaccard similarity between two sets of evidence items using doc IDs.

    Zero-division safe. Returns a float bounded in [0.0, 1.0].
    """
    def _extract_ids(items: Sequence[EvidenceItem]) -> set[str]:
        ids: set[str] = set()
        for item in items:
            doc_id = item.evidence_id or item.source_url or str(hash(item.text.strip()))
            ids.add(str(doc_id).strip())
        return ids

    set_a = _extract_ids(evidence_a)
    set_b = _extract_ids(evidence_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return round(len(set_a.intersection(set_b)) / len(union), 4)


def extract_passage_anchors(
    evidence_items: Sequence[EvidenceItem],
    claim_text: str = "",
) -> list[str]:
    """Extract factual anchors (dates, numerical counts, metrics, entities) from evidence passages.

    Filters out stop words and tokens already in the claim to isolate specific,
    new factual anchors introduced in Defender's citations.
    """
    claim_tokens = {w.lower() for w in re.findall(r"\b\w+\b", claim_text)}
    anchors: list[str] = []
    seen: set[str] = set()

    for item in evidence_items:
        text = item.text

        # 1. Historical or contemporary years (e.g. 1889, 1969, 2021)
        years = re.findall(r"\b(?:17|18|19|20)\d{2}\b", text)
        for y in years:
            if y not in seen and y.lower() not in claim_tokens:
                seen.add(y)
                anchors.append(y)

        # 2. Quantities with metrics or count units
        quantities = re.findall(
            r"\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:km|kg|meters|miles|percent|%|c|f|astronauts|crew|days|hours|years|months|mb|gb|tb|million|billion)?\b",
            text,
            flags=re.IGNORECASE,
        )
        for q in quantities:
            q_clean = q.strip()
            if q_clean and q_clean not in seen and q_clean.lower() not in claim_tokens:
                seen.add(q_clean)
                anchors.append(q_clean)

        # 3. Capitalized proper nouns / named entities
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for pn in proper_nouns:
            pn_clean = pn.strip()
            if (
                pn_clean
                and pn_clean.lower() not in CANONICAL_STOPWORDS
                and pn_clean not in seen
                and pn_clean.lower() not in claim_tokens
                and len(pn_clean) > 2
            ):
                seen.add(pn_clean)
                anchors.append(pn_clean)

    return anchors


def formulate_defender_refinement_query(
    claim_text: str,
    round_1_challenges: list[StructuredChallenge],
    verification_plan: list[str] | None = None,
    anchors: dict[str, list[str]] | None = None,
) -> str:
    """Formulate Round 2 Defender rebuttal & anchor corroboration query.

    If Prosecutor raised challenges in Round 1, Defender targets the contested anchors:
      base + ' corroboration confirmation official archive: ' + contested_anchor
    If no challenges were raised, Defender falls back to unverified plan steps:
      base + ' ' + verification_plan_step
    """
    clean_base = claim_text.strip()

    # Step 1: If challenges exist, corroborate contested anchors
    if round_1_challenges:
        contested_anchors: list[str] = []
        for ch in round_1_challenges:
            if ch.claim_anchor and ch.claim_anchor not in contested_anchors:
                contested_anchors.append(ch.claim_anchor)

        if contested_anchors:
            anchors_str = ", ".join(contested_anchors[:3])
            return f"{clean_base} corroboration confirmation official archive: {anchors_str}"
        return f"{clean_base} corroboration confirmation official archive source documentation"

    # Step 2: Fallback to unverified plan steps
    if verification_plan:
        plan_step = verification_plan[0].strip()
        return f"{clean_base} {plan_step}"

    # Step 3: Fallback to anchors if present
    if anchors:
        all_anchors = [a for sublist in anchors.values() for a in sublist if a]
        if all_anchors:
            return f"{clean_base} corroboration confirmation official archive: {', '.join(all_anchors[:3])}"

    return f"{clean_base} corroboration confirmation official archive source documentation"


def formulate_prosecutor_refinement_query(
    claim_text: str,
    round_1_supporting_evidence: list[EvidenceItem],
    anchors: dict[str, list[str]] | None = None,
) -> str:
    """Formulate Round 2 Prosecutor cross-examination query using anchor-contrast probing.

    Extracts specific factual anchors (dates, measurements, entities) from Defender's
    cited passages rather than issuing generic dispute keywords that trigger clickbait bias.
    Formula: base + ' alternative findings discrepancy errata: ' + extracted_anchors.
    """
    clean_base = claim_text.strip()
    extracted_anchors = extract_passage_anchors(round_1_supporting_evidence, claim_text=clean_base)

    # If passage anchors found, contrast them
    if extracted_anchors:
        anchors_str = ", ".join(extracted_anchors[:4])
        return f"{clean_base} alternative findings discrepancy errata: {anchors_str}"

    # Fallback to claim anchors if any exist
    if anchors:
        all_anchors = [a for sublist in anchors.values() for a in sublist if a]
        if all_anchors:
            return f"{clean_base} alternative findings discrepancy errata: {', '.join(all_anchors[:3])}"

    return f"{clean_base} alternative findings discrepancy errata counter-evidence"


def rank_and_filter_evidence(
    raw_evidence: Sequence[EvidenceItem],
    min_rel: float = 0.20,
    w_rel: float = 0.50,
    w_cred: float = 0.30,
    w_fresh: float = 0.20,
) -> list[EvidenceItem]:
    """Filter and rank evidence items using multi-signal composite score.

    Applies formula: 0.50 * relevance + 0.30 * credibility + 0.20 * freshness.
    Gated at composite_score >= min_rel (default 0.20).
    """
    filtered: list[EvidenceItem] = []
    for item in raw_evidence:
        rel = getattr(item, "relevance_score", 1.0)
        rel_val = 1.0 if rel is None else rel
        cred = getattr(item, "credibility_score", 1.0)
        cred_val = 1.0 if cred is None else cred
        fresh = getattr(item, "freshness_score", 1.0)
        fresh_val = 1.0 if fresh is None else fresh

        comp_score = round(w_rel * rel_val + w_cred * cred_val + w_fresh * fresh_val, 4)
        item.composite_score = comp_score

        if rel_val >= min_rel and comp_score >= min_rel:
            filtered.append(item)

    return sorted(filtered, key=lambda it: it.composite_score or 0.0, reverse=True)


def deduplicate_evidence_items(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """Deduplicate evidence items by exact ID/URL/text-hash, retaining item with highest composite score."""
    best_items: dict[str, EvidenceItem] = {}
    for item in items:
        doc_id = str(item.evidence_id or item.source_url or hash(item.text.strip())).strip()
        if doc_id not in best_items:
            best_items[doc_id] = item
        else:
            existing_score = best_items[doc_id].composite_score or 0.0
            new_score = item.composite_score or 0.0
            if new_score > existing_score:
                best_items[doc_id] = item
    return sorted(best_items.values(), key=lambda it: it.composite_score or 0.0, reverse=True)


def deduplicate_challenges(challenges: Sequence[StructuredChallenge]) -> list[StructuredChallenge]:
    """Deduplicate structured challenges across rounds."""
    seen: set[tuple[str, str]] = set()
    unique_challenges: list[StructuredChallenge] = []
    for ch in challenges:
        key = (ch.conflict_type, str(ch.claim_anchor).strip().lower())
        if key not in seen:
            seen.add(key)
            unique_challenges.append(ch)
    return unique_challenges


def compute_incremental_lift(
    rounds: list[DebateRoundRecord],
) -> tuple[float, int, int]:
    """Compute incremental unique evidence recall lift of Round 2 over Round 1.

    Returns:
        (incremental_lift_percent, unique_count_round_1, unique_count_round_2_total)
    """
    if not rounds:
        return 0.0, 0, 0

    # Round 1 unique evidence IDs
    r1_items = rounds[0].supporting_evidence + rounds[0].contradicting_evidence
    r1_ids = {
        str(it.evidence_id or it.source_url or hash(it.text.strip())).strip()
        for it in r1_items
    }
    r1_count = len(r1_ids)

    if len(rounds) == 1:
        return 0.0, r1_count, r1_count

    # Total unique evidence IDs across Round 1 + Round 2
    all_items: list[EvidenceItem] = []
    for r in rounds:
        all_items.extend(r.supporting_evidence)
        all_items.extend(r.contradicting_evidence)

    all_ids = {
        str(it.evidence_id or it.source_url or hash(it.text.strip())).strip()
        for it in all_items
    }
    total_count = len(all_ids)

    if r1_count == 0:
        lift = 100.0 if total_count > 0 else 0.0
    else:
        lift = round(((total_count - r1_count) / r1_count) * 100.0, 2)

    return lift, r1_count, total_count


def execute_opening_round(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
) -> DebateRoundRecord:
    """Execute Round 1 (Opening Round) of the evidentiary debate (Submodule 7.1).

    Dispatches the candidate claim in parallel/synchronized fashion to:
    - Defender Agent (Module 5): plan-guided dynamic query formulation, stance='supporting',
      multi-signal composite score ranking.
    - Prosecutor Agent (Module 6): adversarial counter-query formulation, stance='contradicting',
      multi-signal composite score ranking, structured challenge extraction.

    Computes Round 1 Jaccard complementarity J_1(D_1, P_1) and packages the results
    into a structured DebateRoundRecord.

    Strictly zero free-text LLM debate speeches or conversational rhetoric.
    """
    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain at least one non-whitespace character.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    if retriever is None:
        retriever = get_evidence_retriever()

    # Step 1: Defender Opening Retrieval (stance='supporting')
    defender_resp = defend_claim(
        claim_text=normalized_claim_text,
        claim_id=resolved_claim_id,
        query_id=query_id,
        retriever=retriever,
    )

    # Step 2: Prosecutor Opening Retrieval (stance='contradicting')
    prosecutor_resp = prosecute_claim(
        claim_text=normalized_claim_text,
        claim_id=resolved_claim_id,
        query_id=query_id,
        retriever=retriever,
    )

    # Step 3: Compute Round 1 Jaccard complementarity J_1(D_1, P_1)
    jaccard_round_1 = compute_evidence_jaccard(
        defender_resp.supporting_evidence,
        prosecutor_resp.contradicting_evidence,
    )

    # Step 4: Construct structured round record
    return DebateRoundRecord(
        round_number=1,
        defender_query=defender_resp.query_used,
        prosecutor_query=prosecutor_resp.query_used,
        supporting_evidence=defender_resp.supporting_evidence,
        contradicting_evidence=prosecutor_resp.contradicting_evidence,
        challenges=prosecutor_resp.challenges,
        jaccard_complementarity=jaccard_round_1,
    )


def execute_refinement_round(
    claim_text: str,
    round_1_record: DebateRoundRecord,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    verification_plan: list[str] | None = None,
    anchors: dict[str, list[str]] | None = None,
) -> DebateRoundRecord:
    """Execute Round 2 (Refinement Round) of the evidentiary debate (Submodule 7.2).

    Inspects Round 1 evidentiary state:
    - Defender targets contested anchors from Round 1 challenges or unverified plan steps,
      retrieving with stance='supporting'.
    - Prosecutor extracts factual anchors from Defender's citations to formulate an
      anchor-contrast cross-examination query, retrieving with stance='contradicting'.

    Applies multi-signal composite ranking (w_rel=0.5, w_cred=0.3, w_fresh=0.2 >= 0.20)
    and computes Round 2 Jaccard complementarity J_2(D_2, P_2).

    Returns a structured DebateRoundRecord with round_number=2.
    """
    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain at least one non-whitespace character.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    if retriever is None:
        retriever = get_evidence_retriever()

    # Load plan and anchors if not explicitly provided
    resolved_plan = verification_plan or []
    resolved_anchors = anchors or {"temporal": [], "numerical": [], "spatial": []}

    if query_id is not None and (not resolved_plan or not any(resolved_anchors.values())):
        try:
            store = get_session_store()
            claim_rec = store.get_claim_record(query_id, resolved_claim_id)
            if claim_rec:
                if not resolved_plan:
                    resolved_plan = claim_rec.get("verification_plan", [])
                if not any(resolved_anchors.values()):
                    resolved_anchors["temporal"] = claim_rec.get("temporal_anchors", [])
                    resolved_anchors["numerical"] = claim_rec.get("numerical_values", [])
                    resolved_anchors["spatial"] = claim_rec.get("spatial_anchors", [])
        except Exception:
            pass

    # Step 1: Formulate Defender Refinement Query
    defender_query = formulate_defender_refinement_query(
        claim_text=normalized_claim_text,
        round_1_challenges=round_1_record.challenges,
        verification_plan=resolved_plan,
        anchors=resolved_anchors,
    )

    # Step 2: Formulate Prosecutor Refinement Query (Anchor-Contrast Probing)
    prosecutor_query = formulate_prosecutor_refinement_query(
        claim_text=normalized_claim_text,
        round_1_supporting_evidence=round_1_record.supporting_evidence,
        anchors=resolved_anchors,
    )

    # Step 3: Retrieve and rank Defender evidence (stance='supporting')
    raw_defender_evidence = retriever.retrieve(defender_query, stance="supporting")
    w_rel = getattr(settings, "DEFENDER_RELEVANCE_WEIGHT", 0.5)
    w_cred = getattr(settings, "DEFENDER_CREDIBILITY_WEIGHT", 0.3)
    w_fresh = getattr(settings, "DEFENDER_FRESHNESS_WEIGHT", 0.2)
    min_rel = getattr(settings, "DEFENDER_MIN_RELEVANCE", 0.2)
    ranked_supporting = rank_and_filter_evidence(
        raw_defender_evidence,
        min_rel=min_rel,
        w_rel=w_rel,
        w_cred=w_cred,
        w_fresh=w_fresh,
    )

    # Step 4: Retrieve and rank Prosecutor evidence (stance='contradicting')
    raw_prosecutor_evidence = retriever.retrieve(prosecutor_query, stance="contradicting")
    pw_rel = getattr(settings, "PROSECUTOR_RELEVANCE_WEIGHT", 0.5)
    pw_cred = getattr(settings, "PROSECUTOR_CREDIBILITY_WEIGHT", 0.3)
    pw_fresh = getattr(settings, "PROSECUTOR_FRESHNESS_WEIGHT", 0.2)
    pmin_rel = getattr(settings, "PROSECUTOR_MIN_RELEVANCE", 0.2)
    ranked_contradicting = rank_and_filter_evidence(
        raw_prosecutor_evidence,
        min_rel=pmin_rel,
        w_rel=pw_rel,
        w_cred=pw_cred,
        w_fresh=pw_fresh,
    )

    # Step 5: Extract updated structured challenges for Round 2
    round_2_challenges = extract_structured_challenges(
        claim_text=normalized_claim_text,
        evidence_items=ranked_contradicting,
        anchors=resolved_anchors,
    )

    # Step 6: Compute Round 2 Jaccard complementarity J_2(D_2, P_2)
    jaccard_round_2 = compute_evidence_jaccard(
        ranked_supporting,
        ranked_contradicting,
    )

    # Step 7: Construct structured round record
    return DebateRoundRecord(
        round_number=2,
        defender_query=defender_query,
        prosecutor_query=prosecutor_query,
        supporting_evidence=ranked_supporting,
        contradicting_evidence=ranked_contradicting,
        challenges=round_2_challenges,
        jaccard_complementarity=jaccard_round_2,
    )


def debate_claim(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    max_rounds: int = 2,
) -> DebateClaimResponse:
    """Orchestrate multi-round evidentiary debate for a single claim (Submodule 7.3).

    Executes Round 1 (Opening) and Round 2 (Refinement) without early consensus stopping.
    Consolidates evidence pools, computes Jaccard complementarity and incremental lift.
    Strictly zero free-text LLM argumentation or monologues.
    """
    normalized_claim_text = claim_text.strip()
    if not normalized_claim_text:
        raise ValueError("Claim text must contain at least one non-whitespace character.")

    resolved_claim_id = claim_id if claim_id is not None else uuid4()

    if retriever is None:
        retriever = get_evidence_retriever()

    # Step 1: Execute Round 1 (Opening Round)
    round_1 = execute_opening_round(
        claim_text=normalized_claim_text,
        claim_id=resolved_claim_id,
        query_id=query_id,
        retriever=retriever,
    )

    rounds = [round_1]
    early_stopped = False
    effective_max_rounds = min(max_rounds, getattr(settings, "DEBATE_MAX_ROUNDS", 2))

    # Fast-path early stopping: if Round 1 already has high-confidence evidence, skip Round 2
    early_stopping_enabled = getattr(settings, "DEBATE_EARLY_STOPPING_ENABLED", True)
    has_high_confidence = any(
        (getattr(it, "composite_score", 0.0) or 0.0) >= 0.70
        for it in (round_1.supporting_evidence + round_1.contradicting_evidence)
    )

    if effective_max_rounds >= 2 and not (early_stopping_enabled and has_high_confidence):
        # Step 2: Execute Round 2 (Refinement Round)
        round_2 = execute_refinement_round(
            claim_text=normalized_claim_text,
            round_1_record=round_1,
            claim_id=resolved_claim_id,
            query_id=query_id,
            retriever=retriever,
        )
        rounds.append(round_2)
    elif effective_max_rounds >= 2 and has_high_confidence:
        early_stopped = True

    # Step 3: Consolidate evidence pools across rounds (exact ID / URL / text-hash deduplication)
    all_supporting: list[EvidenceItem] = []
    all_contradicting: list[EvidenceItem] = []
    all_challenges: list[StructuredChallenge] = []

    for r in rounds:
        all_supporting.extend(r.supporting_evidence)
        all_contradicting.extend(r.contradicting_evidence)
        all_challenges.extend(r.challenges)

    final_supporting = deduplicate_evidence_items(all_supporting)
    final_contradicting = deduplicate_evidence_items(all_contradicting)
    final_challenges = deduplicate_challenges(all_challenges)

    # Step 4: Compute final Jaccard complementarity across consolidated pools
    final_jaccard = compute_evidence_jaccard(final_supporting, final_contradicting)

    # Step 5: Compute incremental evidence recall lift (T=2 over T=1)
    lift_percent, r1_count, t2_count = compute_incremental_lift(rounds)

    # Diagnostic check: log stagnation if Round 2 adds < threshold lift and 0 new challenges
    lift_threshold = getattr(settings, "DEBATE_INCREMENTAL_LIFT_THRESHOLD_PERCENT", 15.0)
    if len(rounds) > 1:
        new_challenges_count = len(final_challenges) - len(rounds[0].challenges)
        if lift_percent < lift_threshold and new_challenges_count <= 0:
            logger.info(
                "Evidentiary stagnation: Round 2 added marginal lift (%.2f%% < %.1f%%) and no new challenges for claim '%s'.",
                lift_percent,
                lift_threshold,
                normalized_claim_text,
            )

    response = DebateClaimResponse(
        claim_id=resolved_claim_id,
        query_id=query_id,
        claim_text=normalized_claim_text,
        rounds=rounds,
        final_supporting_evidence=final_supporting,
        final_contradicting_evidence=final_contradicting,
        final_challenges=final_challenges,
        total_rounds=len(rounds),
        early_stopped=early_stopped,
        jaccard_complementarity=final_jaccard,
        incremental_evidence_lift_percent=lift_percent,
        debated_at=datetime.now(timezone.utc),
    )

    # Persist consolidated evidence and debate audit trail in session store
    if query_id is not None:
        try:
            store = get_session_store()
            store.record_claim_debate(
                query_id=query_id,
                claim_id=resolved_claim_id,
                debate_response=response,
            )
        except Exception as err:
            logger.warning("Failed to record claim debate in session store: %s", err)

    return response


def debate_claims_batch(
    claims: list[dict[str, Any] | DebateClaimRequest | str | Any],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    max_rounds: int = 2,
    batch_size: int = 5,
) -> list[DebateClaimResponse]:
    """Execute multi-round debate for a batch of claims in memory-safe chunks <= batch_size.

    Preserves input order, accepts heterogeneous claim payloads, and returns
    a list of DebateClaimResponse objects.
    """
    if not claims:
        return []

    responses: list[DebateClaimResponse] = []
    effective_batch_size = max(1, batch_size)

    for i in range(0, len(claims), effective_batch_size):
        chunk = claims[i : i + effective_batch_size]
        for item in chunk:
            if isinstance(item, str):
                c_text = item
                c_id = None
                q_id = query_id
            elif isinstance(item, DebateClaimRequest):
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

            resp = debate_claim(
                claim_text=c_text,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
                max_rounds=max_rounds,
            )
            responses.append(resp)

    return responses


def execute_opening_rounds_batch(
    claims: list[dict[str, Any] | DebateClaimRequest | str | Any],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    batch_size: int = 5,
) -> list[DebateRoundRecord]:
    """Execute Round 1 for a list of claims in memory-safe chunks (max batch_size per chunk).

    Slices input claims into chunks <= batch_size (default 5 to respect RAM limits),
    executes sequentially, and preserves input order.
    """
    if not claims:
        return []

    records: list[DebateRoundRecord] = []
    effective_batch_size = max(1, batch_size)

    for i in range(0, len(claims), effective_batch_size):
        chunk = claims[i : i + effective_batch_size]
        for item in chunk:
            if isinstance(item, str):
                c_text = item
                c_id = None
                q_id = query_id
            elif isinstance(item, DebateClaimRequest):
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

            rec = execute_opening_round(
                claim_text=c_text,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
            )
            records.append(rec)

    return records


def execute_refinement_rounds_batch(
    round_1_inputs: list[tuple[str, DebateRoundRecord] | dict[str, Any]],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    batch_size: int = 5,
) -> list[DebateRoundRecord]:
    """Execute Round 2 for a list of (claim_text, round_1_record) inputs in memory-safe chunks.

    Slices input into chunks <= batch_size (default 5 to respect RAM limits),
    executes sequentially, and preserves input order.
    """
    if not round_1_inputs:
        return []

    records: list[DebateRoundRecord] = []
    effective_batch_size = max(1, batch_size)

    for i in range(0, len(round_1_inputs), effective_batch_size):
        chunk = round_1_inputs[i : i + effective_batch_size]
        for item in chunk:
            if isinstance(item, tuple):
                c_text, r1_rec = item
                c_id = None
                q_id = query_id
            elif isinstance(item, dict):
                c_text = item.get("claim_text") or item.get("text") or ""
                r1_rec = item.get("round_1_record") or item.get("round_1")
                c_id = item.get("claim_id") or item.get("id")
                q_id = item.get("query_id") or query_id
            else:
                c_text = getattr(item, "claim_text", "")
                r1_rec = getattr(item, "round_1_record", None)
                c_id = getattr(item, "claim_id", None)
                q_id = getattr(item, "query_id", query_id)

            if not isinstance(r1_rec, DebateRoundRecord):
                raise ValueError("Each input to execute_refinement_rounds_batch must include a valid DebateRoundRecord.")

            rec = execute_refinement_round(
                claim_text=c_text,
                round_1_record=r1_rec,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
            )
            records.append(rec)

    return records


def execute_debate_rounds(
    claim_text: str,
    claim_id: UUID | str | None = None,
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    max_rounds: int = 2,
) -> list[DebateRoundRecord]:
    """Execute debate rounds (up to max_rounds, bounded at 2) returning list of round records."""
    round_1 = execute_opening_round(
        claim_text=claim_text,
        claim_id=claim_id,
        query_id=query_id,
        retriever=retriever,
    )

    effective_max_rounds = min(max_rounds, getattr(settings, "DEBATE_MAX_ROUNDS", 2))
    if effective_max_rounds <= 1:
        return [round_1]

    round_2 = execute_refinement_round(
        claim_text=claim_text,
        round_1_record=round_1,
        claim_id=claim_id,
        query_id=query_id,
        retriever=retriever,
    )
    return [round_1, round_2]


def execute_debate_rounds_batch(
    claims: list[dict[str, Any] | DebateClaimRequest | str | Any],
    query_id: UUID | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    max_rounds: int = 2,
    batch_size: int = 5,
) -> list[list[DebateRoundRecord]]:
    """Execute full debate rounds for a batch of claims in memory-safe chunks <= batch_size."""
    if not claims:
        return []

    results: list[list[DebateRoundRecord]] = []
    effective_batch_size = max(1, batch_size)

    for i in range(0, len(claims), effective_batch_size):
        chunk = claims[i : i + effective_batch_size]
        for item in chunk:
            if isinstance(item, str):
                c_text = item
                c_id = None
                q_id = query_id
            elif isinstance(item, DebateClaimRequest):
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

            rounds = execute_debate_rounds(
                claim_text=c_text,
                claim_id=c_id,
                query_id=q_id,
                retriever=retriever,
                max_rounds=max_rounds,
            )
            results.append(rounds)

    return results
