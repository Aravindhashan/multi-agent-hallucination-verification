"""Recall ablation harness and counter-perspective analysis service.

Evaluates evidence recall across four retrieval configurations:
1. Baseline: Naive single-turn retrieval using verbatim claim text.
2. Defender-only: Plan-guided dynamic supporting retrieval.
3. Prosecutor-only: Adversarial counter-query contradicting retrieval.
4. Dual-perspective: Deduplicated union of Defender and Prosecutor evidence.

Computes total recall, contradiction recall, Jaccard complementarity (target < 0.50),
and dual-perspective lift over baseline (target > 0%).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
from uuid import uuid4

from app.config import settings
from app.dependencies import get_evidence_retriever
from app.schemas import (
    Claim,
    EvidenceItem,
    PerClaimRecallMetrics,
    RecallAblationSummary,
)
from app.services.claim_models import AtomicClaim
from app.services.defender_service import defend_claim
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.prosecutor_service import prosecute_claim

logger = logging.getLogger(__name__)


def _extract_doc_ids(items: Sequence[EvidenceItem]) -> set[str]:
    """Extract a set of unique document/evidence identifiers from evidence items."""
    ids: set[str] = set()
    for item in items:
        # Prefer explicit evidence_id, fall back to source_url or text hash
        doc_id = item.evidence_id or item.source_url or str(hash(item.text.strip()))
        ids.add(str(doc_id).strip())
    return ids


def _compute_jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets with zero-division safety."""
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform safe division with fallback when denominator is zero."""
    if denominator == 0.0:
        return default
    return numerator / denominator


class PerspectiveRecallAblationHarness:
    """Ablation harness comparing single-perspective vs dual-perspective retrieval."""

    def __init__(
        self,
        retriever: BaseEvidenceRetriever | None = None,
        nlp_extractor: NLPClaimExtractor | None = None,
    ) -> None:
        self.retriever = retriever or get_evidence_retriever()
        self.nlp_extractor = nlp_extractor or NLPClaimExtractor()

    def evaluate_claim(
        self,
        claim_input: str | dict[str, Any] | AtomicClaim | Claim,
        gold_evidence_ids: Sequence[str] | None = None,
        gold_contradiction_ids: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> PerClaimRecallMetrics:
        """Evaluate a single claim across Baseline, Defender, Prosecutor, and Dual modes."""
        # 1. Normalize claim object and extract text & ID
        claim_id: str
        claim_text: str
        atomic_claim: AtomicClaim | None = None

        if isinstance(claim_input, str):
            claim_text = claim_input.strip()
            claim_id = f"c_{uuid4().hex[:8]}"
            # Extract anchors using NLP extractor for rich query formulation
            extracted_result = self.nlp_extractor.extract_atomic_claims(claim_text)
            if extracted_result.claims:
                atomic_claim = extracted_result.claims[0]
                claim_id = str(atomic_claim.claim_id)
        elif isinstance(claim_input, (AtomicClaim, Claim)):
            claim_text = claim_input.claim_text.strip()
            claim_id = str(claim_input.claim_id)
            if isinstance(claim_input, AtomicClaim):
                atomic_claim = claim_input
        elif isinstance(claim_input, dict):
            claim_text = str(claim_input.get("claim_text") or claim_input.get("text") or "").strip()
            claim_id = str(claim_input.get("claim_id") or f"c_{uuid4().hex[:8]}")
        else:
            claim_text = str(claim_input).strip()
            claim_id = f"c_{uuid4().hex[:8]}"

        if not claim_text:
            return PerClaimRecallMetrics(
                claim_id=claim_id,
                claim_text="",
                baseline_evidence_count=0,
                defender_evidence_count=0,
                prosecutor_evidence_count=0,
                dual_perspective_evidence_count=0,
                jaccard_complementarity=0.0,
            )

        # 2. Run Baseline Retrieval (Naive single-turn verbatim search)
        raw_baseline = self.retriever.retrieve(claim_text, stance="supporting")
        # Apply baseline relevance threshold
        min_relevance = settings.DEFENDER_MIN_RELEVANCE
        baseline_retained = [
            item for item in raw_baseline
            if (item.relevance_score or 0.0) >= min_relevance
        ][:top_k]
        baseline_doc_ids = _extract_doc_ids(baseline_retained)

        # 3. Run Defender Retrieval (Dynamic plan/anchor-guided)
        defender_result = defend_claim(
            claim_text=claim_text,
            claim_id=claim_id,
            retriever=self.retriever,
        )
        defender_doc_ids = _extract_doc_ids(defender_result.supporting_evidence)

        # 4. Run Prosecutor Retrieval (Adversarial counter-query)
        prosecutor_result = prosecute_claim(
            claim_text=claim_text,
            claim_id=claim_id,
            retriever=self.retriever,
        )
        prosecutor_doc_ids = _extract_doc_ids(prosecutor_result.contradicting_evidence)


        # 5. Dual Perspective (Union of Defender and Prosecutor)
        dual_doc_ids = defender_doc_ids.union(prosecutor_doc_ids)

        # 6. Jaccard Complementarity between Defender and Prosecutor
        jaccard = _compute_jaccard(defender_doc_ids, prosecutor_doc_ids)

        # 7. Establish Ground Truth Evidence Universe
        if gold_evidence_ids is not None:
            gold_set = set(gold_evidence_ids)
        else:
            # Pooled evidence universe across all perspectives
            gold_set = baseline_doc_ids.union(defender_doc_ids).union(prosecutor_doc_ids)

        # 8. Establish Contradiction Ground Truth
        if gold_contradiction_ids is not None:
            gold_contra_set = set(gold_contradiction_ids)
        else:
            # Identify documents flagged by Prosecutor challenges or contradiction tokens
            gold_contra_set = set()
            for ev in prosecutor_result.contradicting_evidence:
                ev_id = ev.evidence_id or ev.source_url or str(hash(ev.text.strip()))
                if ev.challenge or any(
                    token in ev.text.lower()
                    for token in ("dispute", "false", "refuted", "debunked", "untrue", "mistake")
                ):
                    gold_contra_set.add(str(ev_id).strip())

        # 9. Compute Recalls
        gold_len = len(gold_set)
        b_recall = _safe_divide(len(baseline_doc_ids.intersection(gold_set)), gold_len) if gold_len > 0 else 0.0
        d_recall = _safe_divide(len(defender_doc_ids.intersection(gold_set)), gold_len) if gold_len > 0 else 0.0
        p_recall = _safe_divide(len(prosecutor_doc_ids.intersection(gold_set)), gold_len) if gold_len > 0 else 0.0
        dual_recall = _safe_divide(len(dual_doc_ids.intersection(gold_set)), gold_len) if gold_len > 0 else 0.0

        gold_contra_len = len(gold_contra_set)
        contra_recall_b = _safe_divide(len(baseline_doc_ids.intersection(gold_contra_set)), gold_contra_len) if gold_contra_len > 0 else 0.0
        contra_recall_p = _safe_divide(len(prosecutor_doc_ids.intersection(gold_contra_set)), gold_contra_len) if gold_contra_len > 0 else 0.0
        contra_recall_dual = _safe_divide(len(dual_doc_ids.intersection(gold_contra_set)), gold_contra_len) if gold_contra_len > 0 else 0.0

        # 10. Compute Lifts
        if b_recall > 0:
            recall_lift_pct = round(((dual_recall - b_recall) / b_recall) * 100.0, 2)
        elif dual_recall > 0:
            recall_lift_pct = 100.0
        else:
            recall_lift_pct = 0.0

        b_count = len(baseline_doc_ids)
        dual_count = len(dual_doc_ids)
        if b_count > 0:
            evidence_lift_pct = round(((dual_count - b_count) / b_count) * 100.0, 2)
        elif dual_count > 0:
            evidence_lift_pct = 100.0
        else:
            evidence_lift_pct = 0.0

        # 11. Compute Recall@k (k=1, 3, 5)
        # Deduplicate and sort dual-perspective items by composite score
        all_dual_items = defender_result.supporting_evidence + prosecutor_result.contradicting_evidence
        seen_dual: set[str] = set()
        deduped_dual: list[EvidenceItem] = []
        for it in sorted(all_dual_items, key=lambda x: x.composite_score or 0.0, reverse=True):
            it_id = str(it.evidence_id or it.source_url or hash(it.text.strip()))
            if it_id not in seen_dual:
                seen_dual.add(it_id)
                deduped_dual.append(it)

        def _calc_r_at_k(items: Sequence[EvidenceItem], target_gold: set[str], k_val: int) -> float:
            if not target_gold:
                return 0.0
            top_k_ids = _extract_doc_ids(items[:k_val])
            return round(_safe_divide(len(top_k_ids.intersection(target_gold)), len(target_gold)), 4)

        b_r1 = _calc_r_at_k(baseline_retained, gold_set, 1)
        b_r3 = _calc_r_at_k(baseline_retained, gold_set, 3)
        b_r5 = _calc_r_at_k(baseline_retained, gold_set, 5)

        dual_r1 = _calc_r_at_k(deduped_dual, gold_set, 1)
        dual_r3 = _calc_r_at_k(deduped_dual, gold_set, 3)
        dual_r5 = _calc_r_at_k(deduped_dual, gold_set, 5)

        return PerClaimRecallMetrics(
            claim_id=claim_id,
            claim_text=claim_text,
            baseline_evidence_count=b_count,
            defender_evidence_count=len(defender_doc_ids),
            prosecutor_evidence_count=len(prosecutor_doc_ids),
            dual_perspective_evidence_count=dual_count,
            jaccard_complementarity=round(jaccard, 4),
            baseline_recall=round(b_recall, 4),
            defender_recall=round(d_recall, 4),
            prosecutor_recall=round(p_recall, 4),
            dual_recall=round(dual_recall, 4),
            contradiction_recall_baseline=round(contra_recall_b, 4),
            contradiction_recall_prosecutor=round(contra_recall_p, 4),
            contradiction_recall_dual=round(contra_recall_dual, 4),
            recall_lift_percent=recall_lift_pct,
            unique_evidence_lift_percent=evidence_lift_pct,
            baseline_recall_at_1=b_r1,
            baseline_recall_at_3=b_r3,
            baseline_recall_at_5=b_r5,
            dual_recall_at_1=dual_r1,
            dual_recall_at_3=dual_r3,
            dual_recall_at_5=dual_r5,
        )


    def evaluate_claims_batch(
        self,
        claims: Sequence[str | dict[str, Any] | AtomicClaim | Claim],
        gold_evidence_by_claim: dict[str, Sequence[str]] | None = None,
        gold_contradictions_by_claim: dict[str, Sequence[str]] | None = None,
        top_k: int = 5,
    ) -> RecallAblationSummary:
        """Run perspective recall ablation across a batch of claims and aggregate metrics."""
        if not claims:
            return RecallAblationSummary(
                total_claims_evaluated=0,
                mean_baseline_recall=0.0,
                mean_defender_recall=0.0,
                mean_prosecutor_recall=0.0,
                mean_dual_recall=0.0,
                mean_jaccard_complementarity=0.0,
                mean_contradiction_recall_baseline=0.0,
                mean_contradiction_recall_prosecutor=0.0,
                mean_contradiction_recall_dual=0.0,
                mean_recall_lift_percent=0.0,
                mean_unique_evidence_lift_percent=0.0,
                per_claim_results=[],
            )

        gold_ev_map = gold_evidence_by_claim or {}
        gold_contra_map = gold_contradictions_by_claim or {}

        per_claim_results: list[PerClaimRecallMetrics] = []

        for item in claims:
            # Resolve gold IDs for this claim
            cid = ""
            ctext = ""
            if isinstance(item, (AtomicClaim, Claim)):
                cid = str(item.claim_id)
                ctext = item.claim_text
            elif isinstance(item, dict):
                cid = str(item.get("claim_id") or "")
                ctext = str(item.get("claim_text") or item.get("text") or "")
            elif isinstance(item, str):
                ctext = item.strip()

            gold_ev = gold_ev_map.get(cid) or gold_ev_map.get(ctext) or (list(gold_ev_map.values())[0] if len(claims) == 1 and len(gold_ev_map) == 1 else None)
            gold_contra = gold_contra_map.get(cid) or gold_contra_map.get(ctext) or (list(gold_contra_map.values())[0] if len(claims) == 1 and len(gold_contra_map) == 1 else None)


            metrics = self.evaluate_claim(
                claim_input=item,
                gold_evidence_ids=gold_ev,
                gold_contradiction_ids=gold_contra,
                top_k=top_k,
            )
            per_claim_results.append(metrics)

        n = len(per_claim_results)
        mean_b_recall = sum(m.baseline_recall for m in per_claim_results) / n
        mean_d_recall = sum(m.defender_recall for m in per_claim_results) / n
        mean_p_recall = sum(m.prosecutor_recall for m in per_claim_results) / n
        mean_dual_recall = sum(m.dual_recall for m in per_claim_results) / n
        mean_jaccard = sum(m.jaccard_complementarity for m in per_claim_results) / n
        mean_contra_b = sum(m.contradiction_recall_baseline for m in per_claim_results) / n
        mean_contra_p = sum(m.contradiction_recall_prosecutor for m in per_claim_results) / n
        mean_contra_dual = sum(m.contradiction_recall_dual for m in per_claim_results) / n
        mean_recall_lift = sum(m.recall_lift_percent for m in per_claim_results) / n
        mean_unique_lift = sum(m.unique_evidence_lift_percent for m in per_claim_results) / n
        mean_b_r1 = sum(m.baseline_recall_at_1 for m in per_claim_results) / n
        mean_b_r3 = sum(m.baseline_recall_at_3 for m in per_claim_results) / n
        mean_b_r5 = sum(m.baseline_recall_at_5 for m in per_claim_results) / n
        mean_dual_r1 = sum(m.dual_recall_at_1 for m in per_claim_results) / n
        mean_dual_r3 = sum(m.dual_recall_at_3 for m in per_claim_results) / n
        mean_dual_r5 = sum(m.dual_recall_at_5 for m in per_claim_results) / n


        return RecallAblationSummary(
            total_claims_evaluated=n,
            mean_baseline_recall=round(mean_b_recall, 4),
            mean_defender_recall=round(mean_d_recall, 4),
            mean_prosecutor_recall=round(mean_p_recall, 4),
            mean_dual_recall=round(mean_dual_recall, 4),
            mean_jaccard_complementarity=round(mean_jaccard, 4),
            mean_contradiction_recall_baseline=round(mean_contra_b, 4),
            mean_contradiction_recall_prosecutor=round(mean_contra_p, 4),
            mean_contradiction_recall_dual=round(mean_contra_dual, 4),
            mean_recall_lift_percent=round(mean_recall_lift, 2),
            mean_unique_evidence_lift_percent=round(mean_unique_lift, 2),
            mean_baseline_recall_at_1=round(mean_b_r1, 4),
            mean_baseline_recall_at_3=round(mean_b_r3, 4),
            mean_baseline_recall_at_5=round(mean_b_r5, 4),
            mean_dual_recall_at_1=round(mean_dual_r1, 4),
            mean_dual_recall_at_3=round(mean_dual_r3, 4),
            mean_dual_recall_at_5=round(mean_dual_r5, 4),
            per_claim_results=per_claim_results,
        )



def run_perspective_recall_ablation(
    claims: Sequence[str | dict[str, Any] | AtomicClaim | Claim],
    retriever: BaseEvidenceRetriever | None = None,
    gold_evidence_by_claim: dict[str, Sequence[str]] | None = None,
    gold_contradictions_by_claim: dict[str, Sequence[str]] | None = None,
    top_k: int = 5,
) -> RecallAblationSummary:
    """Convenience functional interface to execute perspective recall ablation."""
    harness = PerspectiveRecallAblationHarness(retriever=retriever)
    return harness.evaluate_claims_batch(
        claims=claims,
        gold_evidence_by_claim=gold_evidence_by_claim,
        gold_contradictions_by_claim=gold_contradictions_by_claim,
        top_k=top_k,
    )
