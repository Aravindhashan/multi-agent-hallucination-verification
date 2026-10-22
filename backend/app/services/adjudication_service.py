"""Natural Language Inference (NLI) Cross-Encoder Adjudication Service (Module 9).

Integrates the Federated Evidence Engine, Multi-Turn Evidentiary Debate (Module 7),
and Evidence Quality Firewall & Cluster Pooling (Module 8) with DeBERTa-v3 cross-encoder NLI
to adjudicate open-world claims into calibrated, evidence-backed verdicts.

Key Architectural Invariants:
1. Double-Counting Prevention:
   Downstream DeBERTa NLI evaluates text relevance/entailment without double-counting
   intrinsic source credibility, temporal freshness, or redundancy (Reviewer §3 & §4).
   Cluster reliability weights W(C_k) = S_intrinsic(e_hat_k) gate evidence impact.
2. Downstream NLI Call Optimization:
   NLI inference is executed exclusively on winning cluster representatives (e_hat_k),
   reducing inference volume from M raw items to K_adm clusters (up to ~65% reduction).
3. Memory Safety on Windows:
   Batch processing operates in bounded slices (default batch_size=2) to prevent
   OS paging file exhaustion and PyTorch CUDA/CPU allocator fragmentation.
4. Tri-Class Verdict Taxonomy:
   Outputs SUPPORTED, REFUTED, or NOT_ENOUGH_INFO with calibrated confidence and
   redundancy-discounted effective sample size (n_eff).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import threading
import time
from typing import Any, Sequence
from app.utils.lexical import CANONICAL_STOPWORDS
import re

_TOKEN_PATTERN = re.compile(r"\b\w+\b")
from uuid import UUID, uuid4

from app.schemas import (
    AdjudicationBatchRequest,
    AdjudicationBatchResponse,
    AdjudicationItemResult,
    AdjudicationRequest,
    AdjudicationResponse,
    AdjudicationVerdict,
    DebateClaimResponse,
    EvidenceItem,
)
from app.services.cluster_pooling import combine_nli_probabilities
from app.services.debate_service import debate_claim
from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.nli_engine import NLIEngine, get_nli_engine
from app.services.nli_models import NLILabel, NLIResult
from app.services.sentence_window import select_best_sentence_window
from app.services.session_store import get_session_store

logger = logging.getLogger("hallucination_verifier.adjudication_service")


class AdjudicationService:
    """Orchestrates cross-encoder NLI inference and Bayesian belief aggregation across federated evidence."""

    def __init__(
        self,
        nli_engine: NLIEngine | None = None,
        quality_service: EvidenceQualityService | None = None,
        retriever: BaseEvidenceRetriever | None = None,
    ) -> None:
        """Initialize AdjudicationService with optional service overrides."""
        self._nli_engine = nli_engine
        self._quality_service = quality_service
        self._retriever = retriever
        self._lock = threading.Lock()

    def _get_nli_engine(self) -> NLIEngine:
        """Return or lazily initialize the singleton NLIEngine."""
        if self._nli_engine is None:
            with self._lock:
                if self._nli_engine is None:
                    self._nli_engine = get_nli_engine()
        return self._nli_engine

    def _get_quality_service(self) -> EvidenceQualityService:
        """Return or lazily initialize the singleton EvidenceQualityService."""
        if self._quality_service is None:
            with self._lock:
                if self._quality_service is None:
                    self._quality_service = get_evidence_quality_service()
        return self._quality_service

    def _get_retriever(self) -> BaseEvidenceRetriever:
        """Return or lazily initialize the active evidence retriever."""
        if self._retriever is None:
            from app.dependencies import get_evidence_retriever

            return get_evidence_retriever()
        return self._retriever

    def adjudicate_claim(
        self,
        claim_text: str,
        claim_id: UUID | str | None = None,
        query_id: UUID | None = None,
        supporting_evidence: Sequence[EvidenceItem] | None = None,
        contradicting_evidence: Sequence[EvidenceItem] | None = None,
        run_debate: bool = True,
        max_debate_rounds: int = 2,
    ) -> AdjudicationResponse:
        """Adjudicate a factual claim against federated evidence via DeBERTa cross-encoder NLI.

        Args:
            claim_text: Factual claim text to verify.
            claim_id: Optional claim identifier.
            query_id: Optional pipeline query identifier.
            supporting_evidence: Pre-retrieved supporting evidence. If omitted, runs federated debate.
            contradicting_evidence: Pre-retrieved contradicting evidence. If omitted, runs federated debate.
            run_debate: Whether to run multi-turn debate if evidence is not provided.
            max_debate_rounds: Maximum debate rounds (default 2).

        Returns:
            AdjudicationResponse with verdict, confidence, probabilities, and provenance.
        """
        normalized_claim = claim_text.strip()
        if not normalized_claim:
            raise ValueError("Claim text must contain at least one non-whitespace character.")

        resolved_claim_id = claim_id if claim_id is not None else uuid4()

        # Step 1: Gather Dual-Perspective Evidence if not explicitly provided
        sup_items = list(supporting_evidence) if supporting_evidence is not None else []
        con_items = list(contradicting_evidence) if contradicting_evidence is not None else []

        if not sup_items and not con_items and run_debate:
            retriever = self._get_retriever()
            debate_resp: DebateClaimResponse = debate_claim(
                claim_text=normalized_claim,
                claim_id=resolved_claim_id,
                query_id=query_id,
                retriever=retriever,
                max_rounds=max_debate_rounds,
            )
            sup_items = debate_resp.final_supporting_evidence
            con_items = debate_resp.final_contradicting_evidence

        # Step 2: Evaluate Quality & Perform Cluster Representative Pooling (Module 8)
        qs = self._get_quality_service()
        quality_eval = qs.evaluate_debate_quality(
            claim=normalized_claim,
            supporting_evidence=sup_items,
            contradicting_evidence=con_items,
            claim_id=resolved_claim_id,
            query_id=query_id,
        )

        pooled_sup = quality_eval.pooled_supporting_evidence
        pooled_con = quality_eval.pooled_contradicting_evidence
        n_eff_sup = quality_eval.supporting_n_eff
        n_eff_con = quality_eval.contradicting_n_eff
        joint_n_eff = quality_eval.joint_n_eff

        # Step 3: Run DeBERTa Cross-Encoder Inference on Cluster Representatives
        engine = self._get_nli_engine()
        evaluated_evidence: list[AdjudicationItemResult] = []

        # Prepare evaluation pairs
        eval_pairs: list[tuple[str, str]] = []
        item_metadata: list[dict[str, Any]] = []

        # Process supporting cluster representatives (pruned to top 4 for comprehensive coverage)
        sup_clusters_map = {
            c.representative_id: c.cluster_reliability_weight
            for c in quality_eval.supporting_report.clusters
        }
        claim_content_words = {
            w.lower() for w in _TOKEN_PATTERN.findall(normalized_claim)
            if len(w) >= 3 and w.lower() not in CANONICAL_STOPWORDS
        }
        def _rank_candidate(it: EvidenceItem, cluster_map: dict[str, float]) -> float:
            base_w = (
                cluster_map.get(str(it.evidence_id))
                or getattr(it, "cluster_reliability_weight", None)
                or getattr(it, "composite_score", 0.8)
                or 0.8
            )
            it_words = set(_TOKEN_PATTERN.findall(it.text.lower()))
            overlap = len(claim_content_words.intersection(it_words)) / max(1, len(claim_content_words))
            return 0.5 * float(base_w) + 0.5 * overlap

        top_pooled_sup = sorted(
            pooled_sup,
            key=lambda it: _rank_candidate(it, sup_clusters_map),
            reverse=True,
        )[:4]
        for item in top_pooled_sup:
            windowed_premise = select_best_sentence_window(normalized_claim, item.text).text or item.text
            eval_pairs.append((normalized_claim, windowed_premise))
            rel_weight = (
                sup_clusters_map.get(str(item.evidence_id))
                or getattr(item, "cluster_reliability_weight", None)
                or getattr(item, "composite_score", 0.8)
                or 0.8
            )
            item_metadata.append({
                "item": item,
                "stance": "supporting",
                "reliability_weight": max(0.0, min(1.0, float(rel_weight))),
                "windowed_premise": windowed_premise,
            })

        # Process contradicting cluster representatives (pruned to top 3 for comprehensive coverage)
        con_clusters_map = {
            c.representative_id: c.cluster_reliability_weight
            for c in quality_eval.contradicting_report.clusters
        }
        top_pooled_con = sorted(
            pooled_con,
            key=lambda it: _rank_candidate(it, con_clusters_map),
            reverse=True,
        )[:3]
        for item in top_pooled_con:
            windowed_premise = select_best_sentence_window(normalized_claim, item.text).text or item.text
            eval_pairs.append((normalized_claim, windowed_premise))
            rel_weight = (
                con_clusters_map.get(str(item.evidence_id))
                or getattr(item, "cluster_reliability_weight", None)
                or getattr(item, "composite_score", 0.8)
                or 0.8
            )
            item_metadata.append({
                "item": item,
                "stance": "contradicting",
                "reliability_weight": max(0.0, min(1.0, float(rel_weight))),
                "windowed_premise": windowed_premise,
            })

        # Execute DeBERTa NLI cross-attention predictions
        if eval_pairs:
            nli_results: list[NLIResult] = engine.predict_batch(eval_pairs)
        else:
            nli_results = []

        # Step 4: Map NLI results to per-item records and collect distributions
        probs_list: list[dict[str, float]] = []
        weights_list: list[float] = []

        for meta, nli_res in zip(item_metadata, nli_results):
            raw_item = meta["item"]
            weight = meta["reliability_weight"]

            p_ent = nli_res.p_entailment
            p_con = nli_res.p_contradiction
            p_neu = nli_res.p_neutral

            premise_text = getattr(nli_res, "premise_text", None) or meta.get("windowed_premise") or raw_item.text
            tok_len = getattr(nli_res, "token_length", None)
            raw_tok_len = getattr(nli_res, "raw_token_length", None)
            is_trunc = getattr(nli_res, "truncated", False)

            evaluated_evidence.append(
                AdjudicationItemResult(
                    evidence_id=str(raw_item.evidence_id),
                    text=raw_item.text,
                    premise_text=premise_text,
                    token_length=tok_len,
                    raw_token_length=raw_tok_len,
                    truncated=is_trunc,
                    domain=getattr(raw_item, "domain", None),
                    source_url=getattr(raw_item, "source_url", None),
                    source_title=getattr(raw_item, "source_title", None),
                    stance=meta["stance"],
                    cluster_reliability_weight=round(weight, 4),
                    p_entailment=round(p_ent, 4),
                    p_contradiction=round(p_con, 4),
                    p_neutral=round(p_neu, 4),
                    label=nli_res.label.value if hasattr(nli_res.label, "value") else str(nli_res.label),
                    confidence=round(nli_res.confidence, 4),
                    inference_time_ms=round(nli_res.inference_time_ms, 2),
                )
            )

            probs_list.append({
                "entailment": p_ent,
                "contradiction": p_con,
                "neutral": p_neu,
            })
            weights_list.append(weight)

        # Step 5: Bayesian Probability Aggregation & Verdict Adjudication
        if not evaluated_evidence or sum(weights_list) <= 0.0:
            verdict = AdjudicationVerdict.NOT_ENOUGH_INFO
            confidence = 0.50
            post_sup = 0.33
            post_con = 0.33
            post_neu = 0.34
        else:
            combined = combine_nli_probabilities(
                probs_list=probs_list,
                weights=weights_list,
                prior_prob_support=0.5,
            )
            raw_p_sup = combined["posterior_support_probability"]
            raw_p_con = combined["posterior_contradiction_probability"]

            # Compute decisive evidence strength across items
            max_ent = max(ev.p_entailment for ev in evaluated_evidence)
            max_con = max(ev.p_contradiction for ev in evaluated_evidence)
            max_decisive = max(max_ent, max_con)

            # If evidence contains no decisive statements (all neutral/ambiguous), default to NOT_ENOUGH_INFO
            if max_decisive < 0.25:
                weighted_neu = max(0.60, min(ev.p_neutral for ev in evaluated_evidence))
            else:
                # Decisive grounding evidence exists: residual neutral uncertainty reflects uncaptured nuances
                weighted_neu = max(0.05, min(0.35, 1.0 - max_decisive))

            # Calibrate 3-class distribution
            w_sup = raw_p_sup * (1.0 - weighted_neu)
            w_con = raw_p_con * (1.0 - weighted_neu)
            w_neu = weighted_neu
            denom = w_sup + w_con + w_neu
            if denom > 0:
                post_sup = round(w_sup / denom, 4)
                post_con = round(w_con / denom, 4)
                post_neu = round(1.0 - (post_sup + post_con), 4)
            else:
                post_sup, post_con, post_neu = 0.33, 0.33, 0.34

            # Determine tri-class calibrated verdict with polar margin check
            if post_sup >= 0.50 and post_sup > post_con:
                verdict = AdjudicationVerdict.SUPPORTED
                confidence = post_sup
            elif post_con >= 0.50 and post_con > post_sup:
                verdict = AdjudicationVerdict.REFUTED
                confidence = post_con
            else:
                verdict = AdjudicationVerdict.NOT_ENOUGH_INFO
                confidence = max(post_neu, 0.50)

        # Collect all distinct retrieved internet / benchmark sources
        seen_source_keys: set[str] = set()
        retrieved_sources: list[dict[str, Any]] = []
        for it in (sup_items + con_items):
            url = getattr(it, "source_url", None)
            title = getattr(it, "source_title", None) or "Web Source"
            domain = getattr(it, "domain", None)
            if not domain and url:
                try:
                    domain = urlparse(url).netloc
                except Exception:
                    domain = "web"
            key = url if url else f"{title}_{domain}"
            if key not in seen_source_keys:
                seen_source_keys.add(key)
                retrieved_sources.append({
                    "url": url,
                    "title": title,
                    "domain": domain or "Web Source",
                    "snippet": (it.text[:220] + "...") if len(it.text) > 220 else it.text,
                    "retrieval_method": getattr(it, "retrieval_method", "federated_search"),
                })

        response = AdjudicationResponse(
            claim_id=resolved_claim_id,
            query_id=query_id,
            claim_text=normalized_claim,
            verdict=verdict,
            confidence_score=round(confidence, 4),
            support_probability=round(post_sup, 4),
            contradiction_probability=round(post_con, 4),
            neutral_probability=round(max(0.0, post_neu), 4),
            supporting_n_eff=n_eff_sup,
            contradicting_n_eff=n_eff_con,
            joint_n_eff=joint_n_eff,
            evaluated_evidence=evaluated_evidence,
            total_evidence_evaluated=len(evaluated_evidence),
            retrieved_sources=retrieved_sources,
            adjudicated_at=datetime.now(timezone.utc),
        )

        # Step 6: SQLite SessionStore Audit Persistence
        if query_id is not None:
            store = get_session_store()
            store.record_claim_adjudication(
                query_id=query_id,
                claim_id=resolved_claim_id,
                adjudication_data={
                    "verdict": verdict.value,
                    "confidence_score": round(confidence, 4),
                    "support_probability": round(post_sup, 4),
                    "contradiction_probability": round(post_con, 4),
                    "neutral_probability": round(max(0.0, post_neu), 4),
                    "supporting_n_eff": n_eff_sup,
                    "contradicting_n_eff": n_eff_con,
                    "joint_n_eff": joint_n_eff,
                    "total_evidence_evaluated": len(evaluated_evidence),
                    "evaluated_evidence": [ev.model_dump() for ev in evaluated_evidence],
                    "raw_nli_inputs": [
                        {
                            "evidence_id": ev.evidence_id,
                            "claim_text": normalized_claim,
                            "premise_text": ev.premise_text or ev.text,
                            "token_length": ev.token_length,
                            "raw_token_length": ev.raw_token_length,
                            "truncated": ev.truncated,
                            "stance": ev.stance,
                            "p_entailment": ev.p_entailment,
                            "p_neutral": ev.p_neutral,
                            "p_contradiction": ev.p_contradiction,
                            "label": ev.label,
                            "confidence": ev.confidence,
                        }
                        for ev in evaluated_evidence
                    ],
                    "adjudicated_at": response.adjudicated_at.isoformat(),
                },
            )

        return response

    def adjudicate_claims_batch(
        self,
        claims: Sequence[str | AdjudicationRequest | dict[str, Any]],
        query_id: UUID | None = None,
        batch_size: int = 2,
    ) -> list[AdjudicationResponse]:
        """Adjudicate a batch of claims using memory-safe sequential chunking.

        Slices requests into chunks <= batch_size to strictly respect RAM limits on Windows.
        """
        if not claims:
            return []

        effective_batch_size = max(1, batch_size)
        results: list[AdjudicationResponse] = []

        for i in range(0, len(claims), effective_batch_size):
            chunk = claims[i : i + effective_batch_size]
            for item in chunk:
                if isinstance(item, str):
                    c_text = item
                    c_id = None
                    q_id = query_id
                    sup_ev = None
                    con_ev = None
                elif isinstance(item, AdjudicationRequest):
                    c_text = item.claim_text
                    c_id = item.claim_id
                    q_id = item.query_id or query_id
                    sup_ev = item.supporting_evidence
                    con_ev = item.contradicting_evidence
                elif isinstance(item, dict):
                    c_text = item.get("claim_text") or item.get("text") or ""
                    c_id = item.get("claim_id") or item.get("id")
                    q_id = item.get("query_id") or query_id
                    sup_ev = item.get("supporting_evidence")
                    con_ev = item.get("contradicting_evidence")
                else:
                    c_text = getattr(item, "claim_text", "")
                    c_id = getattr(item, "claim_id", None)
                    q_id = getattr(item, "query_id", query_id)
                    sup_ev = getattr(item, "supporting_evidence", None)
                    con_ev = getattr(item, "contradicting_evidence", None)

                resp = self.adjudicate_claim(
                    claim_text=c_text,
                    claim_id=c_id,
                    query_id=q_id,
                    supporting_evidence=sup_ev,
                    contradicting_evidence=con_ev,
                )
                results.append(resp)

        return results


# Global singleton management
_global_adjudication_service: AdjudicationService | None = None
_global_adjudication_lock = threading.Lock()


def get_adjudication_service(
    nli_engine: NLIEngine | None = None,
    quality_service: EvidenceQualityService | None = None,
    retriever: BaseEvidenceRetriever | None = None,
    force_reload: bool = False,
) -> AdjudicationService:
    """Return thread-safe singleton instance of AdjudicationService."""
    global _global_adjudication_service
    if _global_adjudication_service is not None and not force_reload:
        return _global_adjudication_service

    with _global_adjudication_lock:
        if _global_adjudication_service is not None and not force_reload:
            return _global_adjudication_service
        _global_adjudication_service = AdjudicationService(
            nli_engine=nli_engine,
            quality_service=quality_service,
            retriever=retriever,
        )
        return _global_adjudication_service


def reset_adjudication_service() -> None:
    """Reset the singleton instance for test isolation."""
    global _global_adjudication_service
    with _global_adjudication_lock:
        _global_adjudication_service = None
