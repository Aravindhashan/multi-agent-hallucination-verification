"""Master End-to-End Pipeline Orchestration Service (Module 11 & Module 10 Rollup).

Sequences:
1. Input sanitization & session initiation (Module 1)
2. Atomic claim decomposition & qualifier preservation (Module 3)
3. Complexity analysis & verification planning (Module 4)
4. Federated multi-turn debate (Module 7)
5. Evidence quality firewall, clustering, & Kish n_eff discounting (Module 8)
6. Winning cluster representative pooling (Module 8.7 & 8.8)
7. DeBERTa-v3 cross-encoder NLI inference (Module 9)
8. Calibrated claim adjudication & response-level rollup with global hallucination index (Module 10)
9. SQLite session audit finalization (Module 11.2)
"""

import concurrent.futures
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Sequence
from uuid import UUID, uuid4

from app.config import settings
from app.schemas import (
    AdjudicationVerdict,
    Claim,
    ClaimVerificationSummary,
    EvidenceSnippetSummary,
    ResponseVerdict,
    VerificationBatchRequest,
    VerificationBatchResponse,
    VerificationRequest,
    VerificationResponse,
)
from app.services.adjudication_service import (
    AdjudicationService,
    get_adjudication_service,
)
from app.services.claim_extraction_service import extract_claims
from app.services.federated_evidence_retriever import (
    FederatedEvidenceRetriever,
    get_federated_evidence_retriever,
)
from app.services.session_store import PipelineSessionStore, get_session_store
from app.services.simple_claim_analyzer import SimpleClaimAnalyzer

logger = logging.getLogger(__name__)


class VerificationService:
    """Master orchestrator for end-to-end hallucination verification."""

    def __init__(
        self,
        adjudication_service: AdjudicationService | None = None,
        session_store: PipelineSessionStore | None = None,
        default_evidence_mode: str | None = None,
    ) -> None:
        self._adjudication_service = adjudication_service
        self._session_store = session_store or get_session_store()
        self._default_evidence_mode = default_evidence_mode
        self._lock = threading.RLock()

    def _resolve_adjudication_service(
        self, evidence_mode: str | None = None
    ) -> AdjudicationService:
        """Resolve or configure an AdjudicationService matching the requested evidence mode."""
        requested_mode = (
            evidence_mode
            or self._default_evidence_mode
            or getattr(settings, "EVIDENCE_MODE", "hybrid")
        ).strip().lower()

        # If custom adjudication service was passed in constructor, preserve it and its injected components
        if self._adjudication_service is not None:
            if evidence_mode is None or (
                self._adjudication_service._retriever is not None
                and getattr(self._adjudication_service._retriever, "mode", None) == requested_mode
            ):
                return self._adjudication_service
            retriever = FederatedEvidenceRetriever(mode=requested_mode)
            return AdjudicationService(
                nli_engine=self._adjudication_service._nli_engine,
                quality_service=self._adjudication_service._quality_service,
                retriever=retriever,
            )

        base_service = get_adjudication_service()
        if (
            base_service._retriever is not None
            and getattr(base_service._retriever, "mode", None) == requested_mode
        ):
            return base_service

        # Instantiate federated retriever configured with the requested mode while reusing base services
        retriever = FederatedEvidenceRetriever(mode=requested_mode)
        return AdjudicationService(
            nli_engine=base_service._nli_engine,
            quality_service=base_service._quality_service,
            retriever=retriever,
        )

    def verify_text(
        self,
        text: str,
        query_id: UUID | None = None,
        evidence_mode: str | None = None,
        extraction_method: str = "nlp",
        max_debate_rounds: int = 2,
    ) -> VerificationResponse:
        """Execute the full end-to-end verification pipeline over a statement or passage.

        Args:
            text: Source text passage or statement to verify.
            query_id: Optional query identifier for session tracking.
            evidence_mode: Evidence retrieval mode ('live_web', 'benchmark_scifact', 'benchmark_fever', 'hybrid', 'toy').
            extraction_method: Extraction method ('nlp' or 'llm').
            max_debate_rounds: Maximum debate rounds between Defender and Prosecutor.

        Returns:
            VerificationResponse with claim breakdowns, response rollup, and global hallucination index.
        """
        start_time = time.perf_counter()
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Text must contain at least one non-whitespace character.")

        resolved_query_id = query_id if query_id is not None else uuid4()
        active_mode = (
            evidence_mode
            or self._default_evidence_mode
            or getattr(settings, "EVIDENCE_MODE", "hybrid")
        ).strip().lower()

        # Step 1: Session Initiation (Module 1)
        self._session_store.create_session(
            query_id=resolved_query_id,
            query=normalized_text,
        )

        # Step 2: Atomic Claim Decomposition & Qualifier Preservation (Module 3)
        extraction_resp = extract_claims(
            query_id=resolved_query_id,
            response_text=normalized_text,
            query=normalized_text,
            method=extraction_method,
        )
        claims: list[Claim] = extraction_resp.claims
        decomp_metrics = extraction_resp.decomposition_metrics

        # Persist extracted claims in SQLite session store
        self._session_store.record_claims(
            query_id=resolved_query_id,
            claims=claims,
        )

        # Edge case: No verifiable claims extracted
        if not claims:
            exec_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            summary_dict = {
                "overall_verdict": ResponseVerdict.NOT_ENOUGH_INFO.value,
                "hallucination_index": 0.0,
                "unverified_index": 1.0,
                "confidence_score": 0.0,
                "claim_count": 0,
                "supported_count": 0,
                "refuted_count": 0,
                "unverified_count": 0,
                "evidence_mode_used": active_mode,
                "execution_time_ms": exec_ms,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._session_store.update_metadata(
                query_id=resolved_query_id,
                key="verification_result",
                value=summary_dict,
            )
            return VerificationResponse(
                query_id=resolved_query_id,
                text=normalized_text,
                overall_verdict=ResponseVerdict.NOT_ENOUGH_INFO,
                hallucination_index=0.0,
                unverified_index=1.0,
                confidence_score=0.0,
                claim_count=0,
                supported_count=0,
                refuted_count=0,
                unverified_count=0,
                claims=[],
                decomposition_metrics=decomp_metrics,
                total_evidence_evaluated=0,
                aggregate_joint_n_eff=0.0,
                evidence_mode_used=active_mode,
                execution_time_ms=exec_ms,
            )

        # Step 3: Semantic Complexity Analysis & Verification Planning (Module 4)
        analyzer = SimpleClaimAnalyzer()
        claim_analyses = {}
        for claim in claims:
            analysis = analyzer.analyze(claim.text, atomic_claim=claim)
            claim_analyses[str(claim.claim_id)] = analysis
            self._session_store.record_claim_analysis(
                query_id=resolved_query_id,
                claim_id=claim.claim_id,
                claim_text=claim.text,
                complexity_score=analysis.complexity_score,
                verification_plan=analysis.verification_plan,
                requires_deep_verification=analysis.requires_deep_verification,
                ablation_log=analysis.ablation_log.model_dump() if analysis.ablation_log else None,
            )

        # Step 4: Adjudication Engine Resolution
        adjudication_svc = self._resolve_adjudication_service(evidence_mode=evidence_mode)

        # Step 5: Multi-Turn Federated Debate, Quality Firewall, & Cross-Encoder Adjudication
        def _adjudicate_single(target_claim: Claim) -> tuple[Claim, Any]:
            res = adjudication_svc.adjudicate_claim(
                claim_text=target_claim.text,
                claim_id=target_claim.claim_id,
                query_id=resolved_query_id,
                run_debate=True,
                max_debate_rounds=max_debate_rounds,
            )
            return target_claim, res

        # Execute claim adjudication concurrently across worker threads
        if len(claims) > 1:
            max_workers = min(3, len(claims))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                adjudication_results = list(executor.map(_adjudicate_single, claims))
        else:
            adjudication_results = [_adjudicate_single(claims[0])]

        claim_summaries: list[ClaimVerificationSummary] = []
        total_evidence_evaluated = 0
        sum_joint_n_eff = 0.0

        for claim, adj_resp in adjudication_results:
            analysis = claim_analyses.get(str(claim.claim_id))
            comp_score = analysis.complexity_score if analysis else 0.0

            # Persist per-claim adjudication result to SQLite session store
            self._session_store.record_claim_adjudication(
                query_id=resolved_query_id,
                claim_id=claim.claim_id,
                adjudication_data=adj_resp.model_dump(),
            )

            total_evidence_evaluated += adj_resp.total_evidence_evaluated
            sum_joint_n_eff += adj_resp.joint_n_eff

            # Extract top winning representative snippets
            winning_snippets = [
                ev.text for ev in adj_resp.evaluated_evidence[:3] if ev.text
            ]

            # Separate evidence into supportive and opposing claims
            supportive_claims: list[EvidenceSnippetSummary] = []
            opposing_claims: list[EvidenceSnippetSummary] = []
            raw_nli_inputs_list: list[dict[str, Any]] = []

            for ev in adj_resp.evaluated_evidence:
                p_c = ev.p_contradiction or 0.0
                p_e = ev.p_entailment or 0.0
                is_opposing = (p_c >= 0.40 and p_c > p_e)
                is_support = not is_opposing and ((ev.stance == "supporting") or (p_e >= 0.40))
                conf = p_c if is_opposing else p_e
                snippet_item = EvidenceSnippetSummary(
                    text=ev.text,
                    premise_text=getattr(ev, "premise_text", None) or ev.text,
                    token_length=getattr(ev, "token_length", None),
                    p_entailment=round(p_e, 4),
                    p_neutral=round(ev.p_neutral or 0.0, 4),
                    p_contradiction=round(p_c, 4),
                    source_title=ev.source_title or "Internet Reference",
                    source_url=ev.source_url,
                    domain=ev.domain or "Web Source",
                    stance="opposing" if is_opposing else "supporting",
                    confidence=round(conf, 4) if conf is not None else 0.5,
                )
                if is_opposing:
                    opposing_claims.append(snippet_item)
                elif is_support:
                    supportive_claims.append(snippet_item)

                raw_nli_inputs_list.append({
                    "evidence_id": ev.evidence_id,
                    "stance": ev.stance,
                    "premise_text": getattr(ev, "premise_text", None) or ev.text,
                    "token_length": getattr(ev, "token_length", None),
                    "raw_token_length": getattr(ev, "raw_token_length", None),
                    "truncated": getattr(ev, "truncated", False),
                    "p_entailment": ev.p_entailment,
                    "p_neutral": ev.p_neutral,
                    "p_contradiction": ev.p_contradiction,
                    "label": ev.label,
                    "confidence": ev.confidence,
                })

            # Collect internet sources for this claim
            claim_sources: list[dict[str, Any]] = []
            seen_src: set[str] = set()

            for s in getattr(adj_resp, "retrieved_sources", []):
                key = s.get("url") or s.get("title")
                if key and key not in seen_src:
                    seen_src.add(key)
                    claim_sources.append(s)

            # Ensure any evaluated evidence sources are also represented in sources
            for ev in adj_resp.evaluated_evidence:
                key = ev.source_url or ev.source_title
                if key and key not in seen_src:
                    seen_src.add(key)
                    claim_sources.append({
                        "url": ev.source_url,
                        "title": ev.source_title or "Web Reference",
                        "domain": ev.domain or "Web Source",
                        "snippet": (ev.text[:180] + "...") if len(ev.text) > 180 else ev.text,
                        "retrieval_method": "evaluated_evidence",
                    })

            claim_summaries.append(
                ClaimVerificationSummary(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    order=claim.order,
                    verdict=adj_resp.verdict,
                    confidence_score=adj_resp.confidence_score,
                    support_probability=adj_resp.support_probability,
                    contradiction_probability=adj_resp.contradiction_probability,
                    neutral_probability=adj_resp.neutral_probability,
                    complexity_score=comp_score,
                    temporal_anchors=claim.temporal_anchors,
                    numerical_values=claim.numerical_values,
                    spatial_anchors=claim.spatial_anchors,
                    supporting_n_eff=adj_resp.supporting_n_eff,
                    contradicting_n_eff=adj_resp.contradicting_n_eff,
                    joint_n_eff=adj_resp.joint_n_eff,
                    winning_evidence_snippets=winning_snippets,
                    supportive_claims=supportive_claims,
                    opposing_claims=opposing_claims,
                    internet_sources=claim_sources,
                    raw_nli_inputs=raw_nli_inputs_list,
                    consensus_confirmed=(len(opposing_claims) == 0 and adj_resp.verdict == AdjudicationVerdict.SUPPORTED),
                )
            )

        # Step 6: Calibrated Response-Level Rollup & Hallucination Index Calculation (Module 10)
        n_claims = len(claim_summaries)
        supported_count = sum(
            1 for c in claim_summaries if c.verdict == AdjudicationVerdict.SUPPORTED
        )
        refuted_count = sum(
            1 for c in claim_summaries if c.verdict == AdjudicationVerdict.REFUTED
        )
        unverified_count = sum(
            1 for c in claim_summaries if c.verdict == AdjudicationVerdict.NOT_ENOUGH_INFO
        )

        # Response Verdict taxonomy & Granular Epistemic Category
        if refuted_count > 0 and supported_count > 0:
            overall_verdict = ResponseVerdict.PARTIALLY_SUPPORTED
            verdict_category = "PARTIALLY_SUPPORTED_CONTRADICTED"
        elif refuted_count > 0 and supported_count == 0:
            overall_verdict = ResponseVerdict.REFUTED
            verdict_category = "REFUTED"
        elif supported_count > 0 and refuted_count == 0:
            if unverified_count > 0:
                overall_verdict = ResponseVerdict.PARTIALLY_SUPPORTED
                verdict_category = "PARTIALLY_VERIFIED_COVERAGE_GAP"
            else:
                overall_verdict = ResponseVerdict.SUPPORTED
                verdict_category = "SUPPORTED"
        else:
            overall_verdict = ResponseVerdict.NOT_ENOUGH_INFO
            verdict_category = "NOT_ENOUGH_INFO"

        # Decoupled Mathematical Formulation:
        # 1. Factual Hallucination Index H_refuted in [0.0, 1.0]:
        #    Tracks strictly proven refutations / contradictory fabrications.
        #    Supported claims contribute 0.0 risk.
        #    Refuted claims contribute risk weighted by contradiction confidence.
        #    Inconclusive / NOT_ENOUGH_INFO claims contribute 0.0 refutation risk
        #    (absence of evidence is NOT evidence of hallucination).
        total_refuted_risk = 0.0

        # 2. Epistemic Uncertainty / Coverage Gap Index U_unverified in [0.0, 1.0]:
        #    Tracks claims where external evidence was absent, incomplete, or inconclusive.
        #    Refuted and Supported claims contribute 0.0 uncertainty.
        #    NOT_ENOUGH_INFO claims contribute uncertainty weighted by neutral probability.
        total_unverified_risk = 0.0

        for c in claim_summaries:
            if c.verdict == AdjudicationVerdict.REFUTED:
                total_refuted_risk += max(0.75, c.confidence_score)
            elif c.verdict == AdjudicationVerdict.NOT_ENOUGH_INFO:
                total_unverified_risk += max(0.50, c.neutral_probability)
            else:
                # SUPPORTED claims contribute 0.0 to both refutation and uncertainty
                pass

        hallucination_index = round(min(1.0, max(0.0, total_refuted_risk / n_claims)), 4)
        unverified_index = round(min(1.0, max(0.0, total_unverified_risk / n_claims)), 4)
        mean_confidence = round(
            sum(c.confidence_score for c in claim_summaries) / n_claims, 4
        )

        exec_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Step 7: SQLite Session Audit Finalization (Module 11.2)
        summary_dict = {
            "overall_verdict": overall_verdict.value,
            "verdict_category": verdict_category,
            "hallucination_index": hallucination_index,
            "unverified_index": unverified_index,
            "confidence_score": mean_confidence,
            "claim_count": n_claims,
            "supported_count": supported_count,
            "refuted_count": refuted_count,
            "unverified_count": unverified_count,
            "total_evidence_evaluated": total_evidence_evaluated,
            "aggregate_joint_n_eff": round(sum_joint_n_eff, 4),
            "evidence_mode_used": active_mode,
            "execution_time_ms": exec_ms,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._session_store.update_metadata(
            query_id=resolved_query_id,
            key="verification_result",
            value=summary_dict,
        )
        self._session_store.update_metadata(
            query_id=resolved_query_id,
            key="overall_verdict",
            value=overall_verdict.value,
        )
        self._session_store.update_metadata(
            query_id=resolved_query_id,
            key="verdict_category",
            value=verdict_category,
        )
        self._session_store.update_metadata(
            query_id=resolved_query_id,
            key="hallucination_index",
            value=hallucination_index,
        )
        self._session_store.update_metadata(
            query_id=resolved_query_id,
            key="unverified_index",
            value=unverified_index,
        )

        # Aggregate all internet sources across claims
        all_internet_sources: list[dict[str, Any]] = []
        global_seen_sources: set[str] = set()
        for cs in claim_summaries:
            for src in cs.internet_sources:
                key = src.get("url") or src.get("title")
                if key and key not in global_seen_sources:
                    global_seen_sources.add(key)
                    all_internet_sources.append(src)

        return VerificationResponse(
            query_id=resolved_query_id,
            text=normalized_text,
            overall_verdict=overall_verdict,
            verdict_category=verdict_category,
            hallucination_index=hallucination_index,
            unverified_index=unverified_index,
            confidence_score=mean_confidence,
            claim_count=n_claims,
            supported_count=supported_count,
            refuted_count=refuted_count,
            unverified_count=unverified_count,
            claims=claim_summaries,
            decomposition_metrics=decomp_metrics,
            total_evidence_evaluated=total_evidence_evaluated,
            aggregate_joint_n_eff=round(sum_joint_n_eff, 4),
            evidence_mode_used=active_mode,
            execution_time_ms=exec_ms,
            internet_sources=all_internet_sources,
        )

    def verify_claim(
        self,
        claim_text: str,
        query_id: UUID | None = None,
        evidence_mode: str | None = None,
        max_debate_rounds: int = 2,
    ) -> VerificationResponse:
        """Convenience method to verify a single claim proposition."""
        return self.verify_text(
            text=claim_text,
            query_id=query_id,
            evidence_mode=evidence_mode,
            extraction_method="nlp",
            max_debate_rounds=max_debate_rounds,
        )

    def verify_batch(
        self,
        requests: Sequence[str | VerificationRequest | dict],
        batch_size: int = 2,
    ) -> VerificationBatchResponse:
        """Process batch text verification requests with memory-safe chunking.

        Slices requests into chunks <= batch_size to strictly safeguard RAM
        and prevent Windows virtual memory paging file exhaustion (OS error 1455).

        Args:
            requests: List of text strings or VerificationRequest objects.
            batch_size: Maximum chunk size (default 2).

        Returns:
            VerificationBatchResponse with ordered responses matching input sequence.
        """
        if not requests:
            return VerificationBatchResponse(
                responses=[],
                total_processed=0,
                mean_hallucination_index=0.0,
                mean_confidence=0.0,
            )

        effective_batch_size = max(1, batch_size)
        results: list[VerificationResponse] = []

        for i in range(0, len(requests), effective_batch_size):
            chunk = requests[i : i + effective_batch_size]
            for item in chunk:
                if isinstance(item, str):
                    resp = self.verify_text(text=item)
                elif isinstance(item, VerificationRequest):
                    resp = self.verify_text(
                        text=item.text,
                        query_id=item.query_id,
                        evidence_mode=item.evidence_mode,
                        extraction_method=item.extraction_method,
                        max_debate_rounds=item.max_debate_rounds,
                    )
                elif isinstance(item, dict):
                    resp = self.verify_text(
                        text=item.get("text") or item.get("claim_text") or "",
                        query_id=item.get("query_id"),
                        evidence_mode=item.get("evidence_mode"),
                        extraction_method=item.get("extraction_method", "nlp"),
                        max_debate_rounds=item.get("max_debate_rounds", 2),
                    )
                else:
                    text_val = getattr(item, "text", getattr(item, "claim_text", ""))
                    q_id = getattr(item, "query_id", None)
                    ev_mode = getattr(item, "evidence_mode", None)
                    resp = self.verify_text(text=text_val, query_id=q_id, evidence_mode=ev_mode)
                results.append(resp)

        total_processed = len(results)
        mean_h_index = (
            round(sum(r.hallucination_index for r in results) / total_processed, 4)
            if total_processed > 0
            else 0.0
        )
        mean_u_index = (
            round(sum(r.unverified_index for r in results) / total_processed, 4)
            if total_processed > 0
            else 0.0
        )
        mean_conf = (
            round(sum(r.confidence_score for r in results) / total_processed, 4)
            if total_processed > 0
            else 0.0
        )

        return VerificationBatchResponse(
            responses=results,
            total_processed=total_processed,
            mean_hallucination_index=mean_h_index,
            mean_unverified_index=mean_u_index,
            mean_confidence=mean_conf,
        )


# Singleton instance management
_global_verification_service: VerificationService | None = None
_global_verification_lock = threading.Lock()


def get_verification_service(
    adjudication_service: AdjudicationService | None = None,
    session_store: PipelineSessionStore | None = None,
    default_evidence_mode: str | None = None,
    force_reload: bool = False,
) -> VerificationService:
    """Return thread-safe singleton instance of VerificationService."""
    global _global_verification_service
    if _global_verification_service is not None and not force_reload:
        return _global_verification_service

    with _global_verification_lock:
        if _global_verification_service is not None and not force_reload:
            return _global_verification_service
        _global_verification_service = VerificationService(
            adjudication_service=adjudication_service,
            session_store=session_store,
            default_evidence_mode=default_evidence_mode,
        )
        return _global_verification_service


def reset_verification_service() -> None:
    """Reset the singleton instance for test isolation."""
    global _global_verification_service
    with _global_verification_lock:
        _global_verification_service = None
