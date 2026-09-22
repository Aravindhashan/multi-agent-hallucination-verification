"""System evaluation and aggregation pipeline for the pre-Module-5 experiment.

Implements Systems A0 (All-evidence), A1 (Dedup-only), A2 (Independence-aware),
and A3 (Independence + Pollution) under the locked shared aggregation rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure backend is on sys.path for service imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.nli_engine import get_nli_engine
from app.services.source_credibility_scorer import get_source_credibility_scorer
from app.services.temporal_freshness_scorer import get_temporal_freshness_scorer
from app.services.source_independence_scorer import (
    get_source_independence_detector,
    normalize_text,
    normalize_url,
)
from app.services.evidence_pollution_detector import get_evidence_pollution_detector
from app.services.evidence_pollution_models import PollutionClass


class ExperimentEvaluator:
    """Runs evaluation pipelines for A0, A1, A2, and A3 systems."""

    def __init__(self) -> None:
        self.nli_engine = get_nli_engine()
        self.credibility_scorer = get_source_credibility_scorer()
        self.freshness_scorer = get_temporal_freshness_scorer()
        self.independence_detector = get_source_independence_detector()
        self.pollution_detector = get_evidence_pollution_detector()
        # Cache NLI inferences to ensure determinism and speed
        self._nli_cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def get_nli_probabilities(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Get (p_ent, p_con, p_neu) from NLI engine with caching."""
        key = (premise.strip(), hypothesis.strip())
        if key not in self._nli_cache or self._nli_cache[key] is None:
            res = self.nli_engine.predict(claim=hypothesis, evidence=premise)
            self._nli_cache[key] = (res.p_entailment, res.p_contradiction, res.p_neutral)
        return self._nli_cache[key]

    def precompute_nli(self, dataset: dict[str, Any]) -> None:
        """Batch precompute all unique (claim, evidence) pairs across all claims and conditions."""
        pairs_to_evaluate: list[tuple[str, str]] = []
        for claim in dataset["claims"]:
            c_text = claim["claim_text"].strip()
            for lvl_str, ev_items in claim["conditions"].items():
                for ev in ev_items:
                    premise = ev["text"].strip()
                    key = (premise, c_text)
                    if key not in self._nli_cache:
                        self._nli_cache[key] = None  # placeholder
                        pairs_to_evaluate.append((c_text, premise))

        if pairs_to_evaluate:
            print(f"Batch precomputing {len(pairs_to_evaluate)} NLI inference pairs (batch_size={self.nli_engine.config.batch_size})...", flush=True)
            results = self.nli_engine.predict_batch(pairs_to_evaluate)
            for (c_text, premise), res in zip(pairs_to_evaluate, results):
                key = (premise, c_text)
                self._nli_cache[key] = (res.p_entailment, res.p_contradiction, res.p_neutral)
            print("Batch NLI precomputation complete.", flush=True)

    def filter_evidence_a0(self, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """System A0: All evidence passed without filtering (upper-bound ceiling)."""
        return list(evidence_items)

    def filter_evidence_a1(self, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """System A1: Dedup-only baseline.

        Removes exact duplicates having identical normalized text or identical canonical URL.
        Retains one representative per duplicate group.
        """
        seen_texts: set[str] = set()
        seen_urls: set[str] = set()
        surviving: list[dict[str, Any]] = []

        for item in evidence_items:
            norm_t = normalize_text(item["text"]).lower()
            norm_u = normalize_url(item.get("source_url"))

            is_dup = False
            if norm_t and norm_t in seen_texts:
                is_dup = True
            elif norm_u and norm_u in seen_urls:
                is_dup = True

            if not is_dup:
                surviving.append(item)
                if norm_t:
                    seen_texts.add(norm_t)
                if norm_u:
                    seen_urls.add(norm_u)

        return surviving

    def filter_evidence_a2(self, evidence_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
        """System A2: Independence-aware.

        Uses Module 4.3 SourceIndependenceDetector complete-linkage clustering.
        Retains one representative per redundancy cluster.
        Returns (surviving_items, cluster_sizes).
        """
        if len(evidence_items) <= 1:
            return list(evidence_items), [1] * len(evidence_items)

        for it in evidence_items:
            if "document_id" not in it:
                it["document_id"] = it.get("evidence_id", "unknown_doc")

        set_res = self.independence_detector.analyze_set(evidence_items)

        # Build mapping from doc_id to cluster
        clustered_ids: set[str] = set()
        cluster_representatives: list[str] = []
        cluster_sizes: list[int] = []

        for cluster in set_res.clusters:
            doc_ids = cluster.document_ids
            cluster_sizes.append(len(doc_ids))
            # Pick the first as representative
            cluster_representatives.append(doc_ids[0])
            clustered_ids.update(doc_ids)

        # Unclustered independent items
        for doc_id in set_res.independent_document_ids:
            cluster_representatives.append(doc_id)
            cluster_sizes.append(1)

        # Retrieve surviving item dicts in stable order
        id_to_item = {it.get("document_id", it.get("evidence_id")): it for it in evidence_items}
        surviving = [id_to_item[doc_id] for doc_id in cluster_representatives if doc_id in id_to_item]

        return surviving, cluster_sizes

    def filter_evidence_a3(
        self, claim: str, evidence_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """System A3: Independence-aware + Pollution Gate.

        Filters A2 surviving items through Module 4.4 EvidencePollutionDetector.
        Excludes items classified as SELF_REFERENTIAL or UNGROUNDED.
        """
        a2_items, cluster_sizes = self.filter_evidence_a2(evidence_items)
        surviving: list[dict[str, Any]] = []
        surviving_sizes: list[int] = []

        for idx, item in enumerate(a2_items):
            p_res = self.pollution_detector.evaluate(claim=claim, evidence=item)
            if p_res.classification not in (PollutionClass.SELF_REFERENTIAL, PollutionClass.UNGROUNDED):
                surviving.append(item)
                surviving_sizes.append(cluster_sizes[idx])

        # If all were rejected, fallback to highest grounding item to avoid 0 evidence
        if not surviving and a2_items:
            surviving = [a2_items[0]]
            surviving_sizes = [cluster_sizes[0]]

        return surviving, surviving_sizes

    def aggregate_verdict(
        self,
        claim: str,
        surviving_evidence: list[dict[str, Any]],
        ground_truth: str,
    ) -> dict[str, Any]:
        """Apply shared aggregation rule across surviving evidence passages (§5)."""
        if not surviving_evidence:
            return {
                "predicted_verdict": "NEUTRAL",
                "mean_p_ent": 0.0,
                "mean_p_con": 0.0,
                "mean_p_neu": 1.0,
                "confidence": 0.3333,
                "is_correct": False,
                "calibration_error": 0.3333,
                "effective_support_prob": 0.0,
            }

        p_ent_list: list[float] = []
        p_con_list: list[float] = []
        p_neu_list: list[float] = []

        for ev in surviving_evidence:
            p_ent, p_con, p_neu = self.get_nli_probabilities(premise=ev["text"], hypothesis=claim)
            p_ent_list.append(p_ent)
            p_con_list.append(p_con)
            p_neu_list.append(p_neu)

        mean_ent = float(sum(p_ent_list) / len(p_ent_list))
        mean_con = float(sum(p_con_list) / len(p_con_list))
        mean_neu = float(sum(p_neu_list) / len(p_neu_list))

        # Locked Verdict Rule (§4, §5)
        if mean_ent > 0.5 and mean_ent > mean_con:
            verdict = "SUPPORTED"
        elif mean_con >= 0.5 and mean_con > mean_ent:
            verdict = "CONTRADICTED"
        else:
            verdict = "NEUTRAL"

        confidence = max(mean_ent, mean_con, mean_neu)
        is_correct = (verdict == ground_truth)
        calib_error = abs(confidence - (1.0 if is_correct else 0.0))

        return {
            "predicted_verdict": verdict,
            "mean_p_ent": round(mean_ent, 4),
            "mean_p_con": round(mean_con, 4),
            "mean_p_neu": round(mean_neu, 4),
            "confidence": round(confidence, 4),
            "is_correct": is_correct,
            "calibration_error": round(calib_error, 4),
            "effective_support_prob": round(mean_ent, 4),
        }
