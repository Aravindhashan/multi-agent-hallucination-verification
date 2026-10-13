"""Unit tests for Dedup-Only Baseline vs. Independence-Aware Aggregation Evaluation.

Verifies:
1. Exact-deduplication baseline mechanics (Config B).
2. Independence-aware complete-linkage clustering and Kish n_eff discounting.
3. Echo chamber overconfidence inflation detection (Exp 1).
4. Adversarial pollution rejection via 4-layer firewall (Exp 2).
5. Schema integrity of composite_score vs. cluster_reliability_weight.
"""

from __future__ import annotations

from datetime import date
import pytest

from app.schemas import EvidenceItem
from scripts.evaluate_dedup_vs_independence import (
    evaluate_adversarial_pollution_injection,
    evaluate_syndicated_echo_chamber,
    run_dedup_only_aggregation,
    run_independence_aware_aggregation,
)


def test_dedup_only_mechanics():
    """Verify Dedup-Only baseline handles exact matches and assigns unit weights."""
    claim = "Water boils at 100 degrees Celsius."
    items = [
        {"evidence_id": "1", "text": "Water boils at 100 C at standard pressure.", "source_url": "https://a.org/1"},
        {"evidence_id": "2", "text": "Water boils at 100 C at standard pressure.", "source_url": "https://a.org/1"},  # Exact dup
        {"evidence_id": "3", "text": "Water boils at 100 C at 1 atm.", "source_url": "https://b.org/2"},
    ]

    res = run_dedup_only_aggregation(claim, items)
    assert res["raw_count"] == 3
    assert res["retained_count"] == 2
    assert res["effective_sample_size"] == 2.0
    assert res["nli_calls_required"] == 2
    assert res["weights"] == [1.0, 1.0]


def test_syndicated_echo_chamber_experiment():
    """Verify Exp 1 demonstrates overconfidence inflation on syndicated wire copies."""
    exp1 = evaluate_syndicated_echo_chamber()
    assert exp1["raw_evidence_count"] == 5
    assert exp1["dedup_only"]["retained_count"] == 5
    assert exp1["dedup_only"]["effective_sample_size"] == 5.0
    assert exp1["independence_aware"]["admissible_clusters"] == 2
    assert exp1["independence_aware"]["effective_sample_size"] == pytest.approx(1.47, abs=0.05)
    assert exp1["overconfidence_inflation_ratio"] >= 3.0
    assert exp1["nli_call_savings_percent"] == 60.0


def test_adversarial_pollution_injection_experiment():
    """Verify Exp 2 protects verdict against syndicated disinformation injection."""
    exp2 = evaluate_adversarial_pollution_injection()
    assert exp2["raw_evidence_count"] == 5
    assert exp2["clean_items"] == 2
    assert exp2["polluted_injected_items"] == 3

    # Dedup-only is corrupted by majority vote
    assert "REFUTED" in exp2["dedup_only"]["verdict"]
    assert exp2["dedup_only"]["posterior_support_probability"] < 0.05

    # Independence-aware firewall rejects the polluted cluster
    assert exp2["independence_aware"]["polluted_clusters"] == 1
    assert exp2["independence_aware"]["admissible_clusters"] == 2
    assert "SUPPORTED" in exp2["independence_aware"]["verdict"]
    assert exp2["independence_aware"]["posterior_support_probability"] > 0.50


def test_evidence_item_score_invariance():
    """Verify EvidenceItem schema cleanly differentiates composite_score and cluster_reliability_weight."""
    item = EvidenceItem(
        evidence_id="ev_test",
        text="A test scientific passage.",
        relevance_score=0.95,
        credibility_score=0.85,
        freshness_score=0.90,
        composite_score=0.895,  # Additive retrieval heuristic
        cluster_reliability_weight=0.765,  # Multiplicative channel reliability
    )

    assert item.composite_score == 0.895
    assert item.cluster_reliability_weight == 0.765
    dump = item.model_dump()
    assert "composite_score" in dump
    assert "cluster_reliability_weight" in dump
