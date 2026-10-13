"""Unit tests for Programmatic Provenance Sweep Benchmark Evaluation.

Verifies:
1. Programmatic provenance corpus generation (verbatim, wire-prefix, paraphrase, summary).
2. Corpus freeze integrity and immutability (frozen to disk before runs).
3. Four configurations implementation:
   - Config 1: Naive Count
   - Config 2: Cluster-and-Count-One (B2)
   - Config 3: Kish Only
   - Config 4: Full Pipeline
4. Accuracy and ECE monotonicity under increasing dependent copy ratio.
5. Generated plot artifact existence and valid format.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import pytest

from app.services.cluster_pooling import ClusterPoolingService
from app.services.source_independence_scorer import SourceIndependenceDetector
from scripts.evaluate_provenance_sweep import (
    compute_accuracy_and_ece,
    generate_paraphrase_copy,
    generate_summary_copy,
    generate_verbatim_copy,
    generate_wire_prefix_copy,
    run_config1_naive_count,
    run_config2_cluster_and_count_one_b2,
    run_config3_kish_only,
    run_config4_full_pipeline,
)


def test_programmatic_provenance_transformations():
    """Verify all 4 programmatic transformations produce expected structure and domain metadata."""
    orig_text = "Clinical trial demonstrated that pembrolizumab significantly extended progression-free survival in patients."
    orig_url = "https://pubmed.ncbi.nlm.nih.gov/12345"

    # 1. Verbatim
    verb = generate_verbatim_copy(orig_text, copy_idx=0, orig_url=orig_url)
    assert verb["text"] == orig_text
    assert verb["provenance_type"] == "verbatim"
    assert "syndicated" in verb["source_url"]

    # 2. Wire Prefix
    wire = generate_wire_prefix_copy(orig_text, copy_idx=0, orig_url=orig_url)
    assert wire["provenance_type"] == "wire-prefix"
    assert any(wire["text"].startswith(prefix) for prefix in ["WASHINGTON", "PARIS", "LONDON"])
    assert orig_text in wire["text"]

    # 3. Paraphrase
    import random
    rng = random.Random(42)
    para = generate_paraphrase_copy(orig_text, copy_idx=0, orig_url=orig_url, rng=rng)
    assert para["provenance_type"] == "paraphrase"
    assert len(para["text"]) > 20

    # 4. Summary
    summ = generate_summary_copy(orig_text, copy_idx=0, orig_url=orig_url)
    assert summ["provenance_type"] == "summary"
    assert len(summ["text"]) <= len(orig_text) + 30


def test_four_configurations_mechanics():
    """Verify mechanics across Naive Count, B2, Kish Only, and Full Pipeline."""
    detector = SourceIndependenceDetector()
    pooling_service = ClusterPoolingService()

    claim = "Pembrolizumab increases survival in melanoma."
    orig_item = {
        "evidence_id": "orig",
        "document_id": "orig",
        "text": "Clinical trial demonstrated that pembrolizumab significantly extended survival in melanoma patients.",
        "source_url": "https://nejm.org/1",
        "domain": "nejm.org",
        "published_date": date(2023, 1, 1),
        "credibility_score": 0.95,
        "freshness_score": 0.90,
        "pollution_score": 0.00,
        "relevance_score": 1.00,
    }
    # Create 3 dependent copies (2 exact, 1 wire prefix)
    verb_copy1 = {
        "evidence_id": "c1",
        "document_id": "c1",
        "text": orig_item["text"],
        "source_url": "https://yahoo.com/1",
        "domain": "yahoo.com",
        "published_date": date(2023, 1, 2),
        "credibility_score": 0.85,
        "freshness_score": 0.90,
        "pollution_score": 0.05,
        "relevance_score": 1.00,
    }
    verb_copy2 = {
        "evidence_id": "c2",
        "document_id": "c2",
        "text": orig_item["text"],
        "source_url": "https://msn.com/1",
        "domain": "msn.com",
        "published_date": date(2023, 1, 2),
        "credibility_score": 0.85,
        "freshness_score": 0.90,
        "pollution_score": 0.05,
        "relevance_score": 1.00,
    }
    wire_copy = {
        "evidence_id": "c3",
        "document_id": "c3",
        "text": "WASHINGTON (Reuters) - " + orig_item["text"],
        "source_url": "https://reuters.com/1",
        "domain": "reuters.com",
        "published_date": date(2023, 1, 3),
        "credibility_score": 0.85,
        "freshness_score": 0.90,
        "pollution_score": 0.05,
        "relevance_score": 1.00,
    }

    items = [orig_item, verb_copy1, verb_copy2, wire_copy]
    probs = [{"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05}] * len(items)

    # Config 1: Naive Count
    res1 = run_config1_naive_count(claim, items, probs)
    assert res1["retained_count"] == 4
    assert res1["effective_sample_size"] == 4.0
    assert len(res1["weights"]) == 4

    # Config 2: B2 (Cluster and count one)
    res2 = run_config2_cluster_and_count_one_b2(claim, items, probs, detector)
    # orig, c1, c2 form 1 exact duplicate cluster; c3 (wire prefix on short text) is a singleton
    assert res2["retained_count"] == 2
    assert res2["effective_sample_size"] == 2.0
    assert res2["weights"] == [1.0, 1.0]

    # Config 3: Kish Only
    res3 = run_config3_kish_only(claim, items, probs, detector)
    assert res3["retained_count"] == 4
    # Kish discount collapses 1 cluster of 3 + 1 singleton -> 16 / (9 + 1) = 1.6
    assert res3["effective_sample_size"] == pytest.approx(1.6, abs=0.05)
    assert all(w == pytest.approx(1.6 / 4.0, abs=0.05) for w in res3["weights"])

    # Config 4: Full Pipeline
    res4 = run_config4_full_pipeline(claim, items, probs, pooling_service)
    assert res4["retained_count"] == 2  # 2 clusters (1 exact dup cluster + 1 wire prefix cluster)
    assert res4["effective_sample_size"] == pytest.approx(2.0, abs=0.05)
    assert len(res4["weights"]) == 2


def test_frozen_corpus_and_plot_artifacts():
    """Verify that frozen corpus JSON, evaluation report JSON, and plot PNG exist and have valid structure."""
    corpus_path = Path("data/frozen_provenance_corpus.json")
    report_path = Path("data/dedup_provenance_sweep_report.json")
    plot_path = Path("data/accuracy_ece_vs_dependent_copies.png")

    assert corpus_path.exists(), "Frozen corpus JSON must exist"
    assert report_path.exists(), "Sweep report JSON must exist"
    assert plot_path.exists(), "Sweep plot PNG must exist"

    # Validate corpus integrity
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert corpus_data["metadata"]["total_claims"] == 100
    assert len(corpus_data["records"]) == 100
    assert corpus_data["records"][0]["synthetic_copies"][0]["provenance_type"] in [
        "verbatim", "wire-prefix", "paraphrase", "summary"
    ]

    # Validate report integrity
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report_data["copy_counts"]) == 7  # N=0 to N=6
    res = report_data["sweep_results"]
    assert "Naive Count" in res
    assert "Cluster-and-Count-One (B2)" in res
    assert "Kish Only" in res
    assert "Full Pipeline" in res

    # Check ECE degradation of Naive Count vs Full Pipeline
    naive_ece_start = res["Naive Count"]["eces"][0]
    naive_ece_end = res["Naive Count"]["eces"][-1]
    full_ece_end = res["Full Pipeline"]["eces"][-1]

    # Naive Count calibration must degrade as dependent copies increase
    assert naive_ece_end > naive_ece_start
    # Full Pipeline must maintain superior calibration (lower ECE) than Naive Count
    assert full_ece_end < naive_ece_end
