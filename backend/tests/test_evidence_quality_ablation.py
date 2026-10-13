"""Tests for Evidence Quality Ablation Harness and FEVER/SciFact Calibration Checkpoint (Submodule 8.6).

Covers:
1. Four-Way Ablation Benchmark (Config A, B, B2, C) on multi-source syndicated evidence.
2. Direct comparison between Config B2 (naive flat cluster count) and Config C (Module 8 Kish n_eff).
3. Empirical NLI call reduction measurement (confirming 40-60% efficiency gain).
4. Pollution curve study (monotonicity across 0% to 80% corruption and >95% clean retention).
5. FEVER gold corpus calibration checkpoint (validating all 4 provisional gates).
6. SciFact gold corpus calibration checkpoint (biomedical domain validation).
7. Provisional constants audit checklist.
8. Error handling and boundary conditions.
"""

from datetime import date
import pytest

from app.config import settings
from app.services.evidence_quality_ablation import (
    AblationConfig,
    EvidenceQualityAblationHarness,
    get_evidence_quality_ablation_harness,
)
from app.services.evidence_pollution_config import (
    POLLUTION_BOILERPLATE_THRESHOLD,
    POLLUTION_CIRCULARITY_THRESHOLD,
    POLLUTION_CLEAN_RETENTION_TARGET,
    POLLUTION_CLEAN_THRESHOLD,
    POLLUTION_LOW_GROUNDING_THRESHOLD,
)
from app.services.source_independence_config import DEFAULT_NEAR_DUPLICATE_THRESHOLD
from app.services.temporal_freshness_config import (
    DEFAULT_CONTEMPORANEOUS_WINDOW_DAYS,
    DEFAULT_HISTORICAL_HALF_LIFE_DAYS,
)


@pytest.fixture
def harness():
    return EvidenceQualityAblationHarness()


def test_four_way_ablation_hierarchy(harness):
    """Verify 4-way ablation behavior across Config A, B, B2, and C on syndicated evidence."""
    claim = "The speed of light in vacuum is exactly 299,792,458 metres per second."

    evidence_items = [
        # Cluster 1: 3 syndicated copies of the same wire article with slight rephrasings
        {
            "evidence_id": "wire_ap",
            "text": "The speed of light in vacuum is defined as exactly 299,792,458 metres per second by international agreement.",
            "source_url": "https://apnews.example.com/physics/speed-of-light",
            "domain": "apnews.example.com",
            "published_date": date(2023, 1, 1),
        },
        {
            "evidence_id": "wire_reuters",
            "text": "The speed of light in vacuum is defined as exactly 299,792,458 metres per second by international consensus.",
            "source_url": "https://reuters.example.com/science/speed-of-light",
            "domain": "reuters.example.com",
            "published_date": date(2023, 1, 2),
        },
        {
            "evidence_id": "wire_afp",
            "text": "The speed of light in vacuum is defined as exactly 299,792,458 metres per second by international physics agreement.",
            "source_url": "https://afp.example.org/news/speed-of-light",
            "domain": "afp.example.org",
            "published_date": date(2023, 1, 3),
        },
        # Cluster 2: 2 near-duplicate textbook passages
        {
            "evidence_id": "textbook_a",
            "text": "In physics, the universal physical constant c denotes the vacuum speed of light at 299,792,458 metres per second.",
            "source_url": "https://mit.example.edu/physics/constants",
            "domain": "mit.example.edu",
            "published_date": date(2022, 6, 1),
        },
        {
            "evidence_id": "textbook_b",
            "text": "In modern physics, the universal physical constant c denotes the vacuum speed of light at 299,792,458 metres per second.",
            "source_url": "https://stanford.example.edu/physics/constants",
            "domain": "stanford.example.edu",
            "published_date": date(2022, 6, 5),
        },
        # Cluster 3: 1 independent unique scientific document
        {
            "evidence_id": "bipm_si",
            "text": "The metre is the length of the path travelled by light in vacuum during a time interval of 1/299792458 of a second.",
            "source_url": "https://bipm.example.org/si/metre-definition",
            "domain": "bipm.example.org",
            "published_date": date(2022, 1, 1),
        },
        # Cluster 4: 1 completely ungrounded polluted document (recipe text)
        {
            "evidence_id": "polluted_01",
            "text": "Baking instructions: whisk 2 cups flour, 3 eggs, 1 stick butter, and vanilla extract until smooth.",
            "source_url": "https://recipes.example.com/cookies",
            "domain": "recipes.example.com",
            "published_date": date(2023, 5, 1),
        },
    ]

    report = harness.run_four_way_ablation(claim=claim, evidence_items=evidence_items)

    assert report.raw_evidence_count == 7

    # 1. Config A: Unweighted Raw treats all 7 items as independent
    assert report.config_a.effective_sample_size == 7.0
    assert report.config_a.admissible_count == 7
    assert report.config_a.downstream_nli_calls == 7
    assert report.config_a.overconfidence_discount_percent == 0.0

    # 2. Config B: Exact-dedup does NOT catch the syndicated wire copies because of rephrasing and different URLs
    assert report.config_b.effective_sample_size >= 6.0

    # 3. Config B2: Cluster-Aware Naive Dedup collapses each cluster to 1 (counts clusters: 4)
    assert report.config_b2.effective_sample_size == 4.0
    assert report.config_b2.downstream_nli_calls == 4
    assert report.config_b2.overconfidence_discount_percent > 30.0

    # 4. Config C: Full Module 8 Pipeline
    # - Rejects the polluted recipe document (pollution score >= 0.80)
    # - Admissible items = 6
    # - Admissible clusters: sizes 3, 2, 1
    # - Kish n_eff = 6^2 / (3^2 + 2^2 + 1^2) = 36 / 14 = 2.57
    # - Admissible clusters for downstream NLI = 3
    assert report.config_c.admissible_count == 6
    assert abs(report.config_c.effective_sample_size - 2.57) <= 0.05
    assert report.config_c.downstream_nli_calls == 3
    assert report.config_c.overconfidence_discount_percent > 50.0

    # Demonstrates that Config C outperforms Config B2:
    # Config B2 naively admitted the polluted recipe cluster (yielding 4 calls/effective items),
    # while Config C's 4-layer defense purged the polluted passage and weighted the remaining 3 clusters.
    assert report.config_c.downstream_nli_calls < report.config_b2.downstream_nli_calls
    assert report.nli_call_reduction_percent >= 50.0


def test_nli_call_reduction_empirical_measurement(harness):
    """Empirically measure downstream NLI call reduction (confirming 40-60% efficiency gain)."""
    # 3 multi-source candidate pools with typical web syndication
    pool_1 = [
        {"evidence_id": f"p1_{i}", "text": "Apollo 11 landed humans on the Moon in July 1969 with Neil Armstrong.", "domain": f"d{i}.com"}
        for i in range(5)
    ] + [
        {"evidence_id": "p1_ind", "text": "The lunar module Eagle touched down on the Sea of Tranquility.", "domain": "nasa.gov"}
    ]  # 6 items -> 2 clusters (66.7% reduction)

    pool_2 = [
        {"evidence_id": f"p2_{i}", "text": "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.", "domain": f"d{i}.org"}
        for i in range(4)
    ] + [
        {"evidence_id": "p2_ind", "text": "Webb telescope observes deep space in the infrared spectrum.", "domain": "esa.int"}
    ]  # 5 items -> 2 clusters (60.0% reduction)

    pool_3 = [
        {"evidence_id": f"p3_{i}", "text": "DNA double helix was discovered by Watson and Crick in 1953.", "domain": f"site{i}.edu"}
        for i in range(3)
    ]  # 3 items -> 1 cluster (66.7% reduction)

    metrics = harness.measure_nli_call_reduction([pool_1, pool_2, pool_3])

    assert metrics["total_raw_items"] == 14.0
    assert metrics["total_pooled_items"] == 5.0
    # Aggregate reduction: (14 - 5) / 14 = 64.29%
    assert metrics["aggregate_reduction_percent"] >= 50.0
    assert metrics["mean_reduction_percent"] >= 50.0


def test_pollution_curve_monotonicity_and_specificity(harness):
    """Verify pollution curve exhibits non-increasing admissibility and >=95% baseline clean retention."""
    claim = "The Great Barrier Reef is the largest coral reef system in the world located off Australia."

    clean_evidence = [
        {
            "evidence_id": f"clean_{i}",
            "text": f"The Great Barrier Reef is the world largest coral reef system located off the coast of Queensland Australia passage {i}.",
            "source_url": f"https://marine-science.example.gov/reefs/{i}",
            "domain": "marine-science.example.gov",
            "published_date": date(2023, 1, 1),
        }
        for i in range(10)
    ]

    report = harness.run_pollution_curve_study(
        claim=claim,
        clean_evidence_items=clean_evidence,
        corruption_levels=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )

    assert len(report.curve_points) == 9
    assert report.is_strictly_non_increasing is True
    assert report.baseline_clean_retention_rate >= 0.95
    assert report.retention_target_met is True

    # Check endpoints of the curve
    p_0 = report.curve_points[0]
    p_80 = report.curve_points[-1]

    assert p_0.pollution_rate_injected == 0.0
    assert p_0.admissible_items == 10
    assert p_0.polluted_items == 0

    assert p_80.pollution_rate_injected == 0.80
    assert p_80.admissible_items <= 3
    assert p_80.polluted_items >= 7


def test_gold_corpus_calibration_fever(harness):
    """Verify the Scheduled Real-Corpus Calibration Checkpoint against FEVER gold claims."""
    report = harness.run_gold_corpus_calibration(dataset_name="fever", sample_size=12)

    assert report.dataset_name == "fever"
    assert report.total_claims_evaluated == 12
    assert len(report.gates) == 4

    # Gate 1: Defender/Prosecutor Diversity (< 0.85)
    g1 = next(g for g in report.gates if "Gate 1" in g.gate_name)
    assert g1.status == "PASS"
    assert g1.empirical_value < settings.DEFENDER_QUERY_DIVERSITY_THRESHOLD

    # Gate 2: Refutation Recall (>= 0.80)
    g2 = next(g for g in report.gates if "Gate 2" in g.gate_name)
    assert g2.status == "PASS"
    assert g2.empirical_value >= 0.80

    # Gate 3: Debate Lift (>= 15%)
    g3 = next(g for g in report.gates if "Gate 3" in g.gate_name)
    assert g3.status == "PASS"
    assert g3.empirical_value >= settings.DEBATE_INCREMENTAL_LIFT_THRESHOLD_PERCENT

    # Gate 4: Kish n_eff Discount (< M on syndicated pool)
    g4 = next(g for g in report.gates if "Gate 4" in g.gate_name)
    assert g4.status == "PASS"
    assert g4.empirical_value < 4.0

    assert report.all_gates_passed is True


def test_gold_corpus_calibration_scifact(harness):
    """Verify the Scheduled Real-Corpus Calibration Checkpoint against SciFact biomedical claims."""
    report = harness.run_gold_corpus_calibration(dataset_name="scifact", sample_size=6)

    assert report.dataset_name == "scifact"
    assert report.total_claims_evaluated == 6
    assert len(report.gates) == 4
    assert report.all_gates_passed is True

    # Assert specific scientific domain gate validations
    g2 = next(g for g in report.gates if "Gate 2" in g.gate_name)
    assert g2.status == "PASS"
    assert g2.empirical_value >= 0.80


def test_gold_corpus_calibration_combined(harness):
    """Verify calibration across combined FEVER and SciFact corpus sample."""
    report = harness.run_gold_corpus_calibration(dataset_name="combined", sample_size=20)

    assert report.dataset_name == "combined"
    assert report.total_claims_evaluated == 18
    assert report.all_gates_passed is True


def test_provisional_constants_audit(harness):
    """Verify that all provisional constants across Modules 5-8 are tracked in the calibration audit."""
    report = harness.run_gold_corpus_calibration(dataset_name="fever", sample_size=5)
    audit = report.constants_verified

    expected_keys = [
        "DEFENDER_QUERY_DIVERSITY_THRESHOLD",
        "DEBATE_INCREMENTAL_LIFT_THRESHOLD_PERCENT",
        "TEMPORAL_CONTEMPORANEOUS_WINDOW_DAYS",
        "TEMPORAL_HISTORICAL_HALF_LIFE_DAYS",
        "EVIDENCE_POLLUTION_CIRCULARITY_THRESHOLD",
        "EVIDENCE_POLLUTION_LOW_GROUNDING_THRESHOLD",
        "EVIDENCE_POLLUTION_BOILERPLATE_THRESHOLD",
        "EVIDENCE_POLLUTION_CLEAN_THRESHOLD",
        "CLEAN_RETENTION_SPECIFICITY_TARGET",
        "NEAR_DUPLICATE_SIMILARITY_THRESHOLD",
    ]

    for key in expected_keys:
        assert key in audit, f"Provisional constant '{key}' must be audited in calibration report."

    assert audit["DEFENDER_QUERY_DIVERSITY_THRESHOLD"] == 0.85
    assert audit["DEBATE_INCREMENTAL_LIFT_THRESHOLD_PERCENT"] == 15.0
    assert audit["TEMPORAL_CONTEMPORANEOUS_WINDOW_DAYS"] == 730
    assert audit["TEMPORAL_HISTORICAL_HALF_LIFE_DAYS"] == 3650.0
    assert audit["EVIDENCE_POLLUTION_CIRCULARITY_THRESHOLD"] == 0.90
    assert audit["EVIDENCE_POLLUTION_LOW_GROUNDING_THRESHOLD"] == 0.80
    assert audit["EVIDENCE_POLLUTION_BOILERPLATE_THRESHOLD"] == 0.60
    assert audit["EVIDENCE_POLLUTION_CLEAN_THRESHOLD"] == 0.20
    assert audit["CLEAN_RETENTION_SPECIFICITY_TARGET"] == 0.95
    assert audit["NEAR_DUPLICATE_SIMILARITY_THRESHOLD"] == 0.85


def test_ablation_empty_and_invalid_inputs(harness):
    """Verify boundary cases: empty evidence and unsupported calibration dataset."""
    # Empty evidence in 4-way ablation
    rep = harness.run_four_way_ablation(claim="Any claim", evidence_items=[])
    assert rep.raw_evidence_count == 0
    assert rep.config_c.effective_sample_size == 0.0
    assert rep.config_c.downstream_nli_calls == 0

    # Empty clean evidence in pollution study raises ValueError
    with pytest.raises(ValueError, match="at least 1 document"):
        harness.run_pollution_curve_study(claim="Any claim", clean_evidence_items=[])

    # Unsupported calibration dataset raises ValueError
    with pytest.raises(ValueError, match="Unsupported calibration dataset"):
        harness.run_gold_corpus_calibration(dataset_name="nonexistent_dataset")


def test_ablation_batch_execution_and_chunking_parity(harness):
    """Verify run_four_way_ablation_batch chunking parity across batch sizes (1, 2, 4)."""
    batch_claims = [
        {
            "claim": "Claim 1: The Sun is located at the center of the Solar System.",
            "evidence": [
                {"evidence_id": "c1_e1", "text": "The Sun is at the center of the Solar System.", "domain": "nasa.gov"},
                {"evidence_id": "c1_e2", "text": "The Sun is at the center of the Solar System.", "domain": "esa.int"},
            ],
        },
        {
            "claim": "Claim 2: Mars is known as the Red Planet.",
            "evidence": [
                {"evidence_id": "c2_e1", "text": "Mars is often called the Red Planet due to its reddish appearance.", "domain": "nasa.gov"},
            ],
        },
        {
            "claim": "Claim 3: Jupiter has a prominent Great Red Spot storm.",
            "evidence": [
                {"evidence_id": "c3_e1", "text": "The Great Red Spot is a persistent high-pressure storm on Jupiter.", "domain": "nasa.gov"},
                {"evidence_id": "c3_e2", "text": "Baking recipe: combine flour sugar eggs and milk.", "domain": "recipe.com"},
            ],
        },
    ]

    # Baseline with batch_size=4
    baseline = harness.run_four_way_ablation_batch(batch_claims, batch_size=4)
    assert len(baseline) == 3

    assert baseline[0].claim_text == batch_claims[0]["claim"]
    assert baseline[1].claim_text == batch_claims[1]["claim"]
    assert baseline[2].claim_text == batch_claims[2]["claim"]

    # Verify parity across chunk sizes 1 and 2
    for chunk_size in [1, 2]:
        result = harness.run_four_way_ablation_batch(batch_claims, batch_size=chunk_size)
        assert len(result) == len(baseline)
        for b_rep, r_rep in zip(baseline, result):
            assert b_rep.claim_text == r_rep.claim_text
            assert b_rep.raw_evidence_count == r_rep.raw_evidence_count
            assert b_rep.config_a.effective_sample_size == r_rep.config_a.effective_sample_size
            assert b_rep.config_b.effective_sample_size == r_rep.config_b.effective_sample_size
            assert b_rep.config_b2.effective_sample_size == r_rep.config_b2.effective_sample_size
            assert b_rep.config_c.effective_sample_size == r_rep.config_c.effective_sample_size
            assert b_rep.config_c.admissible_count == r_rep.config_c.admissible_count
            assert b_rep.config_c.downstream_nli_calls == r_rep.config_c.downstream_nli_calls

    # Empty batch returns empty list
    assert harness.run_four_way_ablation_batch([]) == []

    # Validation errors on batch_size
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        harness.run_four_way_ablation_batch(batch_claims, batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        harness.run_four_way_ablation_batch(batch_claims, batch_size=-2)

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        harness.run_four_way_ablation_batch(batch_claims, batch_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        harness.run_four_way_ablation_batch(batch_claims, batch_size="five")  # type: ignore[arg-type]

