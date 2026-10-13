"""Dedicated batch testing and anchor-aware temporal freshness test suite (Submodule 8.2).

Validates:
1. Slope-corrected contemporaneous documentation formula (F >= 0.95 across full 2-year / 730-day window).
2. Boundary continuity across the 5-year historical threshold (|F(4.9 yr) - F(5.1 yr)| < 0.05).
3. 10-year retrospective decay half-life (h = 3650 days) for non-contemporaneous scholarship.
4. Continuous upper envelope dominance (F_final = max(F_contemporary, F_historical)).
5. Flexible temporal anchor format parsing (ISO date, textual date, year, embedded phrase).
6. Batch processing order and count preservation with anchors.
7. Memory-safe chunked batching across varied batch_size (1, 5, 10, 25, 50).
8. Heterogeneous batch inputs (2-tuples, 3-tuples, dicts with scalar and plural anchors).
9. Malformed and missing anchor resilience.
10. Mutation defenses (invalid batch sizes, corrupted config, and regression traps).
"""

from dataclasses import dataclass
from datetime import date, timedelta
import math
import pytest

from app.services.temporal_freshness_config import (
    DEFAULT_CONTEMPORANEOUS_SLOPE,
    DEFAULT_CONTEMPORANEOUS_WINDOW_DAYS,
    DEFAULT_HISTORICAL_HALF_LIFE_DAYS,
    DEFAULT_HISTORICAL_LOOKBACK_YEARS,
    TemporalFreshnessConfigError,
    TemporalFreshnessInputError,
    TemporalFreshnessSettings,
)
from app.services.temporal_freshness_scorer import (
    TemporalFreshnessScorer,
    get_temporal_freshness_scorer,
    reset_temporal_freshness_scorer,
)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Ensure singleton instance is cleared before and after each test."""
    reset_temporal_freshness_scorer()
    yield
    reset_temporal_freshness_scorer()


@pytest.fixture
def scorer() -> TemporalFreshnessScorer:
    """Provide a default TemporalFreshnessScorer instance."""
    return TemporalFreshnessScorer()


# ---------------------------------------------------------------------------
# 1. Slope-Corrected Contemporaneous Documentation Window (F >= 0.95 at 2 yr)
# ---------------------------------------------------------------------------

def test_contemporaneous_documentation_slope_at_boundaries(scorer: TemporalFreshnessScorer):
    """Verify F = 1.00 at delta_t = 0, F >= 0.95 at delta_t = 730 days (2 yr), and F >= 0.95 throughout."""
    ref_d = date(2026, 6, 1)
    event_d = date(1969, 7, 20)
    anchor_str = "1969-07-20"

    # delta_t = 0 days (day of the event)
    res_0 = scorer.score(published_date=event_d, reference_date=ref_d, temporal_anchor=anchor_str)
    assert res_0.freshness_score == 1.00
    assert res_0.is_historical_anchor is True
    assert "contemporaneous documentation" in res_0.explanation

    # delta_t = 365 days (1 year after event)
    res_365 = scorer.score(published_date=event_d + timedelta(days=365), reference_date=ref_d, temporal_anchor=anchor_str)
    expected_365 = 1.0 - (365 / 730) * 0.05  # 0.9750
    assert res_365.freshness_score == pytest.approx(expected_365, abs=1e-4)
    assert res_365.freshness_score >= 0.95

    # delta_t = 730 days (exact 2-year boundary)
    res_730 = scorer.score(published_date=event_d + timedelta(days=730), reference_date=ref_d, temporal_anchor=anchor_str)
    expected_730 = 1.0 - (730 / 730) * 0.05  # 0.9500
    assert res_730.freshness_score == pytest.approx(0.9500, abs=1e-4)
    assert res_730.freshness_score >= 0.95, "Specification requirement: F >= 0.95 across entire 2-year window."


# ---------------------------------------------------------------------------
# 2. Boundary Continuity Across 5-Year Historical Threshold (< 0.05 delta)
# ---------------------------------------------------------------------------

def test_continuity_across_5_year_boundary(scorer: TemporalFreshnessScorer):
    """Verify that events at 4.9 years vs 5.1 years do not exhibit cliff discontinuities (|delta| < 0.05)."""
    ref_d = date(2026, 6, 1)

    # Event 1: 4.9 years ago (~1790 days)
    event_4_9 = ref_d - timedelta(days=int(4.9 * 365.25))
    # Event 2: 5.1 years ago (~1863 days)
    event_5_1 = ref_d - timedelta(days=int(5.1 * 365.25))

    # Scenario A: Contemporaneous documents (published at the time of each event)
    res_contemp_4_9 = scorer.score(published_date=event_4_9, reference_date=ref_d, temporal_anchor=event_4_9.isoformat())
    res_contemp_5_1 = scorer.score(published_date=event_5_1, reference_date=ref_d, temporal_anchor=event_5_1.isoformat())

    delta_contemp = abs(res_contemp_4_9.freshness_score - res_contemp_5_1.freshness_score)
    assert delta_contemp < 0.05, f"Contemporaneous boundary delta {delta_contemp:.4f} must be < 0.05."
    assert res_contemp_4_9.freshness_score == 1.00
    assert res_contemp_5_1.freshness_score == 1.00

    # Scenario B: Retrospective scholarship published 2 years ago (date(2024, 6, 1))
    retro_pub = ref_d - timedelta(days=730)
    res_retro_4_9 = scorer.score(published_date=retro_pub, reference_date=ref_d, temporal_anchor=event_4_9.isoformat())
    res_retro_5_1 = scorer.score(published_date=retro_pub, reference_date=ref_d, temporal_anchor=event_5_1.isoformat())

    delta_retro = abs(res_retro_4_9.freshness_score - res_retro_5_1.freshness_score)
    assert delta_retro < 0.05, f"Retrospective boundary delta {delta_retro:.4f} must be < 0.05."

    # Scenario C: Document published 3 years after each respective event
    doc_after_4_9 = event_4_9 + timedelta(days=int(3 * 365.25))
    doc_after_5_1 = event_5_1 + timedelta(days=int(3 * 365.25))
    res_after_4_9 = scorer.score(published_date=doc_after_4_9, reference_date=ref_d, temporal_anchor=event_4_9.isoformat())
    res_after_5_1 = scorer.score(published_date=doc_after_5_1, reference_date=ref_d, temporal_anchor=event_5_1.isoformat())

    delta_after = abs(res_after_4_9.freshness_score - res_after_5_1.freshness_score)
    assert delta_after < 0.05, f"Offset boundary delta {delta_after:.4f} must be < 0.05."


# ---------------------------------------------------------------------------
# 3. Retrospective Scholarship 10-Year Half-Life (h = 3650 days)
# ---------------------------------------------------------------------------

def test_retrospective_scholarship_10_year_half_life(scorer: TemporalFreshnessScorer):
    """Verify non-contemporaneous retrospective scholarship decays under 10-year half-life."""
    ref_d = date(2026, 6, 1)
    anchor_1969 = "1969-07-20"  # Event is 57 years old

    # Retrospective paper published 10 years ago (delta_t_pub = 3650 days, delta_t_event > 730 days)
    pub_10_yr_ago = ref_d - timedelta(days=3650)
    res_retro_10 = scorer.score(published_date=pub_10_yr_ago, reference_date=ref_d, temporal_anchor=anchor_1969)

    # Contemporary decay would be 2**(-10) = 0.000976
    # Retrospective decay is 2**(-3650 / 3650) = 0.5000
    assert res_retro_10.freshness_score == pytest.approx(0.5000, abs=1e-3)
    assert res_retro_10.is_historical_anchor is True
    assert "retrospective scholarship" in res_retro_10.explanation


# ---------------------------------------------------------------------------
# 4. Continuous Upper Envelope Dominance
# ---------------------------------------------------------------------------

def test_upper_envelope_dominance(scorer: TemporalFreshnessScorer):
    """Verify final score is strictly the upper envelope max(F_contemporary, F_historical)."""
    ref_d = date(2026, 1, 1)

    # Case A: Brand-new document published yesterday about Apollo 11 (1969)
    # Contemporary decay is ~1.00, retrospective is ~1.00 -> max is 1.00
    res_new = scorer.score(published_date=ref_d - timedelta(days=1), reference_date=ref_d, temporal_anchor="1969")
    assert res_new.freshness_score >= 0.99

    # Case B: Historic document published in 1969
    # Contemporary decay is ~0.00, contemporaneous event proximity is 1.00 -> max is 1.00
    res_old = scorer.score(published_date=date(1969, 7, 20), reference_date=ref_d, temporal_anchor="1969")
    assert res_old.freshness_score == 1.00

    # Case C: Retrospective book published in 2018 (8 years ago)
    # Contemporary decay: 2**(-8) = 0.0039
    # Retrospective decay: 2**(-8/10) = 0.5743 -> max is 0.5743
    res_book = scorer.score(published_date=date(2018, 1, 1), reference_date=ref_d, temporal_anchor="1969")
    assert res_book.freshness_score == pytest.approx(0.5743, abs=0.01)


# ---------------------------------------------------------------------------
# 5. Temporal Anchor Format Parsing Flexibility
# ---------------------------------------------------------------------------

def test_temporal_anchor_formats(scorer: TemporalFreshnessScorer):
    """Verify ISO dates, textual dates, 4-digit years, and embedded phrases are cleanly parsed."""
    ref_d = date(2026, 6, 1)
    pub_d = date(1969, 7, 20)

    formats = [
        "1969-07-20",
        "July 20, 1969",
        "Jul 20, 1969",
        "1969",
        "Apollo 11 landed in 1969 on the moon",
        "1969/07/20",
    ]

    for fmt in formats:
        res = scorer.score(published_date=pub_d, reference_date=ref_d, temporal_anchor=fmt)
        assert res.freshness_score == 1.00, f"Failed for format: {fmt}"
        assert res.is_historical_anchor is True


# ---------------------------------------------------------------------------
# 6. Batch Scoring Preserves Order and Counts with Anchors
# ---------------------------------------------------------------------------

def test_score_batch_preserves_order_and_counts_with_anchors(scorer: TemporalFreshnessScorer):
    """Verify batch scoring produces exact 1-to-1 sequential results with anchors."""
    ref_d = date(2026, 1, 1)
    items = [
        (date(2025, 1, 1), ref_d, "2025"),
        (date(1969, 7, 20), ref_d, "1969"),
        (date(2020, 5, 1), ref_d, "1945"),
        (date(2024, 6, 1), ref_d, None),
    ]

    results = scorer.score_batch(items)
    assert len(results) == 4
    # Item 0: 2025 contemporaneous with 2025
    assert results[0].freshness_score == 1.00
    # Item 1: 1969 contemporaneous with 1969
    assert results[1].freshness_score == 1.00
    # Item 2: 2020 retrospective on 1945
    assert results[2].is_historical_anchor is True
    # Item 3: No anchor -> pure contemporary decay
    assert results[3].temporal_anchor is None


# ---------------------------------------------------------------------------
# 7. Chunking Large Payloads with Exact Invariant Parity
# ---------------------------------------------------------------------------

def test_score_batch_chunking_large_payload_anchors(scorer: TemporalFreshnessScorer):
    """Verify chunked execution (batch_size=1, 5, 10, 25, 50) produces exact parity on N=25."""
    ref_d = date(2026, 1, 1)
    items = [
        (
            date(1970 + i, 1, 1),
            ref_d,
            f"{1970 + i}" if i % 2 == 0 else None,
        )
        for i in range(25)
    ]

    baseline = scorer.score_batch(items, batch_size=25)
    chunk_1 = scorer.score_batch(items, batch_size=1)
    chunk_5 = scorer.score_batch(items, batch_size=5)
    chunk_10 = scorer.score_batch(items, batch_size=10)
    chunk_50 = scorer.score_batch(items, batch_size=50)

    assert len(baseline) == 25
    for i in range(25):
        assert baseline[i].freshness_score == chunk_1[i].freshness_score == chunk_10[i].freshness_score
        assert baseline[i].is_historical_anchor == chunk_5[i].is_historical_anchor == chunk_50[i].is_historical_anchor
        assert baseline[i].explanation == chunk_10[i].explanation


# ---------------------------------------------------------------------------
# 8. Heterogeneous Input Batch Support (Tuples, Dicts, Objects)
# ---------------------------------------------------------------------------

@dataclass
class CustomDocItem:
    published_date: date
    temporal_anchor: str


def test_score_batch_heterogeneous_inputs_and_doc_extraction(scorer: TemporalFreshnessScorer):
    """Verify batch processing heterogeneous formats including dicts and dataclasses with metadata anchors."""
    ref_d = date(2026, 1, 1)

    items = [
        # Legacy 2-tuple (uses default None anchor)
        (date(2025, 1, 1), ref_d),
        # 3-tuple with anchor
        (date(1969, 7, 20), ref_d, "1969-07-20"),
        # Mapping with temporal_anchor
        {"published_date": date(1989, 11, 9), "temporal_anchor": "1989"},
        # Mapping with plural temporal_anchors list
        {"published_date": date(1776, 7, 4), "temporal_anchors": ["1776", "1783"]},
        # Custom dataclass
        CustomDocItem(published_date=date(2020, 1, 1), temporal_anchor="1918"),
    ]

    results = scorer.score_batch(items, reference_date=ref_d, batch_size=2)
    assert len(results) == 5
    assert results[0].temporal_anchor is None
    assert results[1].temporal_anchor == "1969-07-20" and results[1].freshness_score == 1.00
    assert results[2].temporal_anchor == "1989" and results[2].freshness_score == 1.00
    assert results[3].temporal_anchor == "1776" and results[3].freshness_score == 1.00
    assert results[4].is_historical_anchor is True


# ---------------------------------------------------------------------------
# 9. Malformed and Missing Anchor Resilience
# ---------------------------------------------------------------------------

def test_score_batch_missing_and_malformed_anchors(scorer: TemporalFreshnessScorer):
    """Verify anchors with no dates or unparseable text fall back gracefully without error."""
    ref_d = date(2026, 1, 1)
    pub_d = date(2024, 1, 1)

    malformed_anchors = [
        "",
        "   ",
        "unrelated text with no year",
        "some random anchor",
        None,
    ]

    for bad_anc in malformed_anchors:
        res = scorer.score(published_date=pub_d, reference_date=ref_d, temporal_anchor=bad_anc)
        # Should gracefully evaluate standard contemporary decay
        assert res.is_historical_anchor is False
        assert res.temporal_anchor is None
        assert res.freshness_score == pytest.approx(2.0 ** (-731 / 365), abs=1e-3)


# ---------------------------------------------------------------------------
# 10. Mutation Testing & Defensive Invariants
# ---------------------------------------------------------------------------

def test_temporal_freshness_mutation_pass(scorer: TemporalFreshnessScorer):
    """Mutation testing: verify invalid inputs, bad configs, and trap mutated slope/half-life."""
    ref_d = date(2026, 1, 1)
    sample_batch = [(date(2025, 1, 1), ref_d)]

    # Mutation 1: Invalid batch size types
    for bad_size in (0, -1, -50, True, False, "10", 3.14):
        with pytest.raises(TemporalFreshnessInputError):
            scorer.score_batch(sample_batch, batch_size=bad_size)  # type: ignore

    # Mutation 2: Invalid configuration parameters
    with pytest.raises(TemporalFreshnessConfigError):
        TemporalFreshnessSettings(half_life_days=-10.0)
    with pytest.raises(TemporalFreshnessConfigError):
        TemporalFreshnessSettings(historical_half_life_days=0.0)
    with pytest.raises(TemporalFreshnessConfigError):
        TemporalFreshnessSettings(contemporaneous_slope=1.5)
    with pytest.raises(TemporalFreshnessConfigError):
        TemporalFreshnessSettings(contemporaneous_window_days=0)

    # Mutation 3: Slope mutation trap (e.g. 0.10 instead of 0.05 fails >= 0.95 at 730 days)
    mutated_settings = TemporalFreshnessSettings(contemporaneous_slope=0.10)
    mutated_scorer = TemporalFreshnessScorer(settings=mutated_settings)
    res_mut = mutated_scorer.score(
        published_date=date(1971, 7, 20),
        reference_date=ref_d,
        temporal_anchor="1969-07-20",  # delta_t = 730 days
    )
    # Under mutated slope 0.10, score is 0.90, failing the >= 0.95 requirement
    assert res_mut.freshness_score < 0.95, "Mutated slope must produce score < 0.95, proving test sensitivity."
