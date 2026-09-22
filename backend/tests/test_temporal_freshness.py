"""Comprehensive tests for Temporal Freshness & Decay Scorer (Step 4.2).

Covers mathematical decay, decision table cases, leap years, determinism,
configuration validation, batch processing, JSON serialization, and static checks.
"""

from datetime import date, datetime, timedelta
import json
import math
import time
import pytest

from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult
from app.services.temporal_freshness_config import (
    DISCLAIMER_PHRASE,
    TemporalFreshnessConfigError,
    TemporalFreshnessInputError,
    TemporalFreshnessSettings,
)
from app.services.temporal_freshness_models import TemporalFreshnessResult
from app.services.temporal_freshness_scorer import (
    TemporalFreshnessScorer,
    get_temporal_freshness_scorer,
    reset_temporal_freshness_scorer,
)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Ensure singleton state is cleanly isolated."""
    reset_temporal_freshness_scorer()
    yield
    reset_temporal_freshness_scorer()


@pytest.fixture
def scorer() -> TemporalFreshnessScorer:
    """Fixture returning a default TemporalFreshnessScorer instance (half_life = 365.0)."""
    return TemporalFreshnessScorer()


# ---------------------------------------------------------------------------
# 1. Basic Scoring & Mathematical Decay (§7, §17)
# ---------------------------------------------------------------------------

def test_same_day_freshness(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 6, 1)
    res = scorer.score(published_date=ref_d, reference_date=ref_d)
    assert res.freshness_score == 1.0
    assert res.age_days == 0
    assert not res.is_future_date
    assert not res.is_missing_date


def test_one_day_old_freshness(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 6, 2)
    pub_d = date(2025, 6, 1)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    expected = 2.0 ** (-1.0 / 365.0)
    assert abs(res.freshness_score - expected) < 1e-4
    assert res.age_days == 1


def test_one_half_life_freshness(scorer: TemporalFreshnessScorer):
    pub_d = date(2024, 1, 1)
    ref_d = pub_d + timedelta(days=365)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert abs(res.freshness_score - 0.50) < 1e-4
    assert res.age_days == 365


def test_two_half_lives_freshness(scorer: TemporalFreshnessScorer):
    pub_d = date(2024, 1, 1)
    ref_d = pub_d + timedelta(days=730)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert abs(res.freshness_score - 0.25) < 1e-4
    assert res.age_days == 730


def test_three_half_lives_freshness(scorer: TemporalFreshnessScorer):
    pub_d = date(2024, 1, 1)
    ref_d = pub_d + timedelta(days=1095)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert abs(res.freshness_score - 0.125) < 1e-4
    assert res.age_days == 1095


# ---------------------------------------------------------------------------
# 2. Mathematical Correctness & Monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "age_days,expected_score",
    [
        (0, 1.0),
        (10, 2.0 ** (-10 / 365.0)),
        (30, 2.0 ** (-30 / 365.0)),
        (90, 2.0 ** (-90 / 365.0)),
        (180, 2.0 ** (-180 / 365.0)),
        (365, 0.5),
        (730, 0.25),
        (1095, 0.125),
        (3650, 2.0 ** (-3650 / 365.0)),
    ],
)
def test_exact_formula_verification_table(
    scorer: TemporalFreshnessScorer, age_days: int, expected_score: float
):
    ref_d = date(2025, 1, 1)
    pub_d = ref_d - timedelta(days=age_days)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert abs(res.freshness_score - expected_score) < 1e-4
    assert 0.0 <= res.freshness_score <= 1.0


def test_monotonic_decrease_with_increasing_age(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 1, 1)
    previous_score = 1.0
    for days_ago in range(0, 2000, 25):
        pub_d = ref_d - timedelta(days=days_ago)
        res = scorer.score(published_date=pub_d, reference_date=ref_d)
        assert res.freshness_score <= previous_score
        assert 0.0 <= res.freshness_score <= 1.0
        previous_score = res.freshness_score


# ---------------------------------------------------------------------------
# 3. Decision Table Cases (§9)
# ---------------------------------------------------------------------------

def test_case_c_future_date_handling(scorer: TemporalFreshnessScorer):
    """Case C: Future publication date assigns F=1.0, flags is_future_date=True, notes anomaly."""
    ref_d = date(2025, 1, 1)
    future_pub_d = date(2026, 1, 1)
    res = scorer.score(published_date=future_pub_d, reference_date=ref_d)

    assert res.freshness_score == 1.0
    assert res.age_days == 0
    assert res.is_future_date is True
    assert res.is_missing_date is False
    assert "later than the supplied reference date" in res.explanation
    assert DISCLAIMER_PHRASE in res.explanation


def test_case_d_missing_date_fallback(scorer: TemporalFreshnessScorer):
    """Case D: Missing publication date uses configured neutral fallback (0.5), flags is_missing_date=True."""
    ref_d = date(2025, 1, 1)
    res = scorer.score(published_date=None, reference_date=ref_d)

    assert res.freshness_score == 0.50
    assert res.age_days is None
    assert res.published_date is None
    assert res.is_missing_date is True
    assert res.is_future_date is False
    assert "Publication date is unavailable" in res.explanation
    assert DISCLAIMER_PHRASE in res.explanation


def test_case_e_invalid_published_date_type(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 1, 1)
    with pytest.raises(TemporalFreshnessInputError, match="published_date"):
        scorer.score(published_date="2024-01-01", reference_date=ref_d)  # type: ignore

    with pytest.raises(TemporalFreshnessInputError, match="published_date"):
        scorer.score(published_date=12345, reference_date=ref_d)  # type: ignore


def test_case_f_invalid_reference_date_type(scorer: TemporalFreshnessScorer):
    pub_d = date(2024, 1, 1)
    with pytest.raises(TemporalFreshnessInputError, match="reference_date"):
        scorer.score(published_date=pub_d, reference_date=None)  # type: ignore

    with pytest.raises(TemporalFreshnessInputError, match="reference_date"):
        scorer.score(published_date=pub_d, reference_date="2025-01-01")  # type: ignore


# ---------------------------------------------------------------------------
# 4. Calendar Boundaries & Leap Years
# ---------------------------------------------------------------------------

def test_leap_year_february_29(scorer: TemporalFreshnessScorer):
    """2024 is a leap year; 2024-02-28 to 2024-03-01 is exactly 2 calendar days."""
    pub_d = date(2024, 2, 28)
    ref_d = date(2024, 3, 1)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert res.age_days == 2


def test_leap_year_to_next_year(scorer: TemporalFreshnessScorer):
    """2024-02-29 to 2025-02-28 is exactly 365 calendar days."""
    pub_d = date(2024, 2, 29)
    ref_d = date(2025, 2, 28)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert res.age_days == 365
    assert abs(res.freshness_score - 0.50) < 1e-4


def test_year_boundary_transition(scorer: TemporalFreshnessScorer):
    """2024-12-31 to 2025-01-01 is exactly 1 day."""
    pub_d = date(2024, 12, 31)
    ref_d = date(2025, 1, 1)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert res.age_days == 1


def test_month_boundary_transition(scorer: TemporalFreshnessScorer):
    """2025-01-31 to 2025-02-01 is exactly 1 day."""
    pub_d = date(2025, 1, 31)
    ref_d = date(2025, 2, 1)
    res = scorer.score(published_date=pub_d, reference_date=ref_d)
    assert res.age_days == 1


# ---------------------------------------------------------------------------
# 5. Determinism & System Clock Independence (§6)
# ---------------------------------------------------------------------------

def test_determinism_identical_runs(scorer: TemporalFreshnessScorer):
    pub_d = date(2023, 5, 15)
    ref_d = date(2024, 5, 15)
    res1 = scorer.score(pub_d, ref_d)
    res2 = scorer.score(pub_d, ref_d)

    assert res1.freshness_score == res2.freshness_score
    assert res1.explanation == res2.explanation
    assert res1.model_dump() == res2.model_dump()


def test_explicit_reference_date_controls_result(scorer: TemporalFreshnessScorer):
    pub_d = date(2023, 1, 1)
    res_2023 = scorer.score(pub_d, date(2023, 1, 1))
    res_2024 = scorer.score(pub_d, date(2024, 1, 1))
    res_2025 = scorer.score(pub_d, date(2025, 1, 1))

    assert res_2023.freshness_score > res_2024.freshness_score > res_2025.freshness_score


def test_system_clock_monkeypatch_does_not_affect_result(
    scorer: TemporalFreshnessScorer, monkeypatch: pytest.MonkeyPatch
):
    """Ensure mock changes to system date have zero impact on score."""
    pub_d = date(2024, 1, 1)
    ref_d = date(2025, 1, 1)
    baseline_res = scorer.score(pub_d, ref_d)

    # Monkeypatch datetime.now to an arbitrary distant year
    class FakeDateTime:
        @classmethod
        def now(cls):
            return datetime(2099, 1, 1)

    monkeypatch.setattr("datetime.datetime", FakeDateTime)
    monkeypatch_res = scorer.score(pub_d, ref_d)

    assert baseline_res.freshness_score == monkeypatch_res.freshness_score
    assert baseline_res.explanation == monkeypatch_res.explanation


# ---------------------------------------------------------------------------
# 6. Configuration Validation (§14)
# ---------------------------------------------------------------------------

def test_custom_half_life_scoring():
    custom_scorer = TemporalFreshnessScorer(
        TemporalFreshnessSettings(half_life_days=180.0)
    )
    pub_d = date(2024, 1, 1)
    ref_d = pub_d + timedelta(days=180)
    res = custom_scorer.score(pub_d, ref_d)
    assert abs(res.freshness_score - 0.50) < 1e-4
    assert res.half_life_days == 180.0


def test_custom_missing_date_fallback():
    custom_scorer = TemporalFreshnessScorer(
        TemporalFreshnessSettings(missing_date_fallback=0.25)
    )
    res = custom_scorer.score(None, date(2025, 1, 1))
    assert res.freshness_score == 0.25
    assert "fallback score of 0.25" in res.explanation


def test_invalid_half_life_raises():
    with pytest.raises(TemporalFreshnessConfigError, match="half_life_days"):
        TemporalFreshnessSettings(half_life_days=0.0)

    with pytest.raises(TemporalFreshnessConfigError, match="half_life_days"):
        TemporalFreshnessSettings(half_life_days=-10.0)

    with pytest.raises(TemporalFreshnessConfigError, match="half_life_days"):
        TemporalFreshnessSettings(half_life_days=float("nan"))

    with pytest.raises(TemporalFreshnessConfigError, match="half_life_days"):
        TemporalFreshnessSettings(half_life_days=float("inf"))


def test_invalid_missing_date_fallback_raises():
    with pytest.raises(TemporalFreshnessConfigError, match="missing_date_fallback"):
        TemporalFreshnessSettings(missing_date_fallback=-0.1)

    with pytest.raises(TemporalFreshnessConfigError, match="missing_date_fallback"):
        TemporalFreshnessSettings(missing_date_fallback=1.1)

    with pytest.raises(TemporalFreshnessConfigError, match="missing_date_fallback"):
        TemporalFreshnessSettings(missing_date_fallback=float("nan"))


# ---------------------------------------------------------------------------
# 7. Document Helper & Corpus Schema Compatibility
# ---------------------------------------------------------------------------

def test_score_corpus_document_helper(scorer: TemporalFreshnessScorer):
    doc = CorpusDocument(
        id="doc_temp_01",
        title="Apollo 11",
        text="Apollo 11 landed on the Moon in 1969.",
        source="NASA History",
        source_url="https://history.nasa.gov/apollo11",
        domain="history.nasa.gov",
        published_date=date(2019, 7, 20),
    )
    ref_d = date(2020, 7, 19)
    res = scorer.score_document(doc, reference_date=ref_d)
    assert res.age_days == 365
    assert abs(res.freshness_score - 0.50) < 1e-4


def test_score_retrieval_result_helper(scorer: TemporalFreshnessScorer):
    ret_res = RetrievalResult(
        document_id="ret_temp_01",
        title="Web Standards",
        text="W3C recommendations.",
        source="W3C",
        source_url="https://w3.org/TR/rec",
        domain="w3.org",
        published_date=None,
        score=0.92,
        retrieval_method="HYBRID",
        rank=1,
    )
    ref_d = date(2025, 1, 1)
    res = scorer.score_document(ret_res, reference_date=ref_d)
    assert res.is_missing_date is True
    assert res.freshness_score == 0.50


def test_score_dict_helper(scorer: TemporalFreshnessScorer):
    doc = {"published_date": date(2024, 1, 1), "title": "Dict Doc"}
    res = scorer.score_document(doc, reference_date=date(2024, 1, 1))
    assert res.freshness_score == 1.0


def test_score_document_unsupported_type_raises(scorer: TemporalFreshnessScorer):
    with pytest.raises(TemporalFreshnessInputError):
        scorer.score_document("invalid_string_doc", reference_date=date(2025, 1, 1))


# ---------------------------------------------------------------------------
# 8. Batch Scoring (§15)
# ---------------------------------------------------------------------------

def test_score_batch_order_and_parity(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 1, 1)
    items = [
        (date(2025, 1, 1), ref_d),
        (date(2024, 1, 2), ref_d),  # 365 days
        (None, ref_d),
        (date(2026, 1, 1), ref_d),  # future
    ]
    batch_results = scorer.score_batch(items)

    assert len(batch_results) == 4
    # Check parity with individual scoring
    for i, (pub_d, r_d) in enumerate(items):
        indiv = scorer.score(pub_d, r_d)
        assert batch_results[i].freshness_score == indiv.freshness_score
        assert batch_results[i].explanation == indiv.explanation
        assert batch_results[i].is_missing_date == indiv.is_missing_date
        assert batch_results[i].is_future_date == indiv.is_future_date


def test_score_batch_empty(scorer: TemporalFreshnessScorer):
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# 9. Serialization & JSON Round-Trip (§11)
# ---------------------------------------------------------------------------

def test_json_round_trip(scorer: TemporalFreshnessScorer):
    pub_d = date(2023, 6, 15)
    ref_d = date(2024, 6, 14)
    res = scorer.score(pub_d, ref_d)

    json_str = json.dumps(res.model_dump(mode="json"))
    data = json.loads(json_str)

    reconstructed = TemporalFreshnessResult(**data)
    assert reconstructed.freshness_score == res.freshness_score
    assert reconstructed.published_date == res.published_date
    assert reconstructed.reference_date == res.reference_date
    assert reconstructed.age_days == res.age_days
    assert reconstructed.half_life_days == res.half_life_days
    assert reconstructed.explanation == res.explanation
    assert reconstructed.is_missing_date == res.is_missing_date
    assert reconstructed.is_future_date == res.is_future_date


# ---------------------------------------------------------------------------
# 10. Performance Benchmarks (§16)
# ---------------------------------------------------------------------------

def test_performance_single_score_under_one_ms(scorer: TemporalFreshnessScorer):
    pub_d = date(2024, 1, 1)
    ref_d = date(2025, 1, 1)

    # Warmup
    for _ in range(50):
        _ = scorer.score(pub_d, ref_d)

    iterations = 1000
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scorer.score(pub_d, ref_d)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) * 1000.0) / iterations
    assert avg_ms < 1.0, f"Average single score latency ({avg_ms:.4f} ms) exceeds 1.0 ms."


def test_performance_batch_100_under_fifty_ms(scorer: TemporalFreshnessScorer):
    ref_d = date(2025, 1, 1)
    batch = [(date(2024, 1, 1) + timedelta(days=i % 300), ref_d) for i in range(100)]

    # Warmup
    for _ in range(10):
        _ = scorer.score_batch(batch)

    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scorer.score_batch(batch)
    t1 = time.perf_counter()

    avg_batch_ms = ((t1 - t0) * 1000.0) / iterations
    assert avg_batch_ms < 50.0, f"Batch 100 latency ({avg_batch_ms:.2f} ms) exceeds 50.0 ms."


# ---------------------------------------------------------------------------
# 11. Singleton Accessor Tests
# ---------------------------------------------------------------------------

def test_singleton_accessor():
    s1 = get_temporal_freshness_scorer()
    s2 = get_temporal_freshness_scorer()
    assert s1 is s2

    reset_temporal_freshness_scorer()
    s3 = get_temporal_freshness_scorer()
    assert s1 is not s3
