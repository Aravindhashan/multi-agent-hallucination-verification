# Step 4.2 Status: Temporal Freshness & Decay Scorer

## Metadata
- **Module**: Module 4 — Evidence Quality, Independence & Pollution Defense
- **Step**: Step 4.2
- **Title**: Temporal Freshness & Decay Scorer (Enhanced)
- **Date**: 2026-09-16
- **Status**: COMPLETE

---

## 1. Objective
Implement a deterministic, offline temporal freshness and decay scoring engine over evidence publication dates relative to an explicit reference date anchor. The scorer operates as a pure mathematical function without network calls, LLMs, or ML models, evaluating temporal staleness via exponential half-life decay.

---

## 2. Discovery Report (§1.2)

- **Python version**: `3.11.9`
- **Pydantic version (v1 or v2)**: `2.13.5` (v2)
- **Existing config pattern**: `@dataclass` settings in `app/services/temporal_freshness_config.py` with runtime range and type validation, following the established `NLISettings` and `SourceCredibilitySettings` conventions.
- **Existing exception base class**: Typed domain exceptions inheriting from `Exception` (`TemporalFreshnessError`, `TemporalFreshnessConfigError`, `TemporalFreshnessInputError`).
- **Existing logging pattern**: `logging.getLogger("hallucination_verifier.temporal_freshness")`.
- **Existing service constructor pattern**: Constructor parameter injection (`__init__(self, settings=None)`), plus module-level singleton accessor (`get_temporal_freshness_scorer()`, `reset_temporal_freshness_scorer()`).
- **Existing test fixture style**: Standard Pytest fixtures, `pytest.mark.parametrize`, direct property-based tests, clock isolation tests.
- **CorpusDocument schema — exact field names and types**:
  - `id: str`
  - `title: str`
  - `text: str`
  - `source: str`
  - `source_url: str | None = None`
  - `domain: str | None = None`
  - `published_date: date | None = None`
- **Corpus loader module + function names**:
  - Module: `app.services.corpus_loader`
  - Class: `CorpusLoader`
  - Factory methods: `from_dict(cls, data: object)`, `from_json(cls, json_str: str)`, `from_file(cls, file_path: Path | str)`
- **RetrievalResult schema — exact fields**:
  - `document_id: str`
  - `title: str`
  - `text: str`
  - `source: str`
  - `source_url: str | None = None`
  - `domain: str | None = None`
  - `published_date: date | None = None`
  - `score: float`
  - `retrieval_method: str`
  - `rank: int`
- **Available temporal metadata fields per document**:
  Only `published_date: datetime.date | None`. No additional timestamps (`created_at`, `modified_at`, `crawled_at`) exist on `CorpusDocument` or `RetrievalResult`.
- **Baseline test count before Step 4.2**:
  `197 passed, 1 warning in 222.36s (0:03:42)`
- **Total test count after Step 4.2**:
  `237 passed, 1 warning in 128.21s (0:02:08)` (40 new tests for Step 4.2)
- **Any mismatch between prompt's assumptions and the repo**: None.
- **Any frozen file that must be touched (with justification)**:
  `app/services/__init__.py` updated to export Step 4.2 classes and functions. *(Justification: Clean service-layer packaging for downstream agents).*

---

## 3. Mathematical Formula & Scoring Model

### Formula Selection (§7.1)
$$F_{\text{temporal}} = 2^{-\frac{\text{age\_days}}{\text{half\_life\_days}}}$$

### Parameter Specifications
- `default_half_life_days = 365.0` (1.0 year): Factual knowledge / encyclopedic decay half-life.
- `missing_date_fallback = 0.50`: Neutral fallback when `published_date is None`, leaving evidence neither rewarded nor penalized.
- `reference_date`: Explicit `datetime.date` required on every score call. Complete system-clock independence.
- `DISCLAIMER_PHRASE`: `"Temporal freshness is a heuristic, not a truth signal."` embedded in all explanations.

### Decision Table (§8.2)

| Case | Scenario | Age Calculation | Freshness Score ($F$) | Flags |
|---|---|---|---|---|
| **Case A** | Same-day publication (`pub == ref`) | `age_days = 0` | $1.0000$ | `is_missing_date=False`, `is_future_date=False` |
| **Case B** | Historical publication (`pub < ref`) | `age_days = (ref - pub).days` | $2^{-\frac{\text{age\_days}}{\text{half\_life\_days}}}$ | `is_missing_date=False`, `is_future_date=False` |
| **Case C** | Future date anomaly (`pub > ref`) | Capped at `0` | $1.0000$ | `is_future_date=True`, flagged in explanation |
| **Case D** | Missing date (`pub is None`) | `age_days = None` | `missing_date_fallback` (0.50) | `is_missing_date=True` |
| **Case E** | Invalid `published_date` type (e.g., `str`, `int`) | N/A | Raises `TemporalFreshnessInputError` | N/A |
| **Case F** | Missing or non-date `reference_date` | N/A | Raises `TemporalFreshnessInputError` | N/A |

---

## 4. Files Created & Modified

### Files Created
- `backend/app/services/temporal_freshness_config.py`: Constants (`DEFAULT_HALF_LIFE_DAYS`, `DEFAULT_MISSING_DATE_FALLBACK`, `DISCLAIMER_PHRASE`), typed exception hierarchy, `TemporalFreshnessSettings` dataclass with boundary validation.
- `backend/app/services/temporal_freshness_models.py`: Pydantic model `TemporalFreshnessResult` with validation enforcing strict float finiteness and presence constraints.
- `backend/app/services/temporal_freshness_scorer.py`: Pure mathematical class `TemporalFreshnessScorer`, convenience helpers `score_document`, `score_batch`, and singleton accessors `get_temporal_freshness_scorer`, `reset_temporal_freshness_scorer`.
- `backend/tests/test_temporal_freshness.py`: 40 unit, mathematical property, calendar boundary, monkeypatch clock isolation, document helper, batch parity, and benchmark tests.

### Files Modified
- `backend/app/services/__init__.py`: Exported Step 4.2 classes and functions. *(Justification: Clean service-layer packaging).*

---

## 5. Audit Checklist (§26)

| # | Check | Evidence | Pass? |
|---|---|---|---|
| A1 | No ML model loaded | Zero PyTorch/transformers/ONNX imports in `temporal_freshness*.py` | PASS |
| A2 | No network calls | Zero socket/urllib/requests/httpx imports | PASS |
| A3 | No LLM calls | Zero LLM prompts or client calls | PASS |
| A4 | Pure functions, stateless scorer | `test_determinism_identical_runs` passes | PASS |
| A5 | Deterministic across runs | Multiple instances and calls return identical outputs | PASS |
| C1 | Config bounds enforced | `test_invalid_half_life_raises`, `test_invalid_missing_date_fallback_raises` pass | PASS |
| C2 | Score always in [0, 1] | `test_exact_formula_verification_table`, property test pass | PASS |
| C3 | Exact exponential half-life formula | Evaluated at 0, 10, 30, 90, 180, 365, 730, 1095, 3650 days matching $2^{-\text{age}/365}$ | PASS |
| C4 | Decision table Cases A–F covered | Covered by dedicated tests for each scenario | PASS |
| C5 | Leap years & calendar boundaries | Feb 29 leap day, multi-year leap boundaries, month & year transitions tested | PASS |
| C6 | Disclaimer phrase present | Explanation string contains `"Temporal freshness is a heuristic, not a truth signal."` | PASS |
| R1 | System clock independence | `test_system_clock_monkeypatch_does_not_affect_result` monkeypatches `date.today` and `datetime.now` | PASS |
| R2 | Zero system clock calls in source | Static grep confirmed 0 occurrences of `date.today`, `datetime.now`, `datetime.utcnow`, `time.time` | PASS |
| R3 | JSON round-trip | `test_json_round_trip` validates Pydantic serialization / deserialization | PASS |
| R4 | Document & batch helpers | `test_score_corpus_document_helper`, `test_score_retrieval_result_helper`, `test_score_batch_order_and_parity` pass | PASS |
| T1 | Step 4.2 tests pass | 40 / 40 passed | PASS |
| T2 | Baseline tests pass | 197 baseline + 40 Step 4.2 = 237 passed, 0 failed | PASS |
| S1 | No independence / pollution code | Grep verified: 0 matches | PASS |
| S2 | No agent / debate / verdict code | Grep verified: 0 matches | PASS |
| S3 | No combination with Step 4.1 ($C_{\text{source}}$) | $F_{\text{temporal}}$ kept strictly separate; no composite scorer created | PASS |
| P1 | score() < 1.0 ms warm | Measured: **0.0198 ms** (< 1.0 ms target) | PASS |
| P2 | batch_100 < 50.0 ms | Measured: **1.8316 ms** (< 50.0 ms target) | PASS |

---

## 6. Performance Benchmark Numbers
- `single_score_ms`: **0.0198 ms** (measured over 10,000 iterations on CPU)
- `batch_100_ms`: **1.8316 ms** (measured over 100 batches of 100 items on CPU)
- `RSS memory impact`: **0.00 MB** (pure mathematical calculation; zero persistent tensors or caches)

---

## 7. Status
**COMPLETE** ✅
