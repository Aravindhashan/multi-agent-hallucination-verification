# Step 4.1 Status: Source Credibility & Domain Authority Scorer

## Metadata
- **Module**: Module 4 — Evidence Quality, Independence & Pollution Defense
- **Step**: Step 4.1
- **Title**: Source Credibility & Domain Authority Scorer
- **Date**: 2026-09-16
- **Status**: COMPLETE

---

## 1. Objective
Implement a lightweight, deterministic, offline source credibility and domain authority scoring engine over evidence metadata. The scorer operates as a pure function without network calls, LLMs, or ML models, evaluating publisher reliability, domain authority tiering, and provenance completeness.

---

## 2. Discovery Report (§1.2)

- **Python version**: `3.11.9`
- **Pydantic version (v1 or v2)**: `2.13.5` (v2)
- **Existing config pattern**: `@dataclass(frozen=True)` in `app/config.py` using `os.getenv()`, exported as singleton `settings = Settings()`, with specialized `@dataclass` configs in `app/services/` (e.g., `NLISettings` in `app/services/nli_config.py`).
- **Existing exception base class**: Domain-specific typed hierarchies inheriting from `Exception` (`LLMError`, `NLIError`, `CorpusValidationError`).
- **Existing logging pattern**: `logging.getLogger("hallucination_verifier.<subsystem>")`.
- **Existing service constructor pattern (DI? singleton?)**: Constructor parameter injection (`__init__(self, settings=None)`), plus module-level singleton accessor (`get_source_credibility_scorer()`, `reset_source_credibility_scorer()`).
- **Existing test fixture style**: Standard Pytest fixtures, `pytest.mark.parametrize`, direct mocking and property-based loops.
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
- **HybridRetriever public interface**:
  - `retrieve_bm25(query: str, top_k: int) -> list[RetrievalResult]`
  - `retrieve_dense(query: str, top_k: int) -> list[RetrievalResult]`
  - `retrieve_hybrid(query: str, top_k: int) -> list[HybridCandidate]`
  - `retrieve(query: str, top_k: int) -> list[EvidenceItem]`
  - `save(save_dir: Path | str) -> None`
  - `load(load_dir: Path | str, model_name: str) -> HybridRetriever`
- **Available provenance fields per document**:
  `document_id` (or `id`), `title`, `source`, `source_url`, `domain`, `published_date`.
- **Available source-type field**:
  `no` (neither `CorpusDocument` nor `RetrievalResult` defines a `source_type` field).
- **URL field name(s)**:
  `source_url` (and `domain` for pre-extracted host).
- **Baseline test count**:
  `119 passed, 1 warning in 134.39s (0:02:14)`
- **Any mismatch between this prompt's assumptions and the repo**:
  None. The prompt explicitly accounts for schemas lacking `source_type` (Section 10.2), specifying that `w_type = 0.0` when absent with weights dynamically re-normalized to 1.0.
- **Any frozen file that must be touched (with justification)**:
  None. `app/services/__init__.py` updated to export new Source Credibility symbols.

---

## 3. Architecture & Scoring Model

### Formula
$$C_{\text{source}} = \operatorname{clamp}(w_{\text{domain}} \cdot D + w_{\text{type}} \cdot T + w_{\text{provenance}} \cdot P, 0.0, 1.0)$$

### Default Weights & Rationale
- `w_domain = 0.50`: Domain authority is the strongest single signal available offline without live network inspection.
- `w_type = 0.20`: Source type categorization is useful but optional/often absent.
- `w_provenance = 0.30`: Provenance completeness is the second strongest signal because it enables subsequent citation integrity and independence analysis.
- When `source_type` is absent: $w_{\text{type}} = 0.0$, remaining weights re-normalized to:
  $$w_{\text{domain}} = \frac{0.50}{0.80} = 0.625, \quad w_{\text{provenance}} = \frac{0.30}{0.80} = 0.375$$

---

## 4. Domain Tier Policy (§7)
- **Tier 1 (High authority, Score = 1.00)**: `.gov`, `.mil`, `.edu`, `.int`, `.ac.uk`, `.gov.uk`
- **Tier 2 (Established, Score = 0.75)**: Curated institutional and scholarly allowlist (`wikipedia.org`, `nature.com`, `science.org`, `reuters.com`, `bbc.com`, `arxiv.org`, etc.)
- **Tier 3 (General, Score = 0.50)**: Curated commercial, news, blog platforms (`nytimes.com`, `theguardian.com`, `forbes.com`, `github.com`, etc.)
- **Tier 4 (Unknown / unverified, Score = 0.25)**: Unregistered domains, placeholders (`.example`, `.test`, `.invalid`, `.localhost`), bare IP addresses, and missing metadata.

---

## 5. Files Created & Modified

### Files Created
- `backend/app/services/source_credibility_config.py`: Weight definitions, allowlists, exception hierarchy, settings dataclass.
- `backend/app/services/source_credibility_models.py`: Pydantic models `CredibilityFactor` and `SourceCredibilityResult`.
- `backend/app/services/source_credibility_scorer.py`: Pure function `SourceCredibilityScorer` and singleton accessors.
- `backend/tests/test_source_credibility_domain.py`: 18 tests for domain normalization grammar and tier assignment.
- `backend/tests/test_source_credibility_provenance.py`: 10 tests for provenance completeness and weight re-normalization.
- `backend/tests/test_source_credibility.py`: 50 tests covering property-based invariants, determinism, JSON round-trip, snapshots, and performance.

### Files Modified
- `backend/app/services/__init__.py`: Exported Source Credibility classes and functions. *(Justification: Clean service-layer packaging).*
- `backend/tests/test_nli_engine.py`: Added explicit `gc.collect()` in `cleanup_singleton` fixture to release PyTorch tensors between test runs. *(Justification: Memory management on Windows host).*

---

## 6. Audit Checklist (§21)

| # | Check | Evidence | Pass? |
|---|---|---|---|
| A1 | No ML model loaded | Zero model imports in `app/services/source_credibility*.py` | PASS |
| A2 | No network calls | No socket, urllib.request, requests, or httpx in module | PASS |
| A3 | No LLM calls | Zero references to llm_service or prompt templates | PASS |
| A4 | Pure functions, stateless scorer | `test_scoring_determinism` passes | PASS |
| A5 | Deterministic across runs | Byte-identical explanations across fresh instances | PASS |
| C1 | Weights sum to 1.0 at load | `test_weights_must_sum_to_one` enforces sum = 1.0 | PASS |
| C2 | Score always in [0,1] | `test_property_score_bounds_and_finite` (100 random combos) | PASS |
| C3 | Tier classification matches §7 | `test_tier_classification_and_scores` | PASS |
| C4 | Provenance uses only real schema fields | Uses `document_id`, `title`, `source`, `source_url`, `domain`, `published_date` | PASS |
| C5 | Source-type mapping is config-driven | Configured via `SOURCE_CRED_SOURCE_TYPE_MAP` | PASS |
| C6 | Explanation contains disclaimer | Verified via `DISCLAIMER_PHRASE` in output | PASS |
| R1 | Malformed URL -> no crash | `test_edge_case_malformed_url` passes | PASS |
| R2 | Empty metadata -> no crash | `test_edge_case_empty_metadata` passes | PASS |
| R3 | IP / localhost / .example -> Tier 4 | `test_section_8_2_decision_table` passes | PASS |
| R4 | JSON round-trip works | `test_json_round_trip` passes | PASS |
| T1 | All Step 4.1 tests pass | 78 / 78 passed | PASS |
| T2 | Baseline tests pass | 119 baseline tests pass | PASS |
| T3 | No network/GPU/API needed for tests | All tests run 100% offline | PASS |
| T4 | Snapshot tests present and passing | `test_snapshots_stability` passes | PASS |
| S1 | No freshness/decay code | Grep verified: 0 matches | PASS |
| S2 | No duplicate detection | Grep verified: 0 matches | PASS |
| S3 | No independence/pollution/agent/verdict | Grep verified: 0 matches | PASS |
| S4 | No FastAPI routes / frontend | Service layer only | PASS |
| S5 | No corpus schema changes | `corpus_schema.py` is frozen and unmodified | PASS |
| P1 | score() < 1 ms warm | Measured: 0.1063 ms warm latency | PASS |
| P2 | No regex ReDoS on untrusted inputs | Uses `urllib.parse.urlsplit` stdlib parser | PASS |

---

## 7. Performance Numbers
- `score_ms_warm`: **0.1063 ms** (measured over 1,000 iterations on CPU)
- `batch_100_ms`: **10.82 ms** (measured over 50 batches of 100 items on CPU)
- `RSS memory impact`: **0.00 MB** (pure function; no weights or state retained)

---

## 8. Status
**COMPLETE** ✅
