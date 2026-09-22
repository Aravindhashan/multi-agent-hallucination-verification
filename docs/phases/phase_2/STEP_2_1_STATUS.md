# Step 2.1 Status: DeBERTa Cross-Encoder NLI Engine

## Metadata
- **Phase**: Phase 2 — NLI & Evidence Quality Engines
- **Step**: Step 2.1
- **Title**: DeBERTa Cross-Encoder NLI Engine
- **Date**: 2026-09-16
- **Status**: COMPLETE

---

## 1. Discovery Report — Step 2.1

- **Python version**: `3.11.9`
- **Torch version**: `2.14.0+cpu`
- **Transformers version**: `5.17.0`
- **Existing config pattern**: `@dataclass(frozen=True)` in `app/config.py` using `os.getenv()`, exported as singleton `settings = Settings()`.
- **Existing exception base class**: Domain-specific typed hierarchies (`LLMError(Exception)`, `CorpusValidationError(ValueError)`).
- **Existing logging pattern**: `logging.getLogger("hallucination_verifier")` / `logging.getLogger(__name__)`.
- **Existing Pydantic version**: `2.13.5` (Pydantic v2).
- **Existing DI / singleton patterns**: Factory functions (`get_llm_client()`, `get_claim_extractor()`), constructor parameter injection for testing, lazy-cached model loaders.
- **Existing test fixtures / conftest**: Standard pytest test suite in `tests/`.
- **Claim model class and fields**: `AtomicClaim` (`claim_id: str`, `claim_text: str`, `subject: str | None`, `predicate: str | None`, `object: str | None`, `temporal_anchor: str | None`, `numerical_value: str | None`, `spatial_anchor: str | None`, `qualifiers: list[str]`).
- **Evidence model class and fields**: `RetrievalResult` (`doc_id: str`, `score: float`, `title: str`, `text: str`, `source: str`, `source_url: str | None`, `domain: str | None`, `published_date: date | None`) and `EvidenceItem`.
- **Current test count**: `119 passed` (96 Phase 1 baseline + 23 Step 2.1 tests).
- **Filename/path mismatches**: None.
- **Phase 1 components requiring adaptation**: None. Phase 1 is frozen and 100% intact.

---

## 2. Baseline Gate Verification
- **Command**: `pytest -q`
- **Output**: `96 passed in 102.97s (0:01:42)`
- **Gate Status**: **PASSED** (Baseline 96/96 tests passing).

---

## 3. Implementation Details

### Components Created
1. `app/services/nli_models.py`:
   - Typed exception hierarchy: `NLIError`, `NLIConfigurationError`, `NLIModelLoadError`, `NLIInferenceError`, `NLIInputError`.
   - Typed enum: `NLILabel` (`ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`).
   - Pydantic model: `NLIResult` with strict mathematical probability contract validation (sum to 1.0, confidence matches max probability, label matches argmax).
2. `app/services/nli_config.py`:
   - Dataclass: `NLISettings` with parameter validation (`model_name`, `device`, `max_length`, `batch_size`, `max_claim_chars`, `max_evidence_chars`, `truncation_policy`, `cache_dir`).
3. `app/services/nli_engine.py`:
   - Core cross-encoder engine: `NLIEngine`.
   - Tokenization with standard NLI pair ordering `(premise=evidence, hypothesis=claim)`.
   - Strict truncation vs truncate policy handling.
   - Dynamic label mapping resolution from `model.config.id2label` (supporting index permutations and case insensitivity).
   - Calibrated softmax computation over raw logits.
   - Batch inference with chunking and strict input pre-validation.
   - Singleton factory: `get_nli_engine()` and `reset_nli_engine()`.
4. `app/services/__init__.py`:
   - Exported all Step 2.1 NLI symbols.
5. `tests/test_nli_engine.py` & `tests/test_nli_label_mapping.py`:
   - 23 unit and integration tests covering configuration validation, input validation, truncation behavior, probability contract enforcement, property-based tests across 100 random logit distributions, single/batch prediction, error propagation, singleton lifecycles, and live DeBERTa-v3 checkpoint probes.

---

## 4. Verification & Test Results
- **All non-slow tests**: `118 passed, 1 deselected in 93.29s`
- **All tests (including live probe)**: `119 passed in 114.20s`
- **Live DeBERTa-v3 probe test**: `1 passed in 61.31s`
  - Entailment test: `label=entailment, p_entailment=0.9983`
  - Contradiction test: `label=contradiction, p_contradiction=0.9998`
  - Neutral test: `label=neutral, p_neutral=0.9995`
- **Step 2.1 Gate Status**: **PASSED (119/119 tests passing, 0 regressions)**
