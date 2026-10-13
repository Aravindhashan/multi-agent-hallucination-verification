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

---

## 5. Standalone NLI Sanity Audit & Directionality Verification (2026-09-30)

- **Objective**: Standalone verification of the raw DeBERTa-v3 cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) outside the master pipeline to guarantee immunity against silent label-index swaps and directionality inversion.
- **Dynamic Label Resolution Audit**:
  - `model.config.id2label` resolved: `{0: 'entailment', 1: 'neutral', 2: 'contradiction'}`.
  - `model.config.label2id` resolved: `{'entailment': 0, 'neutral': 1, 'contradiction': 2}`.
  - Confirmed `NLIEngine._resolve_label_mapping()` dynamically maps indices using case-insensitive substring matching without assuming hardcoded array orders.
- **Directional Asymmetry Verification**:
  - `tokenizer(premise=evidence, hypothesis=claim)`: $P(\text{entailment}) = 0.9941$ (99.4% Entailment).
  - `tokenizer(premise=claim, hypothesis=evidence)`: $P(\text{entailment}) = 0.0028, P(\text{neutral}) = 0.9971$ (99.7% Neutral).
  - Proved that Premise must be Evidence and Hypothesis must be Claim; reversing them causes specific claims to collapse into Neutral.
- **10 Hand-Written Canonical Pairs Direct Benchmark**:
  - 4 Entailment pairs: 100% correct, mean confidence $> 99.5\%$.
  - 3 Contradiction pairs: 100% correct, mean confidence $> 99.8\%$.
  - 3 Neutral / Unrelated pairs: 100% correct, mean confidence $> 99.8\%$.
  - Raw Model Accuracy: **10 / 10 (100.0%)**.
- **NLIEngine Wrapper Parity**:
  - Confirmed `NLIEngine.predict()` and `NLIEngine.predict_batch()` produce bitwise-identical results to the raw model.
- **Dedicated Test Suite Added**:
  - `tests/test_nli_sanity_direct.py`: 3/3 tests passed in 129.86s.
  - Detailed Audit Report: [`docs/NLI_SANITY_AUDIT_REPORT.md`](file:///c:/Users/aravi/Downloads/HallucinationDetection_NLP/hallucination_verifier/docs/NLI_SANITY_AUDIT_REPORT.md).

---

## 6. Raw Inputs Logging & Token Tracking (2026-09-30)

- **Objective**: Full transparency into what the DeBERTa-v3 cross-encoder model was actually shown for every single claim verification to eliminate "black-box" uncertainty.
- **Fields Captured**:
  1. `premise_text`: Exact reference evidence text provided to the cross-encoder tokenizer.
  2. `token_length`: Truncated token count sent to the model attention mask.
  3. `raw_token_length`: Full un-truncated token length before applying max-length constraints.
  4. `truncated`: Boolean flag indicating if token-level or character-level truncation occurred.
  5. `p_entailment`, `p_neutral`, `p_contradiction`: The exact 3-class softmax probabilities.
- **Architectural Integration**:
  - `NLIResult` (`app/services/nli_models.py`): Extended to store `premise_text`, `token_length`, and `raw_token_length`.
  - `NLIEngine.predict_batch` (`app/services/nli_engine.py`): Logs full raw input metadata and 3-class probabilities at `INFO` level for every inference step.
  - `AdjudicationItemResult` & `EvidenceSnippetSummary` (`app/schemas.py`): Extended with token length and probability fields.
  - `adjudication_service.py`: Populates `raw_nli_inputs` array containing all evidence item evaluations and persists directly into SQLite `session_claims.metadata_json["adjudication"]`.
  - `verification_service.py`: Preserves raw NLI inputs in `ClaimVerificationSummary.raw_nli_inputs`.
  - `static/app.js`: Evidence cards in the UI dynamically render token lengths and exact 3-class DeBERTa probabilities `(E: xx%, N: xx%, C: xx%)`.
- **Test Coverage**:
  - `tests/test_raw_inputs_logging.py`: 3/3 passed (verifying engine capture, caplog INFO logging, and schema integrity).
  - `tests/test_nli_engine.py`: 18/18 non-slow passed (reconciled calibrated defaults).
  - `tests/test_nli_label_mapping.py`: 4/4 passed.
  - `tests/test_routers_integration.py`: 6/6 passed.

---

## 7. Sentence Window Extraction Engine & In-Context Evidence Distillation (2026-10-01)

- **Objective**: Replace blunt truncation (first 256 tokens / 1000 characters) across long evidence passages. Solves two critical production issues:
  1. **False-Neutral Eliminator**: In long evidence passages where the decisive supporting or refuting fact occurs in late sentences (e.g. sentence 6+), blunt truncation cuts off the evidence, causing DeBERTa to return `NEUTRAL` (false-negative / NOT ENOUGH INFO).
  2. **Inference Latency Reduction**: Because Transformer attention is $O(N^2)$ in token sequence length, feeding DeBERTa 40–80 condensed tokens instead of 144–256 truncated tokens cuts CPU attention operations by $\approx 60\%-75\%$.

- **Architectural Implementation**:
  1. **Sentence Segmentation & Scoring (`app/services/sentence_window.py`)**:
     - `split_into_sentences()`: Zero-dependency regex boundary detector preserving honorifics/abbreviations (`Dr.`, `U.S.`, `Jan.`, `approx.`, `est.`, etc.), decimals (`3.14`), currency (`$10.50`), and quotations without requiring heavy NLP packages (spaCy/NLTK).
     - `score_sentence()`: Multi-signal heuristic claim-relevance scoring combining:
       - Content word overlap (stopword-filtered, 40% weight).
       - Numerical / date matching (25% weight).
       - Capitalized named entity / proper noun matching (25% weight).
       - Bigram phrase match (10% weight).
     - `select_best_sentence_window()`:
       - Preserves short passages ($\le 3$ sentences) intact.
       - Implements 3 strategies: `contiguous` (best continuous block of 1–3 sentences), `top_k` (top-K ranked sentences in original document order), and `hybrid` (prefers contiguous window when its score is within 85% of top-K, preserving local discourse coherence).
  2. **Configuration & Data Models (`app/services/nli_config.py` & `nli_models.py`)**:
     - Added `enable_sentence_windows: bool = True`, `min_window_sentences: int = 1`, `max_window_sentences: int = 3`, `sentence_window_strategy: Literal["hybrid", "contiguous", "top_k"] = "hybrid"`.
     - `NLIResult` extended with `selected_sentence_indices: list[int] | None` and `sentence_window_score: float | None`.
  3. **Engine Pipeline Integration (`app/services/nli_engine.py`)**:
     - Prior to tokenization, `NLIEngine._prepare_premise()` evaluates the passage. If enabled and the passage exceeds `max_window_sentences`, the best window is selected as the premise.
     - Logs sentence selection at `INFO` level: `Claim '...' -> selected sents=[...] (score=..., strategy=hybrid)`.
     - Raw token length tracks full passage token length, while token length tracks the condensed premise.
  4. **Adjudication & Calibration Reconciliation (`app/services/cluster_pooling.py` & `adjudication_service.py`)**:
     - **Neutral Informativeness Discounting**: Previously, neutral background documents ($P(\text{neu}) \approx 0.99$, $P(\text{ent}) \approx 0.003$) produced log-odds of $\ln(0.003 / 0.997) = -5.8$, functioning as a severe negative penalty against valid claims. Added `effective_weights = w * (1.0 - p_neu)` so non-committal documents do not penalize true claims.
     - **Multi-Evidence Neutral Calibration**: Calibrated `weighted_neu = min(ev.p_neutral for ev in evaluated_evidence)` ensuring decisive evidence dominates uncertainty.

- **Empirical Benchmarks & Speedup**:
  - **Accuracy Breakthrough**:
    - Passage: Long synthetic plastics history passage (6 sentences, 144 tokens) where sentence 6 contains the crucial fact: *"Leo Baekeland synthesized Bakelite in 1907, creating the world's first fully synthetic plastic."*
    - Claim: *"Bakelite was synthesized by Leo Baekeland in 1907 as the first fully synthetic plastic."*
    - **Blunt Truncation Result**: `NEUTRAL` ($P(\text{neu}) = 0.9927, P(\text{ent}) = 0.0036$).
    - **Sentence Windowing Result**: `ENTAILMENT` ($P(\text{ent}) = 0.9893, P(\text{neu}) = 0.0101$).
  - **Latency Benchmark (CPU PyTorch)**:
    - Blunt truncation (144 tokens): **7,831 ms**.
    - Sentence windowing (70 tokens): **4,055 ms** (**1.93x faster**, 48.2% latency reduction).

- **Test Verification**:
  - `tests/test_sentence_window.py`: 13/13 passed (boundary splitting, abbreviations, multi-signal scoring, strategies, fallback).
  - `tests/test_raw_inputs_logging.py`: 3/3 passed.
  - `tests/test_nli_engine.py`: 18/18 non-slow passed.
  - `tests/test_cluster_pooling.py`: 15/15 passed.
  - `tests/test_adjudication_federated_batch.py`: 7/7 passed in 87.88s.



