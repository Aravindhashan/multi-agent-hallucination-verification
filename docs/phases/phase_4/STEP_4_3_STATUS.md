# Step 4.3 Status: Source Independence & Redundancy Detector

## Metadata
- **Module**: Module 4 — Evidence Quality, Independence & Pollution Defense
- **Step**: Step 4.3
- **Title**: Source Independence & Redundancy Detector (Enhanced)
- **Date**: 2026-09-17
- **Status**: COMPLETE

---

## 1. Objective
Implement a deterministic, offline lexical source independence and redundancy detection engine. The system evaluates whether retrieved evidence items represent genuinely independent sources or whether multiple evidence items are substantially redundant, duplicated, or syndicated, preventing artificial confidence inflation in downstream claim verification.

---

## 2. Discovery Report (§5)

```markdown
### Discovery Report — Step 4.3

- Python version: 3.11.9
- Pydantic version: 2.13.5 (v2)
- Existing config pattern: @dataclass(frozen=True) with strict parameter bounds and type validation
- Existing exception base class: Typed domain exceptions inheriting from Exception (SourceIndependenceError, SourceIndependenceConfigError, SourceIndependenceInputError)
- Existing logging pattern: logging.getLogger("hallucination_verifier.source_independence")
- Existing service constructor pattern: Constructor parameter injection (__init__(self, settings=None)), plus module-level singleton accessors (get_source_independence_detector(), reset_source_independence_detector())
- Existing test fixture style: Pytest standard fixtures, autouse cleanup, parameterized tests, property-based tests, clock isolation tests
- CorpusDocument fields: id, title, text, source, source_url, domain, published_date
- RetrievalResult fields: document_id, title, text, source, source_url, domain, published_date, score, retrieval_method, rank
- Existing text normalization utilities: bm25_retriever.tokenize (simple lowercase word regex)
- Existing similarity utilities: FAISS FlatIP (dense retrieval); no general offline lexical similarity utility existed in services
- Existing tokenization utilities: bm25_retriever.tokenize; Step 4.3 implements deterministic NFKC Unicode word tokenization
- Step 4.1 public interface: SourceCredibilityScorer.score(domain, source, source_url, published_date, document_id) -> SourceCredibilityResult
- Step 4.2 public interface: TemporalFreshnessScorer.score(published_date, reference_date) -> TemporalFreshnessResult
- Existing dependency suitable for similarity: Python standard library (unicodedata, re, urllib.parse) provides 100% deterministic Jaccard and n-gram similarity without external ML packages
- Current baseline `pytest -q` summary line (pasted):
  237 passed, 1 warning in 103.48s (0:01:43)
- Frozen files that would need modification:
  backend/app/services/__init__.py
- Why each frozen modification is necessary:
  Clean service-layer exports of Step 4.3 public classes, models, and helpers for downstream agent consumption.
- Assumptions in this prompt that differ from the repository:
  None. The available metadata matches the repository schemas exactly.
```

---

## 3. Important Research Boundary (§7)
> [!IMPORTANT]
> **Research Boundary Notice**:
> Step 4.3 is **NOT** a truth detector.
> - `duplicate ≠ false`
> - `syndicated ≠ false`
> - `independent ≠ true`
>
> It strictly evaluates whether evidence items represent independent observational sources or redundant / syndicated reporting.
> Mandatory heuristic disclaimer:
> *"Independence and redundancy are heuristic signals, not proof of source independence."*

---

## 4. Architecture & Detection Layers (§9, §22)

### 4.1 Normalization & Tokenization Pipeline
1. **Unicode Normalization**: NFKC decomposition (`unicodedata.normalize("NFKC", text)`).
2. **Control Character Stripping**: ASCII non-whitespace control characters removed.
3. **Whitespace Collapse**: Consecutive whitespace runs collapsed to a single space; trimmed.
4. **Tokenization**: Lowercase conversion, word boundary extraction via `re.compile(r"\b\w+\b", flags=re.UNICODE)`.
5. **Raw Text Preservation**: The original input text is never mutated.

### 4.2 Lexical Similarity Metric
Token Jaccard similarity:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- Identical non-empty token sets: $J = 1.0$.
- Disjoint sets: $J = 0.0$.
- Both empty sets: guarded explicitly to return $0.0$ (no division by zero).

### 4.3 Anti-Double-Counting Decision Hierarchy (§22)
Correlated signals are evaluated through a strict precedence hierarchy rather than additive scoring:

```
[Evidence Pair (A, B)]
         │
         ├── Self-Comparison (id_a == id_b)
         │         └── EXACT_DUPLICATE (independence = 0.0)
         │
         ├── Layer 1: Exact Duplicate
         │         ├── Matching normalized canonical URL (with non-empty text)
         │         └── OR: Identical normalized text (with >= minimum_meaningful_tokens)
         │                 └── EXACT_DUPLICATE (independence = 0.0)
         │
         ├── Layer 2: Near-Duplicate (Jaccard >= near_duplicate_threshold [0.85])
         │         ├── If min(tokens_a, tokens_b) < minimum_meaningful_tokens [5]:
         │         │         └── WEAKLY_RELATED (short text policy, independence = 1 - J)
         │         └── Else:
         │                   └── REDUNDANT (independence = 1 - J)
         │
         ├── Layer 3: Weakly Related
         │         ├── Jaccard >= weak_relation_threshold [0.50]:
         │         │         └── WEAKLY_RELATED (independence = 1 - J)
         │         └── OR: Shared domain/source context with low lexical similarity:
         │                   └── WEAKLY_RELATED (contextual signal, independence = (1 - J) * 0.90)
         │
         └── Layer 4: Independent
                   └── INDEPENDENT (independence = 1 - J)
```

### 4.4 Group-Level Complete-Linkage Clustering (§23, §24)
- **Non-Transitivity of Jaccard**: $J(A, B) \ge 0.85$ and $J(B, C) \ge 0.85$ does not imply $J(A, C) \ge 0.85$.
- **Linkage Rule**: Complete-linkage (clique-based) discovery ensures that every document pair within a discovered cluster is mutually redundant, preventing the chaining pathology of single-linkage clustering.

---

## 5. Files Created & Modified

### Files Created
- `backend/app/services/source_independence_config.py`: Thresholds (`near_duplicate_threshold=0.85`, `weak_relation_threshold=0.50`, `minimum_meaningful_tokens=5`), disclaimer phrase, typed exceptions (`SourceIndependenceConfigError`, `SourceIndependenceInputError`), settings dataclass.
- `backend/app/services/source_independence_models.py`: Pydantic models `RedundancyClass`, `IndependencePairResult`, `IndependenceCluster`, `IndependenceSetResult` with float finiteness validators.
- `backend/app/services/source_independence_scorer.py`: Pure lexical engine `SourceIndependenceDetector`, canonical batch comparison (`compare_batch`), group complete-linkage clustering (`analyze_set`), URL/domain/source normalizers, and singleton accessors.
- `backend/tests/test_source_independence.py`: 40 tests covering exact duplicates, near duplicates, Jaccard math, metadata signals, anti-double-counting, edge cases, clock isolation, batch parity, Jaccard non-transitivity, sensitivity sweep, snapshots, and micro-benchmarks.

### Files Modified
- `backend/app/services/__init__.py`: Clean packaging and exports of Step 4.3 symbols. *(Justification: Downstream service-layer access).*
- `docs/phases/phase_4/PHASE_4_STATUS.md`: Recorded Step 4.3 completion. *(Justification: Project tracking).*
- `docs/PROJECT_KNOWLEDGE_GRAPH.md`: Updated roadmap table. *(Justification: Project knowledge graph alignment).*

---

## 6. Audit Checklist (§48)

| # | Check | Verification Command | Actual Output | Result |
|---|---|---|---|---|
| **0.1** | Sacred Baseline Gate | `pytest -q` | `237 passed, 1 warning in 103.48s (0:01:43)` | **PASS** |
| **A1** | Zero ML models loaded | Static Python regex audit | `ML/Network/LLM: 0 matches (CLEAN)` | **PASS** |
| **A2** | Zero network calls | Static Python regex audit | `0 matches (CLEAN)` | **PASS** |
| **A3** | Zero LLM calls | Static Python regex audit | `0 matches (CLEAN)` | **PASS** |
| **A4** | Pure function, stateless scorer | `pytest tests/test_source_independence.py -k test_determinism_identical_runs` | `1 passed` | **PASS** |
| **A5** | Deterministic outputs across runs | `pytest tests/test_source_independence.py -k test_determinism_identical_runs` | Byte-identical explanations and results | **PASS** |
| **C1** | Exact duplicates & normalization | `pytest tests/test_source_independence.py -k "test_exact_duplicate or test_raw_text_not_mutated"` | `5 passed` | **PASS** |
| **C2** | Mathematical Jaccard correctness | `pytest tests/test_source_independence.py -k "test_jaccard"` | `6 passed` | **PASS** |
| **C3** | Near-duplicate threshold boundaries | `pytest tests/test_source_independence.py -k "test_near_duplicate or test_just_below"` | `3 passed` | **PASS** |
| **C4** | Anti-double-counting hierarchy | `pytest tests/test_source_independence.py -k "test_same_domain_low_similarity or test_same_canonical_url"` | `2 passed` | **PASS** |
| **C5** | Edge cases (empty, short, self-comp) | `pytest tests/test_source_independence.py -k "self_comparison or short_text or empty or malformed"` | `4 passed` | **PASS** |
| **C6** | Disclaimer phrase present | `pytest tests/test_source_independence.py -k test_exact_duplicate_identical_raw_text` | `DISCLAIMER_PHRASE in res.explanation` | **PASS** |
| **R1** | System clock independence | `pytest tests/test_source_independence.py -k test_system_clock_monkeypatch` | `1 passed` | **PASS** |
| **R2** | Zero system clock calls in source | Static Python regex audit | `Clock/Random: 0 matches (CLEAN)` | **PASS** |
| **R3** | JSON round-trip | `pytest tests/test_source_independence.py -k test_json_round_trip` | `1 passed` | **PASS** |
| **R4** | Batch canonical ordering & equivalence | `pytest tests/test_source_independence.py -k "test_compare_batch or test_batch_equivalence"` | `3 passed` | **PASS** |
| **R5** | Group complete-linkage clustering | `pytest tests/test_source_independence.py -k "test_jaccard_non_transitivity or test_analyze_set"` | `2 passed` | **PASS** |
| **T1** | Step 4.3 test suite passing | `pytest tests/test_source_independence.py -v` | `40 passed in 7.05s` | **PASS** |
| **T2** | Full regression test suite passing | `pytest -q` | `277 passed, 1 warning in 167.49s (0:02:47)` | **PASS** |
| **S1** | Zero Step 4.4 code present | Static Python regex audit | `Scope Creep: 0 matches (CLEAN)` | **PASS** |
| **S2** | Zero agent / debate / verdict code | Static Python regex audit | `0 matches (CLEAN)` | **PASS** |
| **S3** | No combination with Step 4.1 or 4.2 | Code review of models and services | Kept 100% separate | **PASS** |
| **P1** | Warm single pair comparison | Micro-benchmark (5,000 iterations) | **0.2541 ms** (< 5.0 ms target) | **PASS** |
| **P2** | Warm 100-item set analysis | Micro-benchmark (100 items = 4,950 pairs) | **284.3393 ms** (< 500 ms target) | **PASS** |

---

## 7. Performance Benchmarks
- `single_pair_ms`: **0.2541 ms** (measured over 5,000 warm iterations on CPU)
- `batch_100_ms`: **284.3393 ms** (measured over 5 complete set analyses of 100 items / 4,950 pairwise comparisons on CPU)
- `RSS memory impact`: **0.00 MB** (pure lexical calculation; zero model weights or cache retention)

---

## 8. Status
**COMPLETE** ✅
