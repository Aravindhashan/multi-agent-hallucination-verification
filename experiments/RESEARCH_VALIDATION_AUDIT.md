# Research Validation Audit — Pre-Module-5
**Project**: Dynamic Adversarial Multi-Agent Framework for Explainable Hallucination Verification in Large Language Models  
**Date**: 2026-09-17  
**Status**: COMPLETE  

---

## 1. Objective
Audit the repository architecture and available metadata signals to establish the exact empirical foundation for the Pre-Module-5 experimental validation, strictly identifying available signals, unavailable signals, and public service interfaces without fabricating capabilities.

---

## 2. Available Signals (Verified Against Reality)

| Signal Name | Exact Field Name | Type | Model / Location | Description |
|---|---|---|---|---|
| **Claim Text & ID** | `claim_id`<br>`claim_text` | `str`<br>`str` | `AtomicClaim` in `app/services/claim_models.py` | Unique claim identifier and natural language claim string |
| **Evidence Text & ID** | `id` / `document_id`<br>`text` | `str`<br>`str` | `CorpusDocument` (`corpus_schema.py`)<br>`RetrievalResult` (`retrieval_models.py`) | Evidence identifier and passage text (50–300 words) |
| **Evidence Title** | `title` | `str` | `CorpusDocument`, `RetrievalResult` | Headline or document title |
| **Source URL** | `source_url` | `str \| None` | `CorpusDocument`, `RetrievalResult` | Canonical or raw web URL of origin |
| **Domain Name** | `domain` | `str \| None` | `CorpusDocument`, `RetrievalResult` | Normalized hostname/publisher domain |
| **Source Name** | `source` | `str` | `CorpusDocument`, `RetrievalResult` | Human-readable publisher string |
| **Publication Date** | `published_date` | `date \| None` | `CorpusDocument`, `RetrievalResult` | Calendar publication date |
| **NLI Label & Probabilities** | `label`<br>`probabilities`<br>`entailment_prob`<br>`contradiction_prob`<br>`neutral_prob` | `NLILabel`<br>`dict[NLILabel, float]`<br>`float`<br>`float`<br>`float` | `NLIResult` in `app/services/nli_models.py` | DeBERTa cross-encoder classification and softmax probabilities |
| **Source Credibility Score** | `credibility_score`<br>`factors` | `float`<br>`list[CredibilityFactor]` | `SourceCredibilityResult` in `app/services/source_credibility_models.py` | Composite credibility score $C_{\text{source}} \in [0, 1]$ |
| **Temporal Freshness Score** | `freshness_score`<br>`age_days` | `float`<br>`int \| None` | `TemporalFreshnessResult` in `app/services/temporal_freshness_models.py` | Exponential decay freshness score $F_{\text{temporal}} \in [0, 1]$ |
| **Independence & Redundancy** | `independence_score`<br>`text_similarity`<br>`redundancy_class`<br>`cluster_id` | `float`<br>`float`<br>`RedundancyClass`<br>`int` | `IndependencePairResult`, `IndependenceCluster`, `IndependenceSetResult` (`source_independence_models.py`) | Pairwise independence score and complete-linkage clusters |
| **Pollution Score & Class** | `pollution_score`<br>`classification`<br>`grounding_score`<br>`circularity_supported` | `float`<br>`PollutionClass`<br>`float`<br>`bool` | `EvidencePollutionResult` in `app/services/evidence_pollution_models.py` | Directional grounding, self-referential / duplication pollution scores |
| **Retrieval Metadata** | `score`<br>`retrieval_method`<br>`rank` | `float`<br>`str`<br>`int` | `RetrievalResult` in `app/services/retrieval_models.py` | Hybrid retrieval score, method ('bm25', 'dense', 'hybrid'), rank |

---

## 3. Unavailable Signals (Anti-Fabrication Notice §5)

The following signals are **NOT** present in the repository and **MUST NOT** be fabricated:
1. **Citation Graph / Reference Edges**: Neither `CorpusDocument` nor `RetrievalResult` contains outbound citation links, reference lists, or citation target IDs.
2. **Inbound / Outbound Hyperlink Graphs**: No web graph or page link topology is stored.
3. **Parent / Child Document IDs**: No hierarchical article versioning or derivation trees exist.
4. **Syndication IDs / Canonical Document IDs**: No wire-service grouping keys (`publisher_id`, `syndication_id`, `canonical_document_id`).
5. **Author / Bylines**: Author identity metadata is not defined in any schema.

*Impact on Experiment*:  
Experimental ground truth will explicitly model controlled provenance groups (`provenance_group_id`, `parent_source_id`), but production modules are tested strictly on observable features. Module 4.4 correctly sets `circularity_supported = False` and `is_circular = False`.

---

## 4. Public Interfaces for Experiment Integration

The experimental runners will interact with Modules 1–4 strictly through these frozen public APIs:

### 4.1 Module 2.1 — DeBERTa Cross-Encoder NLI Engine
```python
# app/services/nli_engine.py
def get_nli_engine(settings: NLISettings | None = None) -> NLIEngine: ...
class NLIEngine:
    def predict(self, premise: str, hypothesis: str) -> NLIResult: ...
    def predict_batch(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[NLIResult]: ...
```

### 4.2 Module 4.1 — Source Credibility Scorer
```python
# app/services/source_credibility_scorer.py
def get_source_credibility_scorer(settings: SourceCredibilitySettings | None = None) -> SourceCredibilityScorer: ...
class SourceCredibilityScorer:
    def score(self, domain: str | None, source: str | None, source_url: str | None, published_date: date | None, document_id: str = "") -> SourceCredibilityResult: ...
    def score_document(self, document: Any) -> SourceCredibilityResult: ...
```

### 4.3 Module 4.2 — Temporal Freshness Scorer
```python
# app/services/temporal_freshness_scorer.py
def get_temporal_freshness_scorer(settings: TemporalFreshnessSettings | None = None) -> TemporalFreshnessScorer: ...
class TemporalFreshnessScorer:
    def score(self, published_date: date | None, reference_date: date) -> TemporalFreshnessResult: ...
    def score_document(self, document: Any, reference_date: date) -> TemporalFreshnessResult: ...
```

### 4.4 Module 4.3 — Source Independence Detector
```python
# app/services/source_independence_scorer.py
def get_source_independence_detector(settings: SourceIndependenceSettings | None = None) -> SourceIndependenceDetector: ...
class SourceIndependenceDetector:
    def compare(self, evidence_a: Any, evidence_b: Any) -> IndependencePairResult: ...
    def compare_batch(self, items: Sequence[Any]) -> list[IndependencePairResult]: ...
    def analyze_set(self, items: Sequence[Any]) -> IndependenceSetResult: ...
```

### 4.5 Module 4.4 — Evidence Pollution Detector
```python
# app/services/evidence_pollution_detector.py
def get_evidence_pollution_detector(settings: EvidencePollutionSettings | None = None) -> EvidencePollutionDetector: ...
class EvidencePollutionDetector:
    def evaluate(self, claim: str, evidence: Any, claim_origin_url: str | None = None, seen_urls: set[str] | None = None, seen_texts: set[str] | None = None, seen_doc_ids: set[str] | None = None) -> EvidencePollutionResult: ...
    def evaluate_batch(self, claim: str, evidence_items: Sequence[Any], claim_origin_url: str | None = None) -> BatchPollutionResult: ...
```

---

## 5. Integration Gaps & Constraints

1. **Frozen Production Integrity**:
   - `backend/app/` remains strictly read-only.
   - Experiments reside entirely in `experiments/` outside the frozen source tree.
   - All comparisons consume identical pre-retrieved/annotated evidence.
2. **Deterministic Aggregation**:
   - Systems A0, A1, A2, and A3 share the exact same mathematical confidence formula; only the surviving evidence set varies.
