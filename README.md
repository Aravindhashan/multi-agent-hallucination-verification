# Hallucination Verifier

A modular, adversarial multi-agent FastAPI backend for explainable hallucination detection and verification in Large Language Models (LLMs).

The system decomposes LLM responses into independently verifiable atomic claims, retrieves supporting and contradicting evidence using a hybrid search engine (BM25 + FAISS Dense), validates evidence quality and independence, and conducts an adversarial debate between Defender and Prosecutor agents.

---

## Pipeline Architecture & Implemented Modules

### 1. Module 1: User Query & Input
- **Endpoint**: `POST /api/v1/query`
- Accepts JSON query input with length validation (1–1000 characters).
- Normalizes whitespace, rejects empty or whitespace-only inputs with HTTP 422.
- Generates a UUID4 `query_id` and timezone-aware UTC timestamp.

### 2. Module 2: Base LLM Response Generation
- **Endpoint**: `POST /api/v1/generate`
- Interfaces with LLM providers behind an abstract `BaseLLMClient` interface.
- Provides `RealLLMClient` (OpenAI-compatible) and deterministic `MockLLMClient` (`mock-llm-v0`).
- Stores response provenance, model metadata, and generation timestamps.

### 3. Module 3: Atomic Claim Decomposition & Quality Assessment
- **Endpoint**: `POST /api/v1/extract-claims`
- **Primary Engine**: **Deterministic Rule-Based NLP Pipeline** (`NLPClaimExtractor`):
  - **Sentence Segmentation**: Accurately handles sentences while protecting decimals (`3.14`, `299,792,458.0`) and abbreviations (`Dr.`, `U.S.`, `Prof.`, `Jan.`, `etc.`).
  - **Coordinate & Compound Clause Splitting**: Splits complex coordinate clauses across conjunctions (`and`, `but`, `while`, `;`) and propagates inherited subjects.
  - **Recursive Multi-Fact Appositive Splitting**: Decomposes multi-fact participial and appositive structures (such as date attribution, builder attribution, positional predicates, and location) into atomic claims.
  - **Precision Spatial Anchor Filtering**: Validates geographic entities with exclusion filters preventing false positives from organizations, temporal quarters/days, documents, and historical events.
  - **Anchor Field Synchronization**: Bidirectionally synchronizes plural anchor lists (`temporal_anchors`, `spatial_anchors`, `numerical_values`) with legacy scalar fields (`temporal_anchor`, etc.) ensuring `scalar == plural[0]`.
  - **Coreference Resolution**: Resolves third-person pronouns (`he`, `she`, `it`, `they`) against salient antecedent entities.
  - **Performance & Efficiency**: Operates in <2ms per sentence with zero token cost and low RAM footprint using a transparent deterministic rule-based NLP pipeline (compiled regex vocabularies in `verbs.py` / `lexical.py` and linguistic clause segmenters rather than heavy external dependency parsers like spaCy).
- **Secondary / Ablation Engine**: `LLMClaimExtractor` using prompt-based decomposition for benchmark comparisons.
- **Quality Evaluator**: `DecompositionQualityEvaluator` computing empirical coverage, atomicity, entailment, and high-quality claim rate.

### 4. Module 4: Claim Complexity Analysis & Verification Planning
- **Endpoint**: `POST /api/v1/analyze-claim`
- Evaluates claim complexity score ($0.0 \le C \le 1.0$) using lexical heuristics, entity density, numerical anchors, and absolute/superlative qualifiers.
- Automatically constructs an ordered multi-step verification plan and flags whether deep verification is required.

### 5. Evidence Quality, Freshness & Pollution Defense Suite
- **Endpoint**: `POST /api/v1/evidence/evaluate-quality`
- **Source Credibility & Domain Authority** (Step 4.1): Tier-based domain scoring (Tier 1 Gov/Edu 1.0, Tier 2 Academic/Scientific 0.85, Tier 3 Established News 0.70, Tier 4 General Web 0.50, Tier 5 Low Credibility 0.20).
- **Temporal Freshness & Age Decay** (Step 4.2): Half-life exponential decay scoring modeling claim publication vs. evidence age.
- **Source Independence & Redundancy Clustering** (Step 4.3): Complete-linkage Jaccard distance clustering detecting cross-source syndication and circular reporting.
- **Evidence Pollution & Circularity Defense** (Step 4.4): Directional lexical grounding rejecting ungrounded or self-referential passages.
- **Kish Effective Sample Size**: Calculates $n_{\text{eff}}$ over clustered evidence items.

### 6. Retrieval Engine & Stance Differentiation
- **`HybridRetriever`**: Merges BM25 lexical ranking (`rank_bm25`) with FAISS Dense semantic search (`all-MiniLM-L6-v2`) via reciprocal rank fusion and score preservation.
- **Memory-Safe Architecture**: Employs lazy loading for heavy neural and vector components to prevent out-of-memory errors on resource-constrained systems.
- **Stance Adaptation**:
  - `supporting`: Direct factual retrieval.
  - `contradicting`: Augments query with controversy, refutation, and criticism keywords (`controversy criticism contradiction disputed false refuted`).

### 7. Module 5 & Module 6: Defender & Prosecutor Agents
- **Defender Agent** (`POST /api/v1/defend-claim`): Retrieves supporting evidence using `HybridRetriever` (or mock fallback) with `supporting` stance.
- **Prosecutor Agent** (`POST /api/v1/prosecute-claim`): Retrieves counter-evidence using `HybridRetriever` with `contradicting` stance.
- Dependency injection wired cleanly via FastAPI `Depends(get_evidence_retriever)` allowing mock overrides in testing.

### 8. Module 7: Federated Multi-Turn Evidentiary Debate
- **Endpoint**: `POST /api/v1/debate`
- Dynamic multi-round adversarial debate between Defender and Prosecutor.
- **Federated Evidence Engine**: Queries local curated benchmarks (PubMed/SciFact BM25, FEVER) and escalates to live web retrieval (DuckDuckGo + Wikipedia full plain-text lead paragraphs) when local evidence falls below the escalation confidence threshold.
- Persistent SQLite web caching (`backend/data/web_cache.db`) eliminates redundant network requests with zero API subscription cost ($0.00).

### 9. Module 8: Evidence Quality Firewall & Cluster Representative Pooling
- Deduplicates and clusters evidence via complete-linkage Jaccard similarity.
- Computes Kish Effective Sample Size ($n_{\text{eff}}$) to guard against cross-source syndication and redundant copies.
- Identifies winning cluster representatives ($e_k^*$) to eliminate duplicate downstream inference calls.

### 10. Module 9: Cross-Encoder NLI Engine & Sentence-Window Adjudication
- **Endpoint**: `POST /api/v1/adjudicate-claim`
- DeBERTa-v3 cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) evaluates premise-hypothesis entailment, contradiction, and neutral classification.
- **Claim-Aware Sentence Windowing** (`select_best_sentence_window`): Dynamically extracts the most relevant 1–3 sentence window from evidence passages, avoiding raw snippet truncation and speeding up CPU inference.

### 11. Module 10 & 11: Master End-to-End Verification Pipeline & Decoupled Metrics
- **Endpoint**: `POST /api/v1/verify`
- Master pipeline executing decomposition, complexity planning, federated debate, quality firewalling, NLI cross-encoder adjudication, and Bayesian rollup.
- **Decoupled H-Index Formulation**:
  - **Factual Hallucination Index ($H_{\text{refuted}}$)**: Calibrated risk of verified factual falsehoods:
    $$H_{\text{refuted}} = \frac{1}{|C|} \sum_{c \in C_{\text{refuted}}} \max(0.75, \text{confidence}_c)$$
  - **Epistemic Coverage Gap ($U_{\text{unverified}}$)**: Tracks ungrounded or neutral claims without mislabeling them as hallucinations:
    $$U_{\text{unverified}} = \frac{1}{|C|} \sum_{c \in C_{\text{unverified}}} \max(0.50, P_{\text{neutral}, c})$$
  - Ensures corpus-agnostic calibration across PubMed, Wikipedia, and live search.

### 12. Module 12: Interactive Web UI Dashboard
- Built-in lightweight web interface served at `http://127.0.0.1:8000/`.
- Features real-time verification status, dual-metric badge cards (Hallucination Index & Coverage Gap), live debate stream, and full evidence provenance viewer.

### 13. Session State Management
- In-memory thread-safe `SessionStore` backed by SQLite tracking multi-turn queries, claims, retrieved evidence, and quality audit logs.

### 14. Centralized Configuration
- Centralized thresholds, word limits, domain tiers, and weights in `app/config.py`.

---

## Requirements

- Python 3.10 or newer
- Dependencies listed in `backend/requirements.txt`

---

## Run the Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## API Examples

### 1. Submit Query
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about the Eiffel Tower."}'
```

### 2. Generate Response
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about the Eiffel Tower."}'
```

### 3. Extract Atomic Claims (Deterministic NLP Pipeline)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/extract-claims" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about the Eiffel Tower.",
    "response_text": "The Eiffel Tower, built in 1889 by Gustave Eiffel, stands in Paris and is 330 meters tall."
  }'
```

### 4. Analyze Claim Complexity & Verification Plan
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyze-claim" \
  -H "Content-Type: application/json" \
  -d '{"claim_text": "The Eiffel Tower was built in 1889 by Gustave Eiffel."}'
```

### 5. Evaluate Evidence Quality & Independence
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/evidence/evaluate-quality" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_text": "The Eiffel Tower was built in 1889 by Gustave Eiffel.",
    "evidence_items": [
      {
        "id": "ev_001",
        "text": "The Eiffel Tower was constructed in 1889 under the direction of Gustave Eiffel.",
        "source_url": "https://www.culture.gouv.fr/eiffel-tower",
        "published_date": "2023-01-15T00:00:00Z"
      }
    ]
  }'
```

### 6. Defend Claim (Supporting Evidence)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/defend-claim" \
  -H "Content-Type: application/json" \
  -d '{"claim_text": "The Eiffel Tower stands in Paris."}'
```

### 7. Prosecute Claim (Contradicting Evidence)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/prosecute-claim" \
  -H "Content-Type: application/json" \
  -d '{"claim_text": "The Eiffel Tower was built in 1950."}'
```

---

## Directory Structure

```text
hallucination_verifier/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── dependencies.py
│   │   ├── routers/
│   │   │   ├── query.py
│   │   │   ├── generation.py
│   │   │   ├── claims.py
│   │   │   ├── analysis.py
│   │   │   ├── evidence_quality.py
│   │   │   ├── defender.py
│   │   │   └── prosecutor.py
│   │   ├── services/
│   │   │   ├── llm_client.py
│   │   │   ├── real_llm_client.py
│   │   │   ├── mock_llm_client.py
│   │   │   ├── llm_service.py
│   │   │   ├── claim_extractor.py
│   │   │   ├── nlp_claim_extractor.py
│   │   │   ├── llm_claim_extractor.py
│   │   │   ├── claim_models.py
│   │   │   ├── decomposition_evaluator.py
│   │   │   ├── self_containment_reviser.py
│   │   │   ├── claim_analyzer.py
│   │   │   ├── simple_claim_analyzer.py
│   │   │   ├── corpus_schema.py
│   │   │   ├── corpus_loader.py
│   │   │   ├── retrieval_models.py
│   │   │   ├── bm25_retriever.py
│   │   │   ├── dense_retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── nli_models.py
│   │   │   ├── nli_config.py
│   │   │   ├── nli_engine.py
│   │   │   ├── source_credibility_models.py
│   │   │   ├── source_credibility_config.py
│   │   │   ├── source_credibility_scorer.py
│   │   │   ├── temporal_freshness_models.py
│   │   │   ├── temporal_freshness_config.py
│   │   │   ├── temporal_freshness_scorer.py
│   │   │   ├── source_independence_models.py
│   │   │   ├── source_independence_config.py
│   │   │   ├── source_independence_scorer.py
│   │   │   ├── evidence_pollution_models.py
│   │   │   ├── evidence_pollution_config.py
│   │   │   ├── evidence_pollution_detector.py
│   │   │   ├── evidence_quality_service.py
│   │   │   ├── session_store.py
│   │   │   ├── defender_service.py
│   │   │   ├── prosecutor_service.py
│   │   │   └── startup_gate.py
│   │   └── utils/
│   │       ├── verbs.py
│   │       └── lexical.py
│   └── requirements.txt
├── docs/
│   ├── PROJECT_KNOWLEDGE_GRAPH.md
│   └── phases/
└── README.md
```
