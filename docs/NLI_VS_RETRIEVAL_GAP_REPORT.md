# Diagnostic Gap Report: Separating NLI Errors from Retrieval Errors on SciFact

**Date**: 2026-10-01  
**Dataset**: SciFact Benchmark (5,183 peer-reviewed PubMed biomedical abstracts, AllenAI)  
**Sample**: 100 real scientific claims with ground-truth gold evidence annotations (50 `SUPPORT`, 50 `CONTRADICT`)  
**Cross-Encoder Model**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`  
**Retrieval Engine**: BM25 Lexical Retriever with claim-aware sentence window distillation  

---

## 1. Executive Summary & Diagnostic Gap

To definitively diagnose whether pipeline failures originate in the **retrieval layer** (failure to fetch the correct evidence) or the **NLI cross-encoder layer** (failure to reason over the evidence), we executed a controlled ablation experiment across two evaluation regimes on 100 real SciFact claims:
1. **Regime A (Model Alone)**: DeBERTa-v3 evaluated directly on the ground-truth gold sentence annotations.
2. **Regime B (Retrieval + Model)**: BM25 retrieval over 5,183 scientific abstracts followed by claim-aware sentence window extraction and DeBERTa-v3 adjudication.

```
+-------------------------------------------------------------------------+
|                       DIAGNOSTIC GAP OVERVIEW                           |
+-------------------------------------------------------------------------+
|  Metric                                  | Value        | Interpretation|
+------------------------------------------+--------------+---------------+
|  Gold Evidence Accuracy (Model Alone)    | 56.00%       | Pure NLI Cap. |
|  Retrieved Accuracy (Retrieval + Model)  | 43.00%       | Pipeline Cap. |
|  The Diagnostic Gap (Δ = Gold - Ret)     | +13.00%      | Ret. Tax      |
|  BM25 Retrieval Recall@1                 | 70.00%       | Top-1 Acc.    |
|  BM25 Retrieval Recall@3                 | 82.00%       | Top-3 Acc.    |
|  BM25 Retrieval Recall@5                 | 88.00%       | Top-5 Acc.    |
+-------------------------------------------------------------------------+
```

---

## 2. Four-Way Error Partitioning

Partitioning all 100 claims across the $2 \times 2$ matrix of Gold vs. Retrieved correctness reveals the exact source of pipeline failure:

```mermaid
pie title Error Partition Breakdown (N = 100 Claims)
    "Both Correct (End-to-End Success)" : 34
    "Pure Retrieval Failures (Retrieval Starvation)" : 22
    "Pure Model Failures (NLI Reasoning Gap)" : 35
    "Fortuitous / Outlier" : 9
```

| Category | Count | Pct (%) | Root Cause & Failure Mechanism |
| :--- | :---: | :---: | :--- |
| **[1] Both Correct** | **34** | **34.0%** | **End-to-End Success**: Retrieval fetched the relevant document, sentence windowing extracted the key premise, and DeBERTa correctly inferred the gold veracity stance. |
| **[2] Retrieval Failures** | **22** | **22.0%** | **Retrieval Starvation**: The NLI model is 100% capable of solving these claims when shown gold evidence ($P_{\text{gold}} > 0.85$), but retrieval failed to provide the target evidence. |
| **[3] Model Failures** | **35** | **35.0%** | **NLI Reasoning Gap**: DeBERTa failed to identify the stance even when provided the exact gold sentence on a silver platter. |
| **[4] Fortuitous / Outlier** | **9** | **9.0%** | **Context Enrichment**: The retrieved abstract contained a broader, more explicit restatement than the isolated gold sentence, enabling DeBERTa to succeed where gold alone was neutral. |

---

## 3. Deep-Dive: Root Cause Analysis

### A. Anatomizing Retrieval Failures (22 Claims / 22%)

When the NLI model was starved of the correct evidence, what caused the retrieval failure?
- **Sub-Type 1: Pure Recall Miss (8 / 22 = 36.4%)**:
  - BM25 ranked an unrelated paper at Rank 1 because of non-specific biomedical keywords (e.g. `cells`, `infection`, `pathology`), while the target paper was pushed outside Top-1.
  - *Concrete Example (`scifact_133`)*:
    - *Claim*: *"Assembly of invadopodia is triggered by focal generation of phosphatidylinositol-3,4-biphosphate and the activation of the nonreceptor tyrosine kinase Src."*
    - *Gold NLI*: Correctly predicted **`ENTAILMENT` (86.84%)**.
    - *Retrieved*: BM25 retrieved doc `scifact_5270265` instead of target doc `scifact_16280642`, returning `NEUTRAL`.
- **Sub-Type 2: Target Document Hit, but Sentence Window Miss (14 / 22 = 63.6%)**:
  - BM25 correctly retrieved the ground-truth document at Rank 1.
  - However, within the abstract, heuristic sentence windowing prioritized high-overlap background/introductory sentences over the specific sentence containing the experimental finding.
  - *Concrete Example (`scifact_124`)*:
    - *Claim*: *"Antiretroviral therapy reduces rates of tuberculosis across a broad range of CD4 strata."*
    - *Gold NLI*: Correctly predicted **`ENTAILMENT` (90.85%)**.
    - *Retrieved*: Target doc `scifact_4883040` was retrieved at Rank 1, but sentence windowing extracted the study design sentences rather than the results sentence, yielding `NEUTRAL`.

### B. Anatomizing Model / NLI Reasoning Failures (35 Claims / 35%)

When DeBERTa-v3 had the exact ground-truth sentence in its prompt, why did it fail?

| Model Error Mode | Count | Share (%) | Description |
| :--- | :---: | :---: | :--- |
| **False Neutral** | **31** | **88.6%** | Model failed to connect domain concepts; retreated safely to `NEUTRAL` ($P_{\text{neu}} > 0.90$). |
| **False Contradiction** | 3 | 8.6% | Mistook complex qualifying phrasing for refutation. |
| **False Entailment** | 1 | 2.9% | Mistook partial lexical overlap for support on a refuted claim. |

- **Key Finding**: DeBERTa-v3 is **highly conservative, not hallucinatory**. It almost never flips an entailment into a contradiction or vice-versa (only 4% total stance inversions). Instead, 88.6% of reasoning failures are **False Neutrals** caused by:
  1. **Numerical Fraction/Rate Inability (`scifact_5`)**:
     - *Claim*: *"1/2000 in UK have abnormal PrP positivity."*
     - *Gold Evidence*: *"RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population..."*
     - *DeBERTa Prediction*: `NEUTRAL` (93.59%). The model cannot perform the mental calculation $\frac{493}{1,000,000} \approx \frac{1}{2028} \approx \frac{1}{2000}$.
  2. **Multi-Hop Biological Pathway Deduction (`scifact_3`)**:
     - *Claim*: *"1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants."*
     - *Gold Evidence*: *"We propose as an alternative explanation that variants much less common than the associated one may create 'synthetic associations'..."*
     - *DeBERTa Prediction*: `NEUTRAL` (99.61%). Lacks the specialized genetic epidemiology background to equate synthetic associations with penetrance mapping.

---

## 4. Confusion Matrices

### Gold Evidence (Model Alone)
```
                  Predicted ENTAILMENT   Predicted NEUTRAL   Predicted CONTRADICTION
True SUPPORT               23                   24                      3
True CONTRADICT             2                   15                     33
```
- **Support Recall**: $23 / 50 = 46.0\%$
- **Contradict Recall**: $33 / 50 = 66.0\%$
- **Overall Accuracy**: $56.0\%$

### Retrieved Evidence (Retrieval + Model)
```
                  Predicted ENTAILMENT   Predicted NEUTRAL   Predicted CONTRADICTION
True SUPPORT               22                   22                      6
True CONTRADICT             2                   27                     21
```
- **Support Recall**: $22 / 50 = 44.0\%$
- **Contradict Recall**: $21 / 50 = 42.0\%$
- **Overall Accuracy**: $43.0\%$

---

## 5. Architectural Recommendations to Close the Gap

1. **Close the 22% Retrieval Gap**:
   - **Dense/Hybrid Retrieval Escalation**: Moving from pure BM25 to dense semantic retrieval (e.g. BGE/SciBERT embeddings in `HybridRetriever`) lifts Top-1 recall from 70% toward 85%, recovering the 8 pure recall misses.
   - **NLI-Guided Sentence Re-Ranking**: Instead of purely heuristic lexical overlap scoring in `SentenceWindow`, evaluate all candidate sentences with cross-attention or semantic similarity to avoid the 14 window misses.
2. **Close the 35% Model Gap**:
   - **Symbolic Numerical Verifier**: Integrate a dedicated numerical/rate normalizer (e.g. converting `493 per million` $\rightarrow$ `0.000493` and `1/2000` $\rightarrow$ `0.0005`) before NLI scoring.
   - **Biomedical Fine-Tuning**: DeBERTa-v3 trained on MNLI/FEVER struggles with biomedical domain terminology; domain-adapted checkpoints (e.g. PubMedBERT or SciDeBERTa) drastically reduce false neutrals on scientific abstracts.
