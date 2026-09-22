# Pre-Registration Document
**Experiment**: Pre-Module-5 Empirical Research Validation  
**Project**: Dynamic Adversarial Multi-Agent Framework for Explainable Hallucination Verification in Large Language Models  
**Timestamp**: 2026-09-17T07:37:00+05:30 (Registered prior to experimental execution)  
**Status**: LOCKED  

---

## 1. Central Research Hypothesis
> **Hypothesis**: When multiple retrieved evidence documents are correlated or derived from the same underlying source, conventional evidence aggregation becomes overconfident. Independence-aware aggregation reduces this overconfidence and maintains better calibration as dependent evidence increases.

The primary focus is **calibration robustness under dependent evidence**, not raw classification accuracy.

---

## 2. Pre-Registered Metrics & Test Specifications

### 2.1 Primary Metric
- **Expected Calibration Error (ECE) at 60% Dependent-Evidence Condition**:
  Computed across 10 equal-width confidence bins $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$:
  $$\text{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
  where $\text{conf}(B_b)$ is the average model confidence in bin $B_b$, and $\text{acc}(B_b)$ is the empirical accuracy of the predicted verdict against ground truth.

### 2.2 Primary Statistical Test
- **Test**: Paired Wilcoxon signed-rank test on per-claim calibration error deltas:
  $$\Delta_i = \text{CE}_{A1}^{(i)} - \text{CE}_{A2}^{(i)}$$
  between System A1 (Dedup-only) and System A2 (Independence-aware) at the 60% dependence level.
- **Sample Size**: $N = 120$ claims ($N \ge 100$ requirement satisfied).

### 2.3 Pre-Registered Decision Thresholds for "Supported"
To classify the hypothesis as **SUPPORTED**, the experiment must satisfy BOTH criteria:
1. **Effect Size**: Median per-claim ECE reduction $\ge 0.05$ absolute ($\text{ECE}_{A1} - \text{ECE}_{A2} \ge 0.05$).
2. **Statistical Significance**: $p < 0.05$ after Holm–Bonferroni correction across all 6 dependence levels (0%, 20%, 40%, 60%, 80%, 100%).

If either condition fails, the hypothesis is classified strictly as **PARTIALLY SUPPORTED** (if significant but below 0.05 effect size, or vice versa) or **NOT SUPPORTED**.

---

## 3. Secondary Metrics & Ablations
1. **Brier Score**: $\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$.
2. **Mean Confidence**: $\bar{C} = \frac{1}{N} \sum_{i=1}^N \hat{c}_i$ across dependence levels (testing confidence inflation).
3. **Kish Effective Sample Size**:
   $$n_{\text{eff}} = \frac{\left(\sum w_i\right)^2}{\sum w_i^2}$$
   where $w_i = 1/k$ for an item in a redundancy cluster of size $k$, and $w_i = 1.0$ for independent items.
4. **Systems Compared**:
   - **A0**: All evidence (no filtering ceiling).
   - **A1**: Dedup-only (exact normalized text / URL deduplication).
   - **A2**: Independence-aware (Module 4.3 complete-linkage clustering, 1 representative per cluster).
   - **A3**: Independence + Pollution (A2 with Module 4.4 ungrounded / self-referential gate).

---

## 4. Shared Aggregation Rule
For any surviving evidence set $\{e_1, \dots, e_k\}$ for claim $c$:
$$\bar{P}_{\text{ent}} = \frac{1}{k} \sum_{i=1}^k P(\text{ENTAILMENT} \mid e_i, c), \quad \bar{P}_{\text{con}} = \frac{1}{k} \sum_{i=1}^k P(\text{CONTRADICTION} \mid e_i, c)$$
- Verdict: `SUPPORTED` if $\bar{P}_{\text{ent}} > 0.5$ and $\bar{P}_{\text{ent}} > \bar{P}_{\text{con}}$; `CONTRADICTED` if $\bar{P}_{\text{con}} \ge 0.5$; `NEUTRAL` otherwise.
- Confidence: $\hat{c} = \max(\bar{P}_{\text{ent}}, \bar{P}_{\text{con}}, 1.0 - (\bar{P}_{\text{ent}} + \bar{P}_{\text{con}}))$.

---

## 5. Non-Interference Commitment
- Thresholds are locked before running.
- No post-hoc data filtering or threshold re-tuning.
- Negative results will be reported with equal scientific weight.
