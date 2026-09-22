# Experimental Dataset Documentation
**Experiment**: Pre-Module-5 Research Validation Checkpoint  
**Dataset File**: `experiments/datasets/generated/dataset_120_claims.json`  

---

## 1. Overview & Dataset Provenance
This dataset supports the empirical evaluation of evidence dependence, calibration drift, and independence-aware filtering prior to multi-agent debate.

- **Total Claims ($N$)**: 120 claims.
- **Evidence per Claim**: 5 retrieved passages per dependence condition.
- **Dependence Conditions**: 6 levels (0%, 20%, 40%, 60%, 80%, 100%).
- **Total Evaluated Configurations**: $120 \times 6 = 720$ evidence sets ($3,600$ total claim-evidence pairs).
- **Random Seed**: `42` (ensures 100% bit-exact reproducibility).

---

## 2. Domain & Verdict Distribution (§3.1, §3.2)

### 2.1 Domain Breakdown
The dataset spans four distinct factual knowledge domains with exactly 30 claims each:
- **Science**: 30 claims (astrophysics, molecular biology, biochemistry, geology, thermodynamics).
- **History**: 30 claims (American revolution, ancient Rome, medieval England, Napoleonic wars, space race).
- **Technology**: 30 claims (web architecture, computer networking, machine learning, cryptography, databases).
- **News / Contemporary**: 30 claims (European integration, global sports history, international treaties, space policy).

### 2.2 Ground-Truth Verdict Distribution
- **SUPPORTED**: 75 claims (factual claims verified by real empirical evidence).
- **CONTRADICTED**: 45 claims (plausible misconceptions, historical counterfactuals, or refuting claims).

---

## 3. Controlled Dependence Levels (§3.3)

For each claim, ground-truth provenance is systematically modulated across six levels:

| Level | Dependent Items | Independent Items | Provenance Groups | Description |
|---|---|---|---|---|
| **0%** | 0 / 5 | 5 / 5 | 5 | All evidence items originate from mutually independent sources. |
| **20%** | 1 / 5 | 4 / 5 | 4 | 1 syndicated copy of root item 0; roots 2, 3, 4 independent. |
| **40%** | 2 / 5 | 3 / 5 | 3 | 2 syndicated/duplicate copies of root 0; roots 3, 4 independent. |
| **60%** | 3 / 5 | 2 / 5 | 2 | 3 syndicated/duplicate copies of root 0; root 4 independent. |
| **80%** | 4 / 5 | 1 / 5 | 1 | 4 syndicated/duplicate copies of root 0; 1 independent origin. |
| **100%** | 5 / 5 | 0 / 5 | 1 | All 5 evidence items share the exact same root provenance. |

---

## 4. Ground-Truth Provenance Schema (§3.4)

Each evidence item contains both standard retrieved metadata and explicit experimental ground truth:
```json
{
  "evidence_id": "claim_001_ev_root_0",
  "claim_id": "claim_001",
  "text": "The James Webb Space Telescope...",
  "source_url": "https://www.nature.com/articles/claim_001_0",
  "domain": "nature.com",
  "source": "Nature Publishing",
  "published_date": "2023-05-15",
  "provenance_group_id": "group_root_0",
  "parent_source_id": null,
  "evidence_type": "independent",
  "is_independent": true,
  "is_duplicate": false,
  "is_syndicated": false,
  "is_polluted": false,
  "expected_relation": "supports"
}
```

---

## 5. Synthetic Annotation Disclosure & Known Limitations (§3.5)
- **Factual Text Grounding**: The underlying claims and root evidence narratives are based on verified encyclopedic and scientific facts.
- **Synthetic Annotations**: The wire syndication markers, mirror URLs, and `provenance_group_id` labels were synthetically generated in a controlled manner to guarantee exact, measurable ground-truth dependence ratios.
- **Limitations**: The controlled dependence models represent clean duplication and wire-service syndication; natural organic web syndication may feature noisier paraphrasing and commentary.
