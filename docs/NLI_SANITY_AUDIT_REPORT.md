# NLI Engine Standalone Sanity Audit & Directionality Verification Report

## 1. Executive Summary

This report documents the standalone sanity testing of the DeBERTa-v3 cross-encoder NLI engine outside the verification pipeline to audit against silent failure modes:
1. **Dynamic Label Resolution**: Verifying labels are read dynamically from `model.config.id2label` rather than assuming hardcoded index orders.
2. **Premise/Hypothesis Directionality**: Verifying that `Premise == Evidence` and `Hypothesis == Claim` (demonstrating directional asymmetry).
3. **Logits Softmax Calibration**: Verifying that `torch.softmax(logits, dim=-1)` is applied across logits.
4. **Standalone Empirical Benchmark**: Evaluating 10 hand-written canonical pairs (Entailment, Contradiction, Neutral/Unrelated) directly on the raw checkpoint (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`).
5. **Wrapper Parity**: Verifying that the production `NLIEngine` wrapper produces bitwise identical predictions to the raw model.

---

## 2. Dynamic Label Extraction (`model.config.id2label`)

Inspection of the loaded checkpoint config revealed:
```python
model.config.id2label = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}
model.config.label2id = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}
```

### Silent Index-Swap Risk Assessment
In some NLI checkpoints (e.g., standard MNLI vs. RoBERTa SNLI variants), the contradiction and neutral indices are reversed (`{0: contradiction, 1: neutral, 2: entailment}`). If a wrapper hardcodes `[entailment, contradiction, neutral]`, an entailment or contradiction verdict would be silently inverted.

**Code Invariant in `NLIEngine` ([`app/services/nli_engine.py`](file:///c:/Users/aravi/Downloads/HallucinationDetection_NLP/hallucination_verifier/backend/app/services/nli_engine.py))**:
```python
def _resolve_label_mapping(self) -> None:
    config_obj = getattr(self.model, "config", None)
    id2label = getattr(config_obj, "id2label", None) if config_obj else None
    if id2label and isinstance(id2label, dict):
        for raw_idx, raw_lbl in id2label.items():
            idx = int(raw_idx)
            lbl_lower = str(raw_lbl).strip().lower()
            if "entail" in lbl_lower:
                resolved[idx] = NLILabel.ENTAILMENT
            elif "contradict" in lbl_lower:
                resolved[idx] = NLILabel.CONTRADICTION
            elif "neutral" in lbl_lower:
                resolved[idx] = NLILabel.NEUTRAL
    self.label_mapping = resolved
    self.entailment_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.ENTAILMENT][0]
    self.neutral_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.NEUTRAL][0]
    self.contradiction_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.CONTRADICTION][0]
```
The wrapper dynamically parses `id2label` using case-insensitive substring matching, completely immunizing the system against index transposition bugs.

---

## 3. Premise/Hypothesis Directional Asymmetry Check

In cross-encoder NLI, sequence concatenation order is strictly directional:
$$\text{Tokenizer Input: } [CLS] \circ \text{Premise} \circ [SEP] \circ \text{Hypothesis} \circ [SEP]$$
* **Premise**: The assumed ground truth / reference text (**Evidence**).
* **Hypothesis**: The proposition being verified (**Claim**).

### Empirical Directionality Test
We tested an asymmetric factual statement where the evidence is more specific than the claim:
* **Evidence**: *"Albert Einstein was born in Ulm in the Kingdom of Württemberg in the German Empire on 14 March 1879."*
* **Claim**: *"Albert Einstein was born in Germany."*

| Direction | Tokenizer Input | Entailment | Neutral | Contradiction | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Correct** | `tokenizer(evidence, claim)` | **0.9941** | 0.0051 | 0.0009 | **ENTAILMENT (99.4%)** |
| **Reversed** | `tokenizer(claim, evidence)` | 0.0028 | **0.9971** | 0.0003 | **NEUTRAL (99.7%)** |

**Crucial Finding**:
Reversing the order causes Entailment to collapse from **99.41% to 0.28%**, turning the verdict into **99.71% Neutral**. This occurs because knowing someone was born in Germany does *not* entail they were born in Ulm on March 14, 1879. 

In [`backend/app/services/nli_engine.py`](file:///c:/Users/aravi/Downloads/HallucinationDetection_NLP/hallucination_verifier/backend/app/services/nli_engine.py):
```python
# In standard NLI cross-encoders: Premise = Evidence, Hypothesis = Claim
premises = [e for c, e in batch_slice]
hypotheses = [c for c, e in batch_slice]
tokens = self.tokenizer(premises, hypotheses, ...)
```
The evidence is strictly placed into `premises` (first argument) and the claim into `hypotheses` (second argument).

---

## 4. Empirical Evaluation: 10 Hand-Written Canonical Pairs

All 10 pairs were evaluated directly on the raw model weights:

| ID | Category | Premise (Evidence) | Hypothesis (Claim) | Expected | Predicted | P(Entail) | P(Neutral) | P(Contra) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Entailment | Albert Einstein was born in Ulm in the Kingdom of Württemberg in the German Empire on 14 March 1879. | Albert Einstein was born in Germany. | `entailment` | `entailment` | **0.9941** | 0.0051 | 0.0009 | ✅ PASSED |
| **E2** | Entailment | The Apollo 11 lunar module Eagle landed on the Moon on July 20, 1969, carrying astronauts Neil Armstrong and Buzz Aldrin. | Apollo 11 landed on the Moon in July 1969. | `entailment` | `entailment` | **0.9980** | 0.0014 | 0.0004 | ✅ PASSED |
| **E3** | Entailment | Penicillin was discovered in September 1928 by Scottish bacteriologist Alexander Fleming at St Mary's Hospital in London. | Alexander Fleming discovered penicillin in London. | `entailment` | `entailment` | **0.9966** | 0.0029 | 0.0006 | ✅ PASSED |
| **E4** | Entailment | Jupiter is the largest planet in our Solar System, with a mass more than two and a half times that of all the other planets combined. | Jupiter is the biggest planet in the Solar System. | `entailment` | `entailment` | **0.9951** | 0.0041 | 0.0007 | ✅ PASSED |
| **C1** | Contradiction | Mount Everest is the highest mountain on Earth above sea level, with an official elevation of 8,848.86 meters. | Mount Everest is the lowest mountain on Earth. | `contradiction` | `contradiction` | 0.0004 | 0.0005 | **0.9990** | ✅ PASSED |
| **C2** | Contradiction | Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius under standard atmospheric pressure. | Water boils at 20 degrees Celsius under normal pressure. | `contradiction` | `contradiction` | 0.0009 | 0.0013 | **0.9976** | ✅ PASSED |
| **C3** | Contradiction | The French Revolution began in 1789 with the Storming of the Bastille and ended in the late 1790s with the rise of Napoleon Bonaparte. | The French Revolution began in the year 1995. | `contradiction` | `contradiction` | 0.0001 | 0.0003 | **0.9995** | ✅ PASSED |
| **N1** | Neutral | The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci, considered an archetypal masterpiece of the Renaissance. | Leonardo da Vinci loved drinking Italian espresso in the afternoon. | `neutral` | `neutral` | 0.0003 | **0.9995** | 0.0003 | ✅ PASSED |
| **N2** | Neutral | The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, covering over 165 million square kilometers. | Wolfgang Amadeus Mozart composed his first symphony at the age of eight. | `neutral` | `neutral` | 0.0029 | **0.9966** | 0.0005 | ✅ PASSED |
| **N3** | Neutral | Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to ABC. | Guido van Rossum owns three golden retriever dogs. | `neutral` | `neutral` | 0.0002 | **0.9990** | 0.0007 | ✅ PASSED |

**Benchmark Accuracy**: **10 / 10 (100.0%)** with mean winning confidence $> 99.0\%$.

---

## 5. Automated Regression Test Suite

An automated test suite was added to [`backend/tests/test_nli_sanity_direct.py`](file:///c:/Users/aravi/Downloads/HallucinationDetection_NLP/hallucination_verifier/backend/tests/test_nli_sanity_direct.py) containing:
1. `test_raw_deberta_ten_handwritten_pairs`: Tests the raw model checkpoint on all 10 pairs using dynamic `id2label` mapping and `torch.softmax`.
2. `test_premise_evidence_hypothesis_claim_directionality`: Asserts that `(evidence, claim)` yields Entailment $> 95\%$ while `(claim, evidence)` yields Entailment $< 5\%$.
3. `test_nli_engine_wrapper_parity_with_raw_model`: Asserts that `NLIEngine` produces bitwise-identical results and dynamically synchronizes label indices from `model.config.id2label`.

**Test Execution Result**:
```
tests/test_nli_sanity_direct.py::test_raw_deberta_ten_handwritten_pairs PASSED [ 33%]
tests/test_nli_sanity_direct.py::test_premise_evidence_hypothesis_claim_directionality PASSED [ 66%]
tests/test_nli_sanity_direct.py::test_nli_engine_wrapper_parity_with_raw_model PASSED [100%]
================== 3 passed, 1 warning in 129.86s ===================
```
