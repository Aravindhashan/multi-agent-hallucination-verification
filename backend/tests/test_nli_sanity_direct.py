"""Sanity-test NLI alone, outside the verification pipeline.

Evaluates 10 hand-written pairs (clearly entailed, clearly contradicted, unrelated)
directly through the raw HuggingFace DeBERTa checkpoint and validates:
1. Labels are resolved dynamically from model.config.id2label.
2. Premise is the Evidence and Hypothesis is the Claim (with directional asymmetry check).
3. Softmax is correctly applied to the logits.
4. NLIEngine wrapper produces identical results without silent index swaps.
"""

from typing import Any
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import NLILabel


# ---------------------------------------------------------------------------
# 10 Hand-Written Canonical NLI Test Pairs
# ---------------------------------------------------------------------------
HANDWRITTEN_NLI_PAIRS = [
    # --- Category 1: Clearly Entailed Pairs (4) ---
    {
        "id": "E1_einstein",
        "category": "entailment",
        "evidence_premise": (
            "Albert Einstein was born in Ulm in the Kingdom of Württemberg in the "
            "German Empire on 14 March 1879."
        ),
        "claim_hypothesis": "Albert Einstein was born in Germany.",
        "expected_label": "entailment",
        "min_confidence": 0.90,
    },
    {
        "id": "E2_apollo11",
        "category": "entailment",
        "evidence_premise": (
            "The Apollo 11 lunar module Eagle landed on the Moon on July 20, 1969, "
            "carrying astronauts Neil Armstrong and Buzz Aldrin."
        ),
        "claim_hypothesis": "Apollo 11 landed on the Moon in July 1969.",
        "expected_label": "entailment",
        "min_confidence": 0.90,
    },
    {
        "id": "E3_penicillin",
        "category": "entailment",
        "evidence_premise": (
            "Penicillin was discovered in September 1928 by Scottish bacteriologist "
            "Alexander Fleming at St Mary's Hospital in London."
        ),
        "claim_hypothesis": "Alexander Fleming discovered penicillin in London.",
        "expected_label": "entailment",
        "min_confidence": 0.90,
    },
    {
        "id": "E4_jupiter",
        "category": "entailment",
        "evidence_premise": (
            "Jupiter is the largest planet in our Solar System, with a mass more than "
            "two and a half times that of all the other planets combined."
        ),
        "claim_hypothesis": "Jupiter is the biggest planet in the Solar System.",
        "expected_label": "entailment",
        "min_confidence": 0.90,
    },
    # --- Category 2: Clearly Contradicted Pairs (3) ---
    {
        "id": "C1_everest",
        "category": "contradiction",
        "evidence_premise": (
            "Mount Everest is the highest mountain on Earth above sea level, with an "
            "official elevation of 8,848.86 meters."
        ),
        "claim_hypothesis": "Mount Everest is the lowest mountain on Earth.",
        "expected_label": "contradiction",
        "min_confidence": 0.90,
    },
    {
        "id": "C2_water_boiling",
        "category": "contradiction",
        "evidence_premise": (
            "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius "
            "under standard atmospheric pressure."
        ),
        "claim_hypothesis": "Water boils at 20 degrees Celsius under normal pressure.",
        "expected_label": "contradiction",
        "min_confidence": 0.90,
    },
    {
        "id": "C3_french_revolution",
        "category": "contradiction",
        "evidence_premise": (
            "The French Revolution began in 1789 with the Storming of the Bastille "
            "and ended in the late 1790s with the rise of Napoleon Bonaparte."
        ),
        "claim_hypothesis": "The French Revolution began in the year 1995.",
        "expected_label": "contradiction",
        "min_confidence": 0.90,
    },
    # --- Category 3: Unrelated / Neutral Pairs (3) ---
    {
        "id": "N1_mona_lisa",
        "category": "neutral",
        "evidence_premise": (
            "The Mona Lisa is a half-length portrait painting by Italian artist "
            "Leonardo da Vinci, considered an archetypal masterpiece of the Renaissance."
        ),
        "claim_hypothesis": "Leonardo da Vinci loved drinking Italian espresso in the afternoon.",
        "expected_label": "neutral",
        "min_confidence": 0.85,
    },
    {
        "id": "N2_ocean_mozart",
        "category": "neutral",
        "evidence_premise": (
            "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, "
            "covering over 165 million square kilometers."
        ),
        "claim_hypothesis": "Wolfgang Amadeus Mozart composed his first symphony at the age of eight.",
        "expected_label": "neutral",
        "min_confidence": 0.85,
    },
    {
        "id": "N3_python_guido",
        "category": "neutral",
        "evidence_premise": (
            "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde "
            "& Informatica (CWI) in the Netherlands as a successor to ABC."
        ),
        "claim_hypothesis": "Guido van Rossum owns three golden retriever dogs.",
        "expected_label": "neutral",
        "min_confidence": 0.85,
    },
]


@pytest.fixture(scope="module")
def raw_model_and_tokenizer() -> tuple[Any, Any, dict[int, str]]:
    """Load real DeBERTa model and tokenizer directly outside the pipeline."""
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Dynamically extract label mapping from model config
    raw_id2label = getattr(model.config, "id2label", None)
    assert raw_id2label is not None, "model.config.id2label must be present."
    dynamic_id2label = {int(k): str(v).strip().lower() for k, v in raw_id2label.items()}

    # Assert all 3 standard NLI labels are present
    assert set(dynamic_id2label.values()) == {"entailment", "neutral", "contradiction"}, (
        f"Unexpected labels in model.config.id2label: {dynamic_id2label}"
    )

    return model, tokenizer, dynamic_id2label


@pytest.mark.slow
def test_raw_deberta_ten_handwritten_pairs(raw_model_and_tokenizer):
    """Sanity-test all 10 hand-written pairs directly through the raw DeBERTa model."""
    model, tokenizer, dynamic_id2label = raw_model_and_tokenizer

    # Confirm dynamic mapping values
    entail_idx = [k for k, v in dynamic_id2label.items() if v == "entailment"][0]
    neutral_idx = [k for k, v in dynamic_id2label.items() if v == "neutral"][0]
    contra_idx = [k for k, v in dynamic_id2label.items() if v == "contradiction"][0]

    for item in HANDWRITTEN_NLI_PAIRS:
        premise = item["evidence_premise"]
        hypothesis = item["claim_hypothesis"]
        expected = item["expected_label"]
        min_conf = item["min_confidence"]

        # Tokenize with Premise = Evidence (1st text), Hypothesis = Claim (2nd text_pair)
        inputs = tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Softmax applied across logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        predicted_idx = int(torch.argmax(logits, dim=-1).item())
        predicted_label = dynamic_id2label[predicted_idx]
        predicted_conf = probs[predicted_idx]

        assert predicted_label == expected, (
            f"Failed on pair {item['id']}: expected {expected}, got {predicted_label} "
            f"(probs: entailment={probs[entail_idx]:.4f}, neutral={probs[neutral_idx]:.4f}, "
            f"contradiction={probs[contra_idx]:.4f})"
        )
        assert predicted_conf >= min_conf, (
            f"Confidence below threshold on pair {item['id']}: {predicted_conf:.4f} < {min_conf}"
        )


@pytest.mark.slow
def test_premise_evidence_hypothesis_claim_directionality(raw_model_and_tokenizer):
    """Explicitly verify directional asymmetry: Premise=Evidence vs Premise=Claim."""
    model, tokenizer, dynamic_id2label = raw_model_and_tokenizer
    entail_idx = [k for k, v in dynamic_id2label.items() if v == "entailment"][0]

    # Asymmetric test case:
    # Premise: Specific birth place (Ulm, Württemberg, Germany, 1879)
    # Hypothesis: General claim (Born in Germany)
    evidence = "Albert Einstein was born in Ulm in the Kingdom of Württemberg in the German Empire on 14 March 1879."
    claim = "Albert Einstein was born in Germany."

    # Direction A (Correct): Premise = Evidence, Hypothesis = Claim
    inputs_correct = tokenizer(evidence, claim, return_tensors="pt")
    with torch.no_grad():
        probs_correct = torch.softmax(model(**inputs_correct).logits, dim=-1)[0].tolist()
    p_entail_correct = probs_correct[entail_idx]

    # Direction B (Reversed): Premise = Claim, Hypothesis = Evidence
    inputs_reversed = tokenizer(claim, evidence, return_tensors="pt")
    with torch.no_grad():
        probs_reversed = torch.softmax(model(**inputs_reversed).logits, dim=-1)[0].tolist()
    p_entail_reversed = probs_reversed[entail_idx]

    # Direction A MUST entail with high confidence (> 0.95)
    assert p_entail_correct > 0.95, (
        f"Correct direction failed to entail: p_entail={p_entail_correct:.4f}"
    )

    # Direction B MUST NOT entail (< 0.05), because knowing someone was born in Germany
    # does NOT entail they were born in Ulm on 14 March 1879.
    assert p_entail_reversed < 0.05, (
        f"Reversed direction erroneously entailed: p_entail={p_entail_reversed:.4f}"
    )


@pytest.mark.slow
def test_nli_engine_wrapper_parity_with_raw_model(raw_model_and_tokenizer):
    """Verify NLIEngine wrapper produces identical results and respects model.config.id2label."""
    model, tokenizer, dynamic_id2label = raw_model_and_tokenizer

    # Instantiate wrapper using preloaded model and tokenizer
    engine = NLIEngine(
        config=NLISettings(max_length=256, truncation_policy="truncate"),
        model=model,
        tokenizer=tokenizer,
    )

    # 1. Check index alignment
    for idx, lbl_str in dynamic_id2label.items():
        expected_enum = NLILabel(lbl_str)
        assert engine.label_mapping[idx] == expected_enum, (
            f"Wrapper label_mapping mismatch at index {idx}: "
            f"expected {expected_enum}, got {engine.label_mapping[idx]}"
        )

    # 2. Check predict and predict_batch on all 10 pairs
    eval_pairs = [(item["claim_hypothesis"], item["evidence_premise"]) for item in HANDWRITTEN_NLI_PAIRS]
    batch_results = engine.predict_batch(eval_pairs)

    assert len(batch_results) == len(HANDWRITTEN_NLI_PAIRS)

    for i, item in enumerate(HANDWRITTEN_NLI_PAIRS):
        res = batch_results[i]
        expected_enum = NLILabel(item["expected_label"])
        assert res.label == expected_enum, (
            f"Wrapper failed on pair {item['id']}: expected {expected_enum}, got {res.label}"
        )
        assert res.confidence >= item["min_confidence"], (
            f"Wrapper confidence below threshold on pair {item['id']}: {res.confidence} < {item['min_confidence']}"
        )
