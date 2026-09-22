"""Tests for Step 4.4 Evidence Pollution & Circularity Defense.

Covers exact duplication, self-reference, ungrounded evidence, directional asymmetry,
short text policy, Anti-Fabrication Contract verification, precedence monotonicity,
determinism, clock isolation, batch evaluation, and benchmarks.
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Any
import pytest

from app.services.corpus_schema import CorpusDocument
from app.services.retrieval_models import RetrievalResult
from app.services.evidence_pollution_config import (
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DEFAULT_UNGROUNDED_THRESHOLD,
    DEFAULT_WEAK_GROUNDING_THRESHOLD,
    DISCLAIMER_PHRASE,
    EvidencePollutionConfigError,
    EvidencePollutionInputError,
    EvidencePollutionSettings,
)
from app.services.evidence_pollution_models import (
    BatchPollutionResult,
    EvidencePollutionResult,
    PollutionClass,
)
from app.services.evidence_pollution_detector import (
    EvidencePollutionDetector,
    compute_directional_overlap,
    compute_grounding_score,
    extract_meaningful_tokens,
    get_evidence_pollution_detector,
    reset_evidence_pollution_detector,
)


@pytest.fixture(autouse=True)
def cleanup_detector() -> None:
    """Reset singleton detector before and after each test."""
    reset_evidence_pollution_detector()
    yield
    reset_evidence_pollution_detector()


# =====================================================================
# 1. Anti-Fabrication Contract & Boundary Tests (§1.1, §5)
# =====================================================================


def test_anti_fabrication_flags_and_disclaimer() -> None:
    """Ensure circularity_supported is False and required disclaimer is in every explanation."""
    detector = EvidencePollutionDetector()
    claim = "The Hubble Space Telescope was launched in 1990."
    evidence = {
        "document_id": "hub_1",
        "text": "The Hubble Space Telescope was deployed into low Earth orbit in April 1990.",
    }

    res = detector.evaluate(claim, evidence)
    assert res.circularity_supported is False
    assert res.is_circular is False
    assert DISCLAIMER_PHRASE in res.explanation
    assert res.disclaimer == DISCLAIMER_PHRASE


def test_anti_fabrication_no_invented_fields() -> None:
    """Assert models do not contain fabricated fields like citation_graph or canonical_doc_id."""
    fields = EvidencePollutionResult.model_fields.keys()
    assert "citation_graph" not in fields
    assert "canonical_document_id" not in fields
    assert "parent_document" not in fields
    assert "republication_id" not in fields


# =====================================================================
# 2. Self-Referential & Generated Content Detection (§6)
# =====================================================================


def test_self_referential_identical_claim_text() -> None:
    """Evidence identically repeating a meaningful claim is classified as SELF_REFERENTIAL."""
    detector = EvidencePollutionDetector()
    claim = "James Webb Space Telescope observed galaxy candidate JADES-GS-z14-0 in early universe."
    evidence = {"document_id": "ev_self", "text": claim}

    res = detector.evaluate(claim, evidence)
    assert res.classification == PollutionClass.SELF_REFERENTIAL
    assert res.pollution_score == 1.0
    assert res.is_self_referential is True
    assert res.is_exact_duplicate is True


def test_self_referential_claim_origin_url_match() -> None:
    """Evidence matching the claim's own origin URL is classified as SELF_REFERENTIAL."""
    detector = EvidencePollutionDetector()
    claim = "Atmospheric methane on Mars was measured by Curiosity rover."
    origin_url = "https://mars.nasa.gov/news/curiosity-methane-report/"
    evidence = {
        "document_id": "ev_url",
        "text": "In 2019, NASA Curiosity rover recorded a high spike of methane in Gale crater.",
        "source_url": "https://www.mars.nasa.gov/news/curiosity-methane-report#findings",
    }

    res = detector.evaluate(claim, evidence, claim_origin_url=origin_url)
    assert res.classification == PollutionClass.SELF_REFERENTIAL
    assert res.pollution_score == 1.0
    assert res.is_self_referential is True


def test_high_lexical_overlap_alone_is_not_polluted() -> None:
    """Non-Rule (§6.2): Legitimate well-grounded evidence with high lexical overlap is CLEAN."""
    detector = EvidencePollutionDetector()
    claim = "Photosynthesis in cyanobacteria produces oxygen and fixes atmospheric carbon dioxide."
    evidence = {
        "document_id": "bio_1",
        "text": "Photosynthesis in ancient cyanobacteria produced molecular oxygen and fixed atmospheric carbon dioxide over geological epochs.",
        "domain": "nature.com",
    }

    res = detector.evaluate(claim, evidence)
    assert res.claim_evidence_overlap >= 0.70
    assert res.grounding_score >= 0.70
    assert res.classification == PollutionClass.CLEAN
    assert res.pollution_score == 0.0
    assert res.is_self_referential is False


# =====================================================================
# 3. Directional Asymmetry & Grounding Calculation (§6.1, §7)
# =====================================================================


def test_directional_overlap_asymmetry() -> None:
    """Proof that Overlap(C, E) != Overlap(E, C) for asymmetric lengths (§6.1, §21)."""
    # Short claim with 3 tokens; long evidence with 15 tokens
    tokens_c = frozenset(["quantum", "supremacy", "experiment"])
    tokens_e = frozenset([
        "quantum", "supremacy", "experiment", "demonstrated", "by", "google",
        "sycamore", "processor", "sampling", "random", "circuits", "in", "seconds"
    ])

    overlap_c_e = compute_directional_overlap(tokens_c, tokens_e)
    overlap_e_c = compute_directional_overlap(tokens_e, tokens_c)

    assert overlap_c_e == 1.0  # All 3 claim tokens are in the evidence
    assert pytest.approx(overlap_e_c, rel=1e-4) == 3.0 / 13.0  # Only 3 of 13 evidence tokens in claim
    assert overlap_c_e != overlap_e_c


def test_grounding_score_excludes_stopwords() -> None:
    """Grounding excludes common grammatical stopwords from meaningful token counts."""
    claim = "The speed of light in a vacuum is constant."
    meaningful = extract_meaningful_tokens(claim)
    # Excluded: "the", "of", "in", "a", "is"
    # Preserved: "speed", "light", "vacuum", "constant"
    assert "the" not in meaningful
    assert "is" not in meaningful
    assert "speed" in meaningful
    assert "light" in meaningful
    assert "vacuum" in meaningful
    assert "constant" in meaningful


def test_ungrounded_evidence_classification() -> None:
    """Evidence lacking lexical grounding to the claim (< 0.15) is classified as UNGROUNDED."""
    detector = EvidencePollutionDetector(EvidencePollutionSettings(ungrounded_threshold=0.15))
    claim = "Photosynthesis requires chlorophyll pigments to absorb photons from sunlight."
    evidence = {
        "document_id": "history_1",
        "text": "The Roman Empire expanded across the Mediterranean basin establishing provincial administrations.",
    }

    res = detector.evaluate(claim, evidence)
    assert res.grounding_score == 0.0
    assert res.classification == PollutionClass.UNGROUNDED
    assert res.pollution_score == 0.80
    assert "Lexical grounding is a lexical filter and not semantic entailment" in res.explanation


def test_weak_grounding_classification() -> None:
    """Evidence with moderate lexical grounding (between 0.15 and 0.35) is WEAK_POLLUTION_SIGNAL."""
    detector = EvidencePollutionDetector(
        EvidencePollutionSettings(ungrounded_threshold=0.15, weak_grounding_threshold=0.35)
    )
    claim = "Galileo observed four large moons orbiting Jupiter using an optical telescope."
    # Meaningful claim tokens: galileo, observed, four, large, moons, orbiting, jupiter, optical, telescope (9 tokens)
    # Evidence shares 2 tokens ("telescope", "observed") -> 2/9 = 0.222 (between 0.15 and 0.35)
    evidence = {
        "document_id": "optics_1",
        "text": "Early optical glass instruments were observed and manufactured in Renaissance Venice workshops.",
    }

    res = detector.evaluate(claim, evidence)
    assert 0.15 <= res.grounding_score < 0.35
    assert res.classification == PollutionClass.WEAK_POLLUTION_SIGNAL
    assert res.pollution_score == 0.30


# =====================================================================
# 4. Short Text Policy & Edge Cases (§10)
# =====================================================================


def test_short_text_policy_identical_yes() -> None:
    """'yes' vs 'yes' must NOT become strong pollution solely because overlap is 100% (§10)."""
    detector = EvidencePollutionDetector(EvidencePollutionSettings(minimum_meaningful_tokens=5))
    claim = "yes"
    evidence = {"document_id": "short_1", "text": "yes"}

    res = detector.evaluate(claim, evidence)
    assert res.claim_evidence_overlap == 1.0
    assert res.classification == PollutionClass.WEAK_POLLUTION_SIGNAL
    assert res.classification != PollutionClass.SELF_REFERENTIAL
    assert res.pollution_score == 0.30


def test_empty_claim_and_evidence() -> None:
    """Empty strings safely return 0.0 overlap and conservative classifications without crash."""
    detector = EvidencePollutionDetector()
    res = detector.evaluate("", {"document_id": "empty_ev", "text": ""})
    assert res.claim_evidence_overlap == 0.0
    assert res.grounding_score == 0.0
    assert res.classification == PollutionClass.WEAK_POLLUTION_SIGNAL


def test_malformed_input_raises_typed_error() -> None:
    """None inputs or missing document ID/text raise EvidencePollutionInputError."""
    detector = EvidencePollutionDetector()

    with pytest.raises(EvidencePollutionInputError):
        detector.evaluate(None, {"document_id": "a", "text": "text"})

    with pytest.raises(EvidencePollutionInputError):
        detector.evaluate("valid claim", None)

    with pytest.raises(EvidencePollutionInputError):
        detector.evaluate("valid claim", {"text": "missing document_id"})

    with pytest.raises(EvidencePollutionInputError):
        detector.evaluate("valid claim", {"document_id": "a"})


# =====================================================================
# 5. Precedence & Monotonicity (§8.1, §8.3, §9)
# =====================================================================


def test_precedence_dominance_exact_duplicate_over_ungrounded() -> None:
    """If evidence is both ungrounded and a repeated URL, EXACT_DUPLICATE dominates (§8.1, §9)."""
    detector = EvidencePollutionDetector()
    claim = "General relativity explains gravitational time dilation in black holes."
    evidence_1 = {
        "document_id": "doc_1",
        "text": "Gardening tips for springtime tomato planting.",
        "source_url": "https://gardening.org/tomatoes",
    }
    evidence_2 = {
        "document_id": "doc_2",
        "text": "Gardening tips for springtime tomato planting in pots.",
        "source_url": "https://gardening.org/tomatoes/",
    }

    # Evaluate in batch so evidence_2 encounters repeated URL
    batch_res = detector.evaluate_batch(claim, [evidence_1, evidence_2])

    res_1 = batch_res.evaluations[0]
    res_2 = batch_res.evaluations[1]

    assert res_1.classification == PollutionClass.UNGROUNDED
    assert res_1.pollution_score == 0.80

    # evidence_2 has the same URL, so EXACT_DUPLICATE (severity 1.0) dominates over UNGROUNDED (0.80)
    assert res_2.classification == PollutionClass.EXACT_DUPLICATE
    assert res_2.pollution_score == 1.0


def test_precedence_score_monotonicity() -> None:
    """Scores are monotonic in precedence and bounded in [0.0, 1.0]."""
    detector = EvidencePollutionDetector()
    claim = "Deep neural networks optimize loss functions via backpropagation."

    # 1. Clean
    ev_clean = {"document_id": "c1", "text": "Deep neural networks calculate analytical gradients using backpropagation algorithms."}
    r_clean = detector.evaluate(claim, ev_clean)
    assert r_clean.pollution_score == 0.0

    # 2. Weak (grounding between 0.15 and 0.35: shares 'networks' and 'optimize' -> 2/8 = 0.25)
    ev_weak = {"document_id": "w1", "text": "Complex networks optimize parameter weights across varied domains."}
    r_weak = detector.evaluate(claim, ev_weak)
    assert r_weak.pollution_score == 0.30

    # 3. Ungrounded
    ev_ungrounded = {"document_id": "u1", "text": "Ancient pottery fragments discovered in archaeological excavations."}
    r_ungrounded = detector.evaluate(claim, ev_ungrounded)
    assert r_ungrounded.pollution_score == 0.80

    # 4. Self-referential
    ev_self = {"document_id": "s1", "text": claim}
    r_self = detector.evaluate(claim, ev_self)
    assert r_self.pollution_score == 1.0

    assert r_clean.pollution_score < r_weak.pollution_score < r_ungrounded.pollution_score < r_self.pollution_score


# =====================================================================
# 6. URL Normalization Variants (§12)
# =====================================================================


def test_url_normalization_variants() -> None:
    """Verify canonical URL normalization across case, www, ports, fragments, and trailing slashes."""
    detector = EvidencePollutionDetector()
    claim = "Mars rover perseverance landed in Jezero crater."
    origin_url = "https://WWW.NASA.GOV:443/perseverance/mission/#overview"
    evidence = {
        "document_id": "ev_1",
        "text": "NASA Perseverance rover landed on Mars inside Jezero crater.",
        "source_url": "http://nasa.gov:80/perseverance/mission",
    }

    # Although schemes differ (https vs http), host lowercase, www stripped, port 80/443 stripped, fragment stripped
    # If schemes match:
    ev_matching = {
        "document_id": "ev_2",
        "text": "NASA Perseverance rover landed on Mars inside Jezero crater.",
        "source_url": "https://nasa.gov/perseverance/mission/",
    }
    res = detector.evaluate(claim, ev_matching, claim_origin_url=origin_url)
    assert res.classification == PollutionClass.SELF_REFERENTIAL
    assert res.is_self_referential is True


# =====================================================================
# 7. Determinism, Clock Isolation, and Immutability (§16)
# =====================================================================


def test_determinism_identical_runs() -> None:
    """Fresh detector instances return byte-equal results and explanations."""
    det1 = EvidencePollutionDetector()
    det2 = EvidencePollutionDetector()

    claim = "CRISPR-Cas9 enables targeted genome editing in living organisms."
    evidence = {
        "document_id": "crispr_1",
        "text": "CRISPR-Cas9 endonuclease complexes introduce targeted double-strand DNA breaks.",
    }

    r1 = det1.evaluate(claim, evidence)
    r2 = det2.evaluate(claim, evidence)

    assert r1.pollution_score == r2.pollution_score
    assert r1.classification == r2.classification
    assert r1.grounding_score == r2.grounding_score
    assert r1.explanation == r2.explanation


def test_system_clock_monkeypatch_does_not_affect_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatching date.today confirms complete clock independence."""
    fake_date = datetime.date(2099, 1, 1)

    class FakeDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            return fake_date

    monkeypatch.setattr(datetime, "date", FakeDate)

    detector = EvidencePollutionDetector()
    res = detector.evaluate("Test claim sentence for clock isolation.", {"document_id": "d1", "text": "Test evidence sentence for clock isolation."})
    assert isinstance(res, EvidencePollutionResult)
    assert "2099" not in res.explanation


def test_input_immutability() -> None:
    """Ensure raw claim and evidence dictionary are not mutated during evaluation."""
    detector = EvidencePollutionDetector()
    raw_claim = "  Unnormalized  Claim  String \n"
    raw_evidence = {"document_id": "imm_1", "text": "  Unnormalized  Evidence  Text \t"}

    _ = detector.evaluate(raw_claim, raw_evidence)
    assert raw_claim == "  Unnormalized  Claim  String \n"
    assert raw_evidence["text"] == "  Unnormalized  Evidence  Text \t"


# =====================================================================
# 8. Batch Evaluation (§15)
# =====================================================================


def test_evaluate_batch_canonical_ordering_and_grouping() -> None:
    """Batch evaluation processes evidence in input order and categorizes document IDs."""
    detector = EvidencePollutionDetector()
    claim = "James Webb Space Telescope observes infrared light from distant galaxies."
    items = [
        {"document_id": "clean_1", "text": "James Webb observes infrared radiation from distant cosmic galaxies."},
        {"document_id": "ungrounded_1", "text": "The Renaissance painter Raphael created numerous frescoes in the Vatican."},
        {"document_id": "dup_1", "text": "James Webb observes infrared radiation from distant cosmic galaxies.", "source_url": "https://stsci.edu/jwst"},
        {"document_id": "dup_2", "text": "James Webb observes infrared radiation from distant cosmic galaxies.", "source_url": "https://stsci.edu/jwst#details"},
    ]

    batch_res = detector.evaluate_batch(claim, items)
    assert batch_res.total_items == 4
    assert batch_res.evaluations[0].document_id == "clean_1"
    assert batch_res.evaluations[1].document_id == "ungrounded_1"
    assert batch_res.evaluations[2].document_id == "dup_1"
    assert batch_res.evaluations[3].document_id == "dup_2"

    assert "clean_1" in batch_res.clean_document_ids
    assert "ungrounded_1" in batch_res.ungrounded_document_ids
    assert "dup_2" in batch_res.polluted_document_ids  # Detected as EXACT_DUPLICATE of dup_1 URL


# =====================================================================
# 9. Configuration & Validation (§24)
# =====================================================================


def test_config_valid_and_invalid() -> None:
    """Test configuration range bounds and validation errors."""
    valid_cfg = EvidencePollutionSettings(ungrounded_threshold=0.20, weak_grounding_threshold=0.40)
    assert valid_cfg.ungrounded_threshold == 0.20

    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(ungrounded_threshold=-0.1)

    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(ungrounded_threshold=0.5, weak_grounding_threshold=0.3)

    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(minimum_meaningful_tokens=0)


# =====================================================================
# 10. Serialization & JSON Round-Trip (§14.1)
# =====================================================================


def test_json_round_trip() -> None:
    """Verify Pydantic model serialization and round-trip fidelity."""
    detector = EvidencePollutionDetector()
    claim = "Mitochondria produce ATP through cellular respiration."
    evidence = {"document_id": "bio_mito", "text": "Mitochondria synthesize ATP molecules via oxidative phosphorylation."}

    res = detector.evaluate(claim, evidence)
    json_str = res.model_dump_json()
    reconstituted = EvidencePollutionResult.model_validate_json(json_str)

    assert reconstituted.document_id == res.document_id
    assert reconstituted.pollution_score == res.pollution_score
    assert reconstituted.classification == res.classification
    assert reconstituted.grounding_score == res.grounding_score
    assert reconstituted.circularity_supported is False


# =====================================================================
# 11. Performance Targets (§22)
# =====================================================================


def test_performance_single_evaluation_under_five_ms() -> None:
    """Warm single evaluation executes well under 5.0 ms."""
    detector = EvidencePollutionDetector()
    claim = "Plate tectonics describes the large-scale motion of seven major plates."
    evidence = {"document_id": "geo_1", "text": "Plate tectonics models the dynamics of Earth's lithospheric plates."}

    # Warm-up
    for _ in range(20):
        detector.evaluate(claim, evidence)

    t0 = time.perf_counter()
    for _ in range(100):
        detector.evaluate(claim, evidence)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / 100) * 1000
    assert avg_ms < 5.0


def test_singleton_accessor() -> None:
    """Verify singleton lifecycle for get_evidence_pollution_detector."""
    det1 = get_evidence_pollution_detector()
    det2 = get_evidence_pollution_detector()
    assert det1 is det2
    reset_evidence_pollution_detector()
    det3 = get_evidence_pollution_detector()
    assert det3 is not det1


# =====================================================================
# 12. Optional Add-ons: Compatibility, Sensitivity & Snapshots (§30)
# =====================================================================


def test_compatibility_with_corpus_document_and_retrieval_result() -> None:
    """Ensure evaluate() works natively with real CorpusDocument and RetrievalResult objects."""
    detector = EvidencePollutionDetector()
    claim = "The Hubble Space Telescope was launched into low Earth orbit in 1990."

    corpus_doc = CorpusDocument(
        id="corpus_101",
        title="Hubble Space Mission",
        text="The Hubble Space Telescope was launched into low Earth orbit in 1990 by Space Shuttle Discovery.",
        source="NASA",
        domain="nasa.gov",
    )
    retrieval_res = RetrievalResult(
        document_id="retrieval_202",
        title="Hubble Space Mission",
        text="The Hubble Space Telescope was launched into low Earth orbit in 1990 by Space Shuttle Discovery.",
        source="NASA Press",
        domain="nasa.gov",
        score=0.98,
        retrieval_method="dense",
        rank=1,
    )

    r1 = detector.evaluate(claim, corpus_doc)
    r2 = detector.evaluate(claim, retrieval_res)

    assert r1.document_id == "corpus_101"
    assert r1.classification == PollutionClass.CLEAN
    assert r2.document_id == "retrieval_202"
    assert r2.classification == PollutionClass.CLEAN


def test_threshold_sensitivity_sweep() -> None:
    """Sweep ungrounded_threshold across {0.05, 0.10, 0.15, 0.20, 0.30} on a fixed pair."""
    claim = "Deep neural networks optimize loss functions via gradient descent algorithms."
    # Meaningful claim tokens: deep, neural, networks, optimize, loss, functions, gradient, descent, algorithms (9 tokens)
    # Evidence shares 1 token ("algorithms") -> 1/9 = 0.1111
    evidence = {"document_id": "math_1", "text": "Mathematical optimization algorithms solve constrained numerical equations."}

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = {}
    for th in thresholds:
        det = EvidencePollutionDetector(EvidencePollutionSettings(ungrounded_threshold=th, weak_grounding_threshold=0.40))
        res = det.evaluate(claim, evidence)
        results[th] = res.classification

    # At 0.05, 0.10: grounding 0.1111 >= th -> WEAK_POLLUTION_SIGNAL (since < 0.40)
    assert results[0.05] == PollutionClass.WEAK_POLLUTION_SIGNAL
    assert results[0.10] == PollutionClass.WEAK_POLLUTION_SIGNAL
    # At 0.15, 0.20, 0.30: grounding 0.1111 < th -> UNGROUNDED
    assert results[0.15] == PollutionClass.UNGROUNDED
    assert results[0.20] == PollutionClass.UNGROUNDED
    assert results[0.30] == PollutionClass.UNGROUNDED


def test_repeated_text_in_batch_detection() -> None:
    """Detect repeated identical text in a batch as EXACT_DUPLICATE."""
    detector = EvidencePollutionDetector()
    claim = "Photosynthesis requires chlorophyll to absorb light."
    items = [
        {"document_id": "doc_a", "text": "Chlorophyll absorbs photon wavelengths during photosynthesis."},
        {"document_id": "doc_b", "text": "Chlorophyll absorbs photon wavelengths during photosynthesis."},
    ]
    batch_res = detector.evaluate_batch(claim, items)
    assert batch_res.evaluations[0].classification == PollutionClass.CLEAN
    assert batch_res.evaluations[1].classification == PollutionClass.EXACT_DUPLICATE
    assert "doc_b" in batch_res.polluted_document_ids


def test_snapshots_stability() -> None:
    """Representative snapshot tests verifying byte-exact stability."""
    detector = EvidencePollutionDetector()
    snapshots = [
        (
            "Photosynthesis converts solar light energy into chemical glucose.",
            {"document_id": "s1", "text": "Photosynthesis converts solar light energy into chemical glucose molecules in plant chloroplasts."},
            PollutionClass.CLEAN,
            0.0,
        ),
        (
            "Photosynthesis converts solar light energy into chemical glucose.",
            {"document_id": "s2", "text": "Ancient Roman roads were built across Western Europe."},
            PollutionClass.UNGROUNDED,
            0.80,
        ),
        (
            "Photosynthesis converts solar light energy into chemical glucose.",
            {"document_id": "s3", "text": "Photosynthesis converts solar light energy into chemical glucose."},
            PollutionClass.SELF_REFERENTIAL,
            1.0,
        ),
    ]

    for claim, ev, expected_class, expected_score in snapshots:
        res = detector.evaluate(claim, ev)
        assert res.classification == expected_class
        assert res.pollution_score == expected_score
