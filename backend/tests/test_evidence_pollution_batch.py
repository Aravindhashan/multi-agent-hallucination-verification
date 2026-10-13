"""Dedicated Batch, 4-Layer Hierarchy, and Memory-Safe Chunking Test Suite for Submodule 8.4.

Covers:
1. 4-Layer Pollution Defense Hierarchy (§8.4):
   - Circularity / Self-Referential / Duplicate (P >= 0.90)
   - Low Grounding / Ungrounded (P >= 0.80)
   - Boilerplate / Scraped Template (P >= 0.60)
   - Clean (P < 0.20)
2. Gate classification function classify_pollution_gate().
3. Scraper boilerplate detection and non-destructive filtering.
4. Clean evidence retention target (>95% specificity).
5. Memory-safe chunked batching (batch_size=10) with order preservation.
6. Cross-chunk duplicate and circularity detection across boundaries.
7. Heterogeneous input type support (CorpusDocument, RetrievalResult, dict, custom).
8. Input validation errors on invalid batch_size.
9. Configuration constants centralization and [PROVISIONAL] docstring audits.
10. Mutation testing traps.
"""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from app.config import settings as app_settings
from app.services.corpus_schema import CorpusDocument
from app.services.evidence_pollution_config import (
    DEFAULT_BOILERPLATE_PATTERNS,
    DISCLAIMER_PHRASE,
    POLLUTION_BOILERPLATE_THRESHOLD,
    POLLUTION_CIRCULARITY_THRESHOLD,
    POLLUTION_CLEAN_RETENTION_TARGET,
    POLLUTION_CLEAN_THRESHOLD,
    POLLUTION_LOW_GROUNDING_THRESHOLD,
    EvidencePollutionConfigError,
    EvidencePollutionInputError,
    EvidencePollutionSettings,
)
from app.services.evidence_pollution_detector import (
    EvidencePollutionDetector,
    classify_pollution_gate,
    detect_boilerplate,
    get_evidence_pollution_detector,
    reset_evidence_pollution_detector,
)
from app.services.evidence_pollution_models import (
    BatchPollutionResult,
    EvidencePollutionResult,
    PollutionClass,
)
from app.services.retrieval_models import RetrievalResult


@pytest.fixture(autouse=True)
def cleanup_detector() -> None:
    """Reset singleton detector before and after each test."""
    reset_evidence_pollution_detector()
    yield
    reset_evidence_pollution_detector()


# =====================================================================
# 1. Four-Layer Hierarchy & Threshold Invariants (§8.4)
# =====================================================================


def test_four_layer_pollution_hierarchy_and_scores() -> None:
    """Verify distinct score separation and monotonic ranking across all 4 layers."""
    detector = EvidencePollutionDetector()
    claim = "The Hubble Space Telescope was deployed into low Earth orbit in 1990."

    # Layer 1: Circularity / Exact Duplicate (P >= 0.90 -> 1.00)
    ev_circ = {"document_id": "circ_1", "text": claim}
    r_circ = detector.evaluate(claim, ev_circ)
    assert r_circ.classification == PollutionClass.SELF_REFERENTIAL
    assert r_circ.pollution_score >= detector.settings.circularity_threshold
    assert r_circ.pollution_score == 1.00

    # Layer 2: Ungrounded / Low Grounding (P >= 0.80 -> 0.80)
    ev_ungrounded = {
        "document_id": "ungrounded_1",
        "text": "Photosynthesis produces adenosine triphosphate in plant chloroplasts.",
    }
    r_ungrounded = detector.evaluate(claim, ev_ungrounded)
    assert r_ungrounded.classification == PollutionClass.UNGROUNDED
    assert r_ungrounded.pollution_score >= detector.settings.low_grounding_threshold
    assert r_ungrounded.pollution_score == 0.80

    # Layer 3: Boilerplate (P >= 0.60 -> 0.60)
    ev_boilerplate = {
        "document_id": "bp_1",
        "text": "Please accept all cookies to continue. All rights reserved. Terms of service apply.",
    }
    r_bp = detector.evaluate(claim, ev_boilerplate)
    assert r_bp.classification == PollutionClass.BOILERPLATE
    assert r_bp.pollution_score >= detector.settings.boilerplate_threshold
    assert r_bp.pollution_score == 0.60
    assert r_bp.is_boilerplate is True

    # Layer 4: Clean (P < 0.20 -> 0.00)
    ev_clean = {
        "document_id": "clean_1",
        "text": "The Hubble Space Telescope was deployed into low Earth orbit in April 1990 by Space Shuttle Discovery.",
    }
    r_clean = detector.evaluate(claim, ev_clean)
    assert r_clean.classification == PollutionClass.CLEAN
    assert r_clean.pollution_score < detector.settings.clean_threshold
    assert r_clean.pollution_score == 0.00

    # Strict monotonicity
    assert (
        r_clean.pollution_score
        < r_bp.pollution_score
        < r_ungrounded.pollution_score
        <= r_circ.pollution_score
    )


@pytest.mark.parametrize(
    "score, expected_gate",
    [
        (1.00, "CIRCULARITY"),
        (0.95, "CIRCULARITY"),
        (0.90, "CIRCULARITY"),
        (0.89, "LOW_GROUNDING"),
        (0.80, "LOW_GROUNDING"),
        (0.79, "BOILERPLATE"),
        (0.60, "BOILERPLATE"),
        (0.59, "WEAK_SIGNAL"),
        (0.30, "WEAK_SIGNAL"),
        (0.20, "WEAK_SIGNAL"),
        (0.19, "CLEAN"),
        (0.00, "CLEAN"),
    ],
)
def test_classify_pollution_gate(score: float, expected_gate: str) -> None:
    """Verify classify_pollution_gate maps continuous scores to exact discrete gate categories."""
    assert classify_pollution_gate(score) == expected_gate


# =====================================================================
# 2. Scraper Boilerplate Detection (§8.4)
# =====================================================================


def test_detect_boilerplate_patterns() -> None:
    """Verify detection of various web template and scraper boilerplate headers/footers."""
    detector = EvidencePollutionDetector()
    claim = "Quantum computing uses qubits to execute superposition algorithms."

    boilerplate_samples = [
        ("bp_cookie", "Accept all cookies and manage cookie preferences to continue reading."),
        ("bp_terms", "All rights reserved. Terms of use and privacy policy apply to all users."),
        ("bp_sub", "Subscribe to continue reading this article and sign up for our newsletter today."),
        ("bp_social", "Share on twitter or follow us on facebook for more updates."),
        ("bp_ad", "Advertisement: Sponsored content. Click here to read full article."),
        ("bp_comment", "Leave a comment below. Related articles and stories will appear here."),
    ]

    for doc_id, text in boilerplate_samples:
        is_bp, matches = detect_boilerplate(text)
        assert is_bp is True, f"Failed to detect boilerplate in '{text}'"
        assert len(matches) >= 1

        res = detector.evaluate(claim, {"document_id": doc_id, "text": text})
        assert res.is_boilerplate is True
        assert res.classification in (PollutionClass.BOILERPLATE, PollutionClass.UNGROUNDED)
        assert res.pollution_score >= 0.60


def test_clean_evidence_retention_target_above_95_percent() -> None:
    """Non-destructive retrieval pre-filtering: clean evidence achieves >95% retention specificity."""
    detector = EvidencePollutionDetector()

    clean_test_cases = [
        (
            "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.",
            "The James Webb Space Telescope orbits the second Sun-Earth Lagrange point (L2), approximately 1.5 million kilometers from Earth.",
        ),
        (
            "Scottish biologist Alexander Fleming discovered penicillin antibiotic in London.",
            "Scottish bacteriologist Alexander Fleming discovered penicillin at St Mary's Hospital in London in September 1928.",
        ),
        (
            "The Federal Reserve lowered the federal funds rate by 50 basis points.",
            "The Federal Reserve voted to lower the target range for the federal funds rate by 50 basis points at its September meeting.",
        ),
        (
            "The French Revolution began in 1789 with the storming of the Bastille.",
            "The French Revolution began in July 1789 when Parisian revolutionaries seized the Bastille medieval fortress.",
        ),
        (
            "CRISPR-Cas9 acts as RNA-guided molecular scissors to edit DNA sequences.",
            "The CRISPR-Cas9 endonuclease functions as molecular scissors guided by single-guide RNA to introduce double-strand DNA breaks.",
        ),
        (
            "Python 3.11 introduced substantial performance optimizations.",
            "Python 3.11 included major runtime performance improvements from the Faster CPython initiative.",
        ),
        (
            "Mount Everest is the highest mountain peak above sea level.",
            "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
        ),
        (
            "King John granted the Magna Carta historic charter at Runnymede.",
            "King John of England granted the Magna Carta at Runnymede near Windsor in June 1215 under pressure from rebel barons.",
        ),
        (
            "The speed of light in vacuum is approximately 299792 kilometers per second.",
            "In vacuum, the exact speed of light c is defined as 299,792,458 meters per second.",
        ),
        (
            "Photosynthesis converts carbon dioxide and water into oxygen and glucose.",
            "Photosynthesis in green plants utilizes sunlight to convert atmospheric carbon dioxide and water into glucose and oxygen.",
        ),
    ]

    clean_count = 0
    total = len(clean_test_cases)

    for i, (claim, text) in enumerate(clean_test_cases):
        res = detector.evaluate(claim, {"document_id": f"clean_{i}", "text": text})
        if res.classification == PollutionClass.CLEAN:
            clean_count += 1

    retention_rate = clean_count / float(total)
    assert retention_rate >= detector.settings.clean_retention_target
    assert retention_rate >= 0.95


# =====================================================================
# 3. Memory-Safe Chunked Batch Processing & Order Preservation (§8.4)
# =====================================================================


def test_evaluate_batch_preserves_order_and_counts() -> None:
    """Batch evaluation evaluates 15 items maintaining 100% exact sequential order and counts."""
    detector = EvidencePollutionDetector()
    claim = "The Hubble Space Telescope was launched in 1990."

    items = [
        {"document_id": f"doc_{i:02d}", "text": f"Passage {i} concerning Hubble telescope launch in 1990."}
        for i in range(15)
    ]

    batch_res = detector.evaluate_batch(claim, items, batch_size=10)
    assert batch_res.total_items == 15
    assert len(batch_res.evaluations) == 15

    for i, eval_item in enumerate(batch_res.evaluations):
        assert eval_item.document_id == f"doc_{i:02d}"


def test_evaluate_batch_chunking_large_payload_parity() -> None:
    """Large batch (N=25) scored across chunk sizes 1, 5, 10, 25, 50 produces identical output."""
    detector = EvidencePollutionDetector()
    claim = "Photosynthesis produces oxygen and glucose from carbon dioxide and water."

    items: list[dict[str, str]] = []
    for i in range(25):
        if i % 5 == 0:
            # Ungrounded
            items.append({"document_id": f"item_{i:02d}", "text": f"Historical treatise on Roman coinage {i}."})
        elif i % 5 == 1:
            # Boilerplate
            items.append({"document_id": f"item_{i:02d}", "text": f"Accept all cookies to read item {i}. Terms of service apply."})
        elif i % 5 == 2:
            # Clean
            items.append({"document_id": f"item_{i:02d}", "text": f"Photosynthesis generates oxygen and glucose {i}."})
        elif i % 5 == 3:
            # Near duplicate
            items.append({"document_id": f"item_{i:02d}", "text": f"Photosynthesis generates oxygen and glucose {i} in plants."})
        else:
            # Weak signal
            items.append({"document_id": f"item_{i:02d}", "text": f"Green water oxygen {i}."})

    baseline = detector.evaluate_batch(claim, items, batch_size=50)

    for chunk_size in [1, 5, 10, 25]:
        chunked = detector.evaluate_batch(claim, items, batch_size=chunk_size)
        assert chunked.total_items == baseline.total_items
        assert chunked.polluted_document_ids == baseline.polluted_document_ids
        assert chunked.ungrounded_document_ids == baseline.ungrounded_document_ids
        assert chunked.boilerplate_document_ids == baseline.boilerplate_document_ids
        assert chunked.clean_document_ids == baseline.clean_document_ids

        for b_item, c_item in zip(baseline.evaluations, chunked.evaluations):
            assert b_item.document_id == c_item.document_id
            assert b_item.classification == c_item.classification
            assert b_item.pollution_score == c_item.pollution_score
            assert b_item.grounding_score == c_item.grounding_score
            assert b_item.is_boilerplate == c_item.is_boilerplate


def test_cross_chunk_duplicate_and_circularity_detection() -> None:
    """In-batch duplicates appearing across chunk boundaries (doc 0 vs doc 12) are detected."""
    detector = EvidencePollutionDetector()
    claim = "James Webb Space Telescope observes the early universe."

    # Create 15 items: item 0 and item 12 have identical text
    items = [
        {"document_id": f"d_{i}", "text": f"Distinct scientific observation {i} of early universe galaxies."}
        for i in range(15)
    ]
    # Inject exact text and URL at index 0 and index 12 (across chunk boundary of batch_size=10)
    duplicate_text = "James Webb Space Telescope detected the earliest known galaxy candidate."
    items[0]["text"] = duplicate_text
    items[0]["source_url"] = "https://stsci.edu/jwst/earliest-galaxy"

    items[12]["text"] = duplicate_text
    items[12]["source_url"] = "https://stsci.edu/jwst/earliest-galaxy#findings"

    batch_res = detector.evaluate_batch(claim, items, batch_size=10)
    assert batch_res.evaluations[0].classification != PollutionClass.EXACT_DUPLICATE
    assert batch_res.evaluations[12].classification == PollutionClass.EXACT_DUPLICATE
    assert "d_12" in batch_res.polluted_document_ids


def test_evaluate_batch_heterogeneous_input_types() -> None:
    """Evaluate batch containing CorpusDocument, RetrievalResult, dict, and custom object."""
    detector = EvidencePollutionDetector()
    claim = "The Hubble Space Telescope was launched in 1990."

    @dataclass
    class CustomDoc:
        id: str
        text: str
        source_url: str | None = None

    corpus_doc = CorpusDocument(
        id="c1",
        title="Hubble",
        text="The Hubble Space Telescope was launched in 1990 by NASA.",
        source="NASA",
    )
    retrieval_doc = RetrievalResult(
        document_id="r1",
        title="Hubble",
        text="The Hubble Space Telescope was launched in 1990 by NASA.",
        source="NASA",
        score=0.95,
        retrieval_method="dense",
        rank=1,
    )
    dict_doc = {"document_id": "d1", "text": "The Hubble Space Telescope was launched in 1990 by NASA."}
    custom_doc = CustomDoc(id="cust1", text="The Hubble Space Telescope was launched in 1990 by NASA.")

    items = [corpus_doc, retrieval_doc, dict_doc, custom_doc]
    res = detector.evaluate_batch(claim, items, batch_size=2)
    assert res.total_items == 4
    assert res.evaluations[0].document_id == "c1"
    assert res.evaluations[1].document_id == "r1"
    assert res.evaluations[2].document_id == "d1"
    assert res.evaluations[3].document_id == "cust1"


# =====================================================================
# 4. Input Validation & Error Handling
# =====================================================================


@pytest.mark.parametrize(
    "invalid_batch_size",
    [0, -1, -10, True, False, "10", 3.14, None],
)
def test_batch_size_validation_errors(invalid_batch_size: any) -> None:
    """Invalid batch_size values raise EvidencePollutionInputError."""
    detector = EvidencePollutionDetector()
    claim = "Valid claim text."
    items = [{"document_id": "d1", "text": "Valid evidence text."}]

    with pytest.raises(EvidencePollutionInputError):
        detector.evaluate_batch(claim, items, batch_size=invalid_batch_size)


# =====================================================================
# 5. Configuration Centralization & Provisional Labels (§8.4)
# =====================================================================


def test_centralized_constants_and_provisional_audit() -> None:
    """Verify centralized constants in app.config and evidence_pollution_config match spec."""
    assert app_settings.POLLUTION_CIRCULARITY_THRESHOLD == 0.90
    assert app_settings.POLLUTION_LOW_GROUNDING_THRESHOLD == 0.80
    assert app_settings.POLLUTION_BOILERPLATE_THRESHOLD == 0.60
    assert app_settings.POLLUTION_CLEAN_THRESHOLD == 0.20
    assert app_settings.POLLUTION_CLEAN_RETENTION_TARGET == 0.95

    assert POLLUTION_CIRCULARITY_THRESHOLD == 0.90
    assert POLLUTION_LOW_GROUNDING_THRESHOLD == 0.80
    assert POLLUTION_BOILERPLATE_THRESHOLD == 0.60
    assert POLLUTION_CLEAN_THRESHOLD == 0.20
    assert POLLUTION_CLEAN_RETENTION_TARGET == 0.95


def test_settings_monotonicity_and_bounds_validation() -> None:
    """EvidencePollutionSettings validates bounds and monotonicity."""
    # Valid default
    settings = EvidencePollutionSettings()
    assert settings.clean_threshold <= settings.boilerplate_threshold
    assert settings.boilerplate_threshold <= settings.low_grounding_threshold
    assert settings.low_grounding_threshold <= settings.circularity_threshold

    # Inverted clean > boilerplate
    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(clean_threshold=0.70, boilerplate_threshold=0.60)

    # Inverted boilerplate > low_grounding
    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(boilerplate_threshold=0.85, low_grounding_threshold=0.80)

    # Out of range (> 1.0)
    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(circularity_threshold=1.5)

    # Out of range (< 0.0)
    with pytest.raises(EvidencePollutionConfigError):
        EvidencePollutionSettings(clean_threshold=-0.1)


# =====================================================================
# 6. Mutation Testing Pass
# =====================================================================


def test_mutation_traps_boilerplate_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: if boilerplate detection is bypassed, boilerplate items falsely pass as CLEAN."""
    detector = EvidencePollutionDetector()
    claim = "General factual statement."
    boilerplate_text = (
        "Please accept all cookies to continue. All rights reserved. "
        "Terms of service and privacy policy apply to all users."
    )

    # Normal detector correctly classifies as BOILERPLATE
    normal_res = detector.evaluate(claim, {"document_id": "b1", "text": boilerplate_text})
    assert normal_res.classification == PollutionClass.BOILERPLATE
    assert normal_res.pollution_score == 0.60

    # Mutate detector settings with empty patterns
    mutated_settings = EvidencePollutionSettings(boilerplate_patterns=())
    mutated_detector = EvidencePollutionDetector(settings=mutated_settings)
    mutated_res = mutated_detector.evaluate(claim, {"document_id": "b1", "text": boilerplate_text})

    # Trapped: without patterns, it fails to classify as BOILERPLATE
    assert mutated_res.classification != PollutionClass.BOILERPLATE
