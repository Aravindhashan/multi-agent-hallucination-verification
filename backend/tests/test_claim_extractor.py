"""Unit tests for Atomic Claim Extraction (NLP & LLM pipelines)."""

import json
import pytest
from pydantic import ValidationError

from app.schemas import ClaimExtractionResponse
from app.services.claim_extractor import BaseClaimExtractor
from app.services.claim_models import AtomicClaim, ClaimExtractionResult
from app.services.claim_extraction_service import (
    extract_atomic_claims,
    extract_claims,
    get_claim_extractor,
)
from app.services.llm_claim_extractor import LLMClaimExtractor
from app.services.llm_client import BaseLLMClient, LLMResponseError
from app.services.mock_llm_client import MockLLMClient
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.simple_claim_extractor import SimpleClaimExtractor
from uuid import uuid4


# ---------------------------------------------------------------------------
# 1. Data Model Tests
# ---------------------------------------------------------------------------

def test_valid_atomic_claim_full_fields():
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Apollo 11 landed on the Moon in 1969.",
        subject="Apollo 11",
        predicate="landed on",
        object="the Moon",
        temporal_anchor="1969",
        numerical_value="1969",
        spatial_anchor="on the Moon",
        qualifiers=["in 1969"],
    )
    assert claim.claim_id == "c1"
    assert claim.claim_text == "Apollo 11 landed on the Moon in 1969."
    assert claim.subject == "Apollo 11"
    assert claim.predicate == "landed on"
    assert claim.object == "the Moon"
    assert claim.temporal_anchor == "1969"
    assert claim.spatial_anchor == "on the Moon"
    assert claim.qualifiers == ["in 1969"]


def test_atomic_claim_optional_fields_default():
    claim = AtomicClaim(
        claim_id="c2",
        claim_text="Paris is the capital of France.",
    )
    assert claim.claim_id == "c2"
    assert claim.claim_text == "Paris is the capital of France."
    assert claim.subject is None
    assert claim.predicate is None
    assert claim.object is None
    assert claim.temporal_anchor is None
    assert claim.numerical_value is None
    assert claim.spatial_anchor is None
    assert claim.qualifiers == []


def test_atomic_claim_empty_claim_id_fails():
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="   ", claim_text="Valid claim text.")


def test_atomic_claim_empty_claim_text_fails():
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="c1", claim_text="   ")


def test_claim_extraction_result_structure():
    result = ClaimExtractionResult(
        claims=[
            AtomicClaim(claim_id="c1", claim_text="Paris is the capital of France.")
        ],
        extraction_method="nlp_rule_based",
        raw_response_text="Paris is the capital of France.",
    )
    assert len(result.claims) == 1
    assert result.extraction_method == "nlp_rule_based"
    assert result.raw_response_text == "Paris is the capital of France."
    assert result.extracted_at is not None


# ---------------------------------------------------------------------------
# 2. NLP Pipeline Component Tests
# ---------------------------------------------------------------------------

def test_nlp_sentence_segmentation_with_abbreviations():
    extractor = NLPClaimExtractor()
    text = "Dr. Smith visited the U.S. in Jan. 2020. He met Prof. Jones at CERN."
    sentences = extractor._segment_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith visited the U.S. in Jan. 2020." in sentences[0]
    assert "He met Prof. Jones at CERN." in sentences[1]


def test_nlp_sentence_segmentation_with_decimals():
    extractor = NLPClaimExtractor()
    text = "The value of pi is approximately 3.14159. The speed of light is 299,792,458 meters per second."
    sentences = extractor._segment_sentences(text)
    assert len(sentences) == 2
    assert "3.14159" in sentences[0]
    assert "299,792,458" in sentences[1]


def test_nlp_clause_splitting_compound_conjunction():
    extractor = NLPClaimExtractor()
    sentence = "Apollo 11 landed on the Moon in 1969 and Neil Armstrong became the first person to walk on the Moon."
    clauses = extractor._split_clauses(sentence)
    assert len(clauses) == 2
    assert "Apollo 11 landed on the Moon in 1969" in clauses[0]
    assert "Neil Armstrong became the first person to walk on the Moon" in clauses[1]


def test_nlp_clause_splitting_compound_predicate_inherits_subject():
    extractor = NLPClaimExtractor()
    sentence = "Python was created by Guido van Rossum and was first released in 1991."
    clauses = extractor._split_clauses(sentence)
    assert len(clauses) == 2
    assert "Python was created by Guido van Rossum" in clauses[0]
    assert "Python" in clauses[1]
    assert "first released in 1991" in clauses[1]


def test_nlp_coreference_resolution_pronoun_propagation():
    extractor = NLPClaimExtractor()
    salient = {"subject": "Guido van Rossum", "object": "Python"}
    clause = "He first released it publicly in 1991"
    resolved = extractor._resolve_coreference(clause, salient)
    assert "Guido van Rossum" in resolved
    assert "Python" in resolved
    assert "He" not in resolved


def test_nlp_temporal_anchor_extraction():
    extractor = NLPClaimExtractor()
    analysis = extractor._analyze_clause("The Apollo 11 mission occurred on July 20, 1969.")
    assert analysis["temporal_anchor"] is not None
    assert "1969" in analysis["temporal_anchor"]


def test_nlp_numerical_value_and_unit_extraction():
    extractor = NLPClaimExtractor()
    analysis = extractor._analyze_clause("The speed of light in vacuum is approximately 299,792,458 meters per second.")
    assert analysis["numerical_value"] is not None
    assert "299,792,458" in analysis["numerical_value"]
    assert "meters per second" in analysis["numerical_value"]


def test_nlp_spatial_anchor_extraction():
    extractor = NLPClaimExtractor()
    analysis = extractor._analyze_clause("Apollo 11 touched down on the Moon.")
    assert analysis["spatial_anchor"] == "on the Moon"


def test_nlp_qualifier_extraction():
    extractor = NLPClaimExtractor()
    analysis = extractor._analyze_clause("The speed of light in vacuum is approximately 299,792,458 meters per second.")
    assert "in vacuum" in analysis["qualifiers"]
    assert "approximately" in analysis["qualifiers"]


def test_nlp_subject_predicate_object_decomposition():
    extractor = NLPClaimExtractor()
    analysis = extractor._analyze_clause("Paris is the capital of France.")
    assert analysis["subject"] == "Paris"
    assert analysis["predicate"] == "is the capital of" or analysis["predicate"] == "is"
    assert "France" in analysis["object"]


# ---------------------------------------------------------------------------
# 3. Required Factual Sanity Cases (Section 18)
# ---------------------------------------------------------------------------

def test_sanity_case_1_paris_france():
    extractor = NLPClaimExtractor()
    result = extractor.extract_atomic_claims("Paris is the capital of France.")
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.claim_id == "c1"
    assert "Paris is the capital of France" in claim.claim_text
    assert claim.subject == "Paris"


def test_sanity_case_2_python_rossum():
    extractor = NLPClaimExtractor()
    text = "Python was created by Guido van Rossum. Python was first released in 1991."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 2
    assert result.claims[0].claim_id == "c1"
    assert result.claims[1].claim_id == "c2"
    assert "Guido van Rossum" in result.claims[0].claim_text
    assert "1991" in result.claims[1].claim_text


def test_sanity_case_3_speed_of_light_preservation():
    extractor = NLPClaimExtractor()
    text = "The speed of light in vacuum is approximately 299,792,458 meters per second."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert "299,792,458" in claim.claim_text
    assert "meters per second" in claim.claim_text
    assert "in vacuum" in claim.claim_text
    assert "approximately" in claim.claim_text
    assert claim.numerical_value is not None
    assert "299,792,458" in claim.numerical_value
    assert "meters per second" in claim.numerical_value
    assert "in vacuum" in claim.qualifiers
    assert "approximately" in claim.qualifiers


def test_sanity_case_4_apollo_11_compound():
    extractor = NLPClaimExtractor()
    text = "Apollo 11 landed on the Moon in 1969 and Neil Armstrong became the first person to walk on the Moon."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 2
    assert result.claims[0].claim_id == "c1"
    assert result.claims[1].claim_id == "c2"
    assert "Apollo 11" in result.claims[0].claim_text
    assert "1969" in result.claims[0].claim_text
    assert "Neil Armstrong" in result.claims[1].claim_text
    assert "first person to walk on the Moon" in result.claims[1].claim_text


def test_sanity_case_5_coreference_resolution():
    extractor = NLPClaimExtractor()
    text = "Guido van Rossum created Python. He first released it publicly in 1991."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 2
    assert result.claims[0].claim_id == "c1"
    assert result.claims[1].claim_id == "c2"
    assert "Guido van Rossum created Python" in result.claims[0].claim_text
    # Coreference should replace He with Guido van Rossum and it with Python
    claim2 = result.claims[1].claim_text
    assert "Guido van Rossum" in claim2
    assert "Python" in claim2
    assert "1991" in claim2


def test_appositive_clause_splitting_three_claims():
    """Verify that participial appositives are decomposed into 3 atomic claims (Rank 4)."""
    extractor = NLPClaimExtractor()
    text = "The Eiffel Tower, built in 1889 by Gustave Eiffel, is 330 meters tall."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 3
    claims_text = [c.claim_text for c in result.claims]
    assert any("330 meters tall" in ct for ct in claims_text)
    assert any("1889" in ct for ct in claims_text)
    assert any("Gustave Eiffel" in ct for ct in claims_text)


def test_bare_year_disambiguation_rank_10():
    """Verify bare 4-digit years are not conflated with numerical quantities unless qualified (Rank 10)."""
    extractor = NLPClaimExtractor()
    text = "Apollo 11 landed on the Moon in 1969."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.temporal_anchor is not None
    assert "1969" in claim.temporal_anchor
    assert claim.numerical_value is None  # Must NOT be conflated as numerical quantity

    # Conversely, a real number alongside a year should be properly extracted
    text2 = "Apollo 11 landed 2 astronauts on the Moon in 1969."
    result2 = extractor.extract_atomic_claims(text2)
    assert len(result2.claims) == 1
    claim2 = result2.claims[0]
    assert claim2.temporal_anchor is not None
    assert "1969" in claim2.temporal_anchor
    assert claim2.numerical_value == "2"


def test_plural_anchor_fields_rank_6():
    """Verify multiple temporal, numerical, and spatial anchors are preserved in list fields (Rank 6)."""
    extractor = NLPClaimExtractor()
    text = "The Wright brothers flew the Wright Flyer in 1903 at Kitty Hawk and again in 1904 in Dayton."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) >= 1
    all_temporal = []
    all_spatial = []
    for c in result.claims:
        all_temporal.extend(c.temporal_anchors)
        all_spatial.extend(c.spatial_anchors)
    assert "1903" in all_temporal
    assert any("Kitty Hawk" in s for s in all_spatial)


def test_generalized_spatial_anchors_rank_8():
    """Verify spatial prepositions recognize non-whitelisted geographical and astronomical locations (Rank 8)."""
    extractor = NLPClaimExtractor()
    analysis1 = extractor._analyze_clause("The flight occurred at Kitty Hawk.")
    assert analysis1["spatial_anchor"] == "at Kitty Hawk"

    analysis2 = extractor._analyze_clause("The telescope operates at Sun-Earth Lagrange Point 2.")
    assert analysis2["spatial_anchor"] is not None
    assert "Lagrange Point 2" in analysis2["spatial_anchor"]

    analysis3 = extractor._analyze_clause("He worked at CERN in Geneva.")
    assert len(analysis3["spatial_anchors"]) >= 1


# ---------------------------------------------------------------------------
# 4. Normalization, Deduplication, & Error Handling Tests
# ---------------------------------------------------------------------------

def test_duplicate_claims_removed():
    extractor = NLPClaimExtractor()
    text = "Paris is the capital of France. Paris is the capital of France."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "c1"


def test_conversational_prefix_stripping():
    extractor = NLPClaimExtractor()
    text = "Sure! Here is what I found: Paris is the capital of France."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 1
    assert "Sure" not in result.claims[0].claim_text
    assert "Here is what I found" not in result.claims[0].claim_text
    assert "Paris is the capital of France" in result.claims[0].claim_text


def test_empty_response_rejected():
    extractor = NLPClaimExtractor()
    with pytest.raises(ValueError, match="at least one non-whitespace character"):
        extractor.extract_atomic_claims("")


def test_whitespace_only_response_rejected():
    extractor = NLPClaimExtractor()
    with pytest.raises(ValueError, match="at least one non-whitespace character"):
        extractor.extract_atomic_claims("   \n\t  ")


def test_invalid_type_rejected():
    extractor = NLPClaimExtractor()
    with pytest.raises(ValueError, match="Expected response_text to be a string"):
        extractor.extract_atomic_claims(12345)  # type: ignore


# ---------------------------------------------------------------------------
# 5. LLM Claim Extractor Tests (with Mock LLM Client)
# ---------------------------------------------------------------------------

class FakeLLMClient(BaseLLMClient):
    """Fake LLM client returning predictable JSON responses for testing."""

    def __init__(self, response_json: dict):
        self.response_json = response_json

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    def generate(self, prompt: str) -> str:
        return json.dumps(self.response_json)


def test_llm_claim_extractor_successful_json():
    fake_client = FakeLLMClient({
        "claims": [
            "Paris is the capital of France.",
            "Python was first released in 1991."
        ]
    })
    extractor = LLMClaimExtractor(llm_client=fake_client)
    result = extractor.extract_atomic_claims("Sample response text")

    assert len(result.claims) == 2
    assert result.claims[0].claim_id == "c1"
    assert result.claims[0].claim_text == "Paris is the capital of France."
    assert result.claims[1].claim_id == "c2"
    assert result.claims[1].claim_text == "Python was first released in 1991."
    assert result.extraction_method == "llm"


def test_llm_claim_extractor_with_markdown_codefences():
    class CodeFenceLLMClient(BaseLLMClient):
        @property
        def model_name(self) -> str:
            return "codefence-model"

        def generate(self, prompt: str) -> str:
            return '```json\n{"claims": ["Apollo 11 landed on the Moon in 1969."]}\n```'

    extractor = LLMClaimExtractor(llm_client=CodeFenceLLMClient())
    result = extractor.extract_atomic_claims("Apollo response")
    assert len(result.claims) == 1
    assert result.claims[0].claim_text == "Apollo 11 landed on the Moon in 1969."


def test_llm_claim_extractor_empty_response_raises():
    class EmptyLLMClient(BaseLLMClient):
        @property
        def model_name(self) -> str:
            return "empty-model"

        def generate(self, prompt: str) -> str:
            return ""

    extractor = LLMClaimExtractor(llm_client=EmptyLLMClient())
    with pytest.raises(LLMResponseError, match="empty response"):
        extractor.extract_atomic_claims("Some text")


def test_llm_claim_extractor_invalid_json_raises():
    class MalformedLLMClient(BaseLLMClient):
        @property
        def model_name(self) -> str:
            return "malformed-model"

        def generate(self, prompt: str) -> str:
            return "This is not JSON at all."

    extractor = LLMClaimExtractor(llm_client=MalformedLLMClient())
    with pytest.raises(LLMResponseError, match="No valid JSON object found"):
        extractor.extract_atomic_claims("Some text")


# ---------------------------------------------------------------------------
# 6. Service Layer & Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_service_extract_atomic_claims_nlp():
    result = extract_atomic_claims(
        response_text="Paris is the capital of France.",
        method="nlp",
    )
    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "c1"
    assert result.claims[0].claim_text == "Paris is the capital of France."


def test_service_extract_atomic_claims_llm_strategy():
    fake_client = FakeLLMClient({
        "claims": ["Jupiter is the largest planet in the Solar System."]
    })
    result = extract_atomic_claims(
        response_text="Jupiter is the largest planet.",
        method="llm",
        llm_client=fake_client,
    )
    assert len(result.claims) == 1
    assert result.claims[0].claim_text == "Jupiter is the largest planet in the Solar System."
    assert result.extraction_method == "llm"


def test_service_extract_claims_legacy_api_format():
    qid = uuid4()
    response = extract_claims(
        query_id=qid,
        response_text="Paris is the capital of France. Python was created by Guido van Rossum.",
    )
    assert isinstance(response, ClaimExtractionResponse)
    assert response.query_id == qid
    assert len(response.claims) == 2
    assert response.claims[0].order == 0
    assert response.claims[1].order == 1
    assert "Paris" in response.claims[0].text
    assert "Python" in response.claims[1].text


def test_get_claim_extractor_factory():
    nlp_extractor = get_claim_extractor("nlp")
    assert isinstance(nlp_extractor, NLPClaimExtractor)

    llm_extractor = get_claim_extractor("llm", llm_client=MockLLMClient(model_name="mock-model"))
    assert isinstance(llm_extractor, LLMClaimExtractor)


def test_spatial_anchor_adversarial_precision():
    """Verify precision hardening: non-spatial entities (orgs, days, documents, events) are rejected."""
    extractor = NLPClaimExtractor()

    # Adversarial negative cases: corporate, financial, academic, temporal, document, and event entities
    negatives = [
        # Corporate & financial entities
        "Revenue increased in Microsoft's report.",
        "Analysts met at Goldman Sachs on Tuesday's call.",
        "Investors gathered at Morgan Stanley.",
        "The drug was produced by Pfizer Pharmaceuticals.",
        "The system was announced at Google Technologies.",
        "The survey was conducted at Barclays Group.",
        # Academic & healthcare institutions
        "Researchers at Stanford University published the findings.",
        "The patient was admitted at Mayo Clinic.",
        # Days of week, calendar periods, quarters, fiscal years
        "The executive spoke on Monday's briefing.",
        "Profits increased in Q3 earnings.",
        "Target milestones are detailed in FY2024 guidance.",
        "The event was announced in yesterday's broadcast.",
        # Document sections, chapters, tables, and figures
        "The theorem is proven in Chapter 3.",
        "Data is tabulated in Table 2.",
        "Illustrations are presented in Figure 4.",
        "Guidelines are outlined in Section 5.",
        "References are listed in Appendix B.",
        # Historical events & conferences
        "Major agreements were discussed in World War II conferences.",
        "Emergency policies were implemented in COVID-19 pandemic.",
        "Athletes competed at the Olympics.",
    ]
    for text in negatives:
        result = extractor.extract_atomic_claims(text)
        for claim in result.claims:
            assert len(claim.spatial_anchors) == 0, (
                f"False positive spatial anchor '{claim.spatial_anchors}' found in '{text}'"
            )
            assert claim.spatial_anchor is None

    # Positive cases: genuine cities, countries, continents, astronomical bodies, and landmarks
    positives = [
        ("The rover landed on Mars.", "on Mars"),
        ("The treaty was signed in Paris.", "in Paris"),
        ("The headquarters is in Canada.", "in Canada"),
        ("The assembly plant is in Mumbai.", "in Mumbai"),
        ("The conference was hosted in Berlin.", "in Berlin"),
        ("Climbers reached the summit on Mount Everest.", "on Mount Everest"),
        ("The ship traveled across Europe.", "across Europe"),
        ("The telescope operates in low Earth orbit.", "in low Earth orbit"),
        ("Astronauts walked on the Moon.", "on the Moon"),
    ]
    for text, expected in positives:
        result = extractor.extract_atomic_claims(text)
        assert len(result.claims) >= 1
        claim = result.claims[0]
        assert any(expected.lower() in sa.lower() for sa in claim.spatial_anchors), (
            f"Expected spatial anchor '{expected}' not found in '{claim.spatial_anchors}' for '{text}'"
        )


def test_recursive_appositive_multi_fact_splitting():
    """Verify appositive splitting recurses and decomposes multi-fact embedded clauses into discrete atomic claims."""
    extractor = NLPClaimExtractor()

    # Case 1: Eiffel Tower (date + builder attribution + height) -> exactly 3 atomic claims
    text1 = "The Eiffel Tower, built in 1889 by Gustave Eiffel, is 330 meters tall."
    res1 = extractor.extract_atomic_claims(text1)
    claim_texts1 = [c.claim_text.lower() for c in res1.claims]
    assert len(res1.claims) == 3, f"Expected 3 atomic claims, got {len(res1.claims)}: {claim_texts1}"
    assert any("330 meters" in t for t in claim_texts1)
    assert any("gustave eiffel" in t for t in claim_texts1)
    assert any("1889" in t for t in claim_texts1)

    # Case 2: 3-embedded facts (date + discoverer + location + impact) -> 4 atomic claims
    text2 = "The telescope, discovered in 1920 by Edwin Hubble at Mount Wilson Observatory, changed modern astronomy."
    res2 = extractor.extract_atomic_claims(text2)
    claim_texts2 = [c.claim_text.lower() for c in res2.claims]
    assert len(res2.claims) == 4, f"Expected 4 atomic claims, got {len(res2.claims)}: {claim_texts2}"
    assert any("changed modern astronomy" in t for t in claim_texts2)
    assert any("edwin hubble" in t for t in claim_texts2)
    assert any("1920" in t for t in claim_texts2)
    assert any("mount wilson observatory" in t for t in claim_texts2)

    # Case 3: Coordinate participles
    text3 = "The monument, designed by Maurice Koechlin and built in 1889, stands in Paris."
    res3 = extractor.extract_atomic_claims(text3)
    claim_texts3 = [c.claim_text.lower() for c in res3.claims]
    assert len(res3.claims) == 3, f"Expected 3 atomic claims, got {len(res3.claims)}: {claim_texts3}"
    assert any("maurice koechlin" in t for t in claim_texts3)
    assert any("1889" in t for t in claim_texts3)
    assert any("stands in paris" in t for t in claim_texts3)

    # Case 4: Dash-delimited multi-fact appositive
    text4 = "The James Webb Space Telescope — built by Northrop Grumman and launched in 2021 — observes distant galaxies."
    res4 = extractor.extract_atomic_claims(text4)
    claim_texts4 = [c.claim_text.lower() for c in res4.claims]
    assert len(res4.claims) == 3, f"Expected 3 atomic claims, got {len(res4.claims)}: {claim_texts4}"
    assert any("observes distant galaxies" in t for t in claim_texts4)
    assert any("northrop grumman" in t for t in claim_texts4)
    assert any("2021" in t for t in claim_texts4)

    # Case 5: Role and title appositive with temporal tenure
    text5 = "Satya Nadella, CEO of Microsoft since 2014, spoke at the conference."
    res5 = extractor.extract_atomic_claims(text5)
    claim_texts5 = [c.claim_text.lower() for c in res5.claims]
    assert len(res5.claims) == 2, f"Expected 2 atomic claims, got {len(res5.claims)}: {claim_texts5}"
    assert any("spoke at the conference" in t for t in claim_texts5)
    assert any("ceo of microsoft" in t for t in claim_texts5)

    # Case 6: Noun phrase with embedded relative clause (identity + discovery fact)
    text6 = "Marie Curie, a Polish-born physicist who discovered radium in 1898, won two Nobel Prizes."
    res6 = extractor.extract_atomic_claims(text6)
    claim_texts6 = [c.claim_text.lower() for c in res6.claims]
    assert len(res6.claims) == 3, f"Expected 3 atomic claims, got {len(res6.claims)}: {claim_texts6}"
    assert any("won two nobel prizes" in t for t in claim_texts6)
    assert any("polish-born physicist" in t for t in claim_texts6)
    assert any("discovered radium in 1898" in t for t in claim_texts6)


def test_anchor_field_bidirectional_synchronization_invariant():
    """Verify scalar == plural[0] synchronization invariant on all schemas and models."""
    from app.schemas import Claim, ClaimAnalysisRequest, AtomicClaim as SchemaAtomicClaim
    from app.services.claim_models import AtomicClaim as ServiceAtomicClaim
    from app.services.simple_claim_analyzer import SimpleClaimAnalyzer
    from uuid import uuid4

    # 1. From list only -> auto-populates scalar on Claim
    c1 = Claim(
        claim_id=uuid4(),
        text="Sample claim",
        order=0,
        temporal_anchors=["2024", "2025"],
        numerical_values=["42 km", "100 km"],
        spatial_anchors=["in London", "in UK"],
    )
    assert c1.temporal_anchor == "2024"
    assert c1.numerical_value == "42 km"
    assert c1.spatial_anchor == "in London"

    # 2. From scalar only -> auto-populates list on Claim
    c2 = Claim(
        claim_id=uuid4(),
        text="Sample claim",
        order=0,
        temporal_anchor="1969",
        numerical_value="100",
        spatial_anchor="on the Moon",
    )
    assert c2.temporal_anchors == ["1969"]
    assert c2.numerical_values == ["100"]
    assert c2.spatial_anchors == ["on the Moon"]

    # 3. Mismatch reconciliation on Claim (plural list is canonical source of truth)
    c3 = Claim(
        claim_id=uuid4(),
        text="Sample claim",
        order=0,
        temporal_anchor="old_date",
        temporal_anchors=["2024"],
        numerical_value="999",
        numerical_values=["42"],
        spatial_anchor="wrong_place",
        spatial_anchors=["Paris"],
    )
    assert c3.temporal_anchor == "2024"
    assert c3.numerical_value == "42"
    assert c3.spatial_anchor == "Paris"

    # 4. Parity on ServiceAtomicClaim
    sac = ServiceAtomicClaim(
        claim_id="c1",
        claim_text="Sample atomic claim",
        temporal_anchors=["July 20, 1969", "1969"],
        numerical_values=["2", "400000 km"],
        spatial_anchors=["on the Moon", "in orbit"],
    )
    assert sac.temporal_anchor == "July 20, 1969"
    assert sac.numerical_value == "2"
    assert sac.spatial_anchor == "on the Moon"

    sac_scalar = ServiceAtomicClaim(
        claim_id="c2",
        claim_text="Sample atomic claim",
        temporal_anchor="1969",
        numerical_value="42",
        spatial_anchor="Mars",
    )
    assert sac_scalar.temporal_anchors == ["1969"]
    assert sac_scalar.numerical_values == ["42"]
    assert sac_scalar.spatial_anchors == ["Mars"]

    # 5. Parity on SchemaAtomicClaim
    schema_ac = SchemaAtomicClaim(
        claim_id="c3",
        claim_text="Sample schema atomic claim",
        temporal_anchors=["2023"],
        numerical_values=["500"],
        spatial_anchors=["Berlin"],
    )
    assert schema_ac.temporal_anchor == "2023"
    assert schema_ac.numerical_value == "500"
    assert schema_ac.spatial_anchor == "Berlin"

    schema_ac_scalar = SchemaAtomicClaim(
        claim_id="c4",
        claim_text="Sample schema atomic claim",
        temporal_anchor="2023",
        numerical_value="500",
        spatial_anchor="Berlin",
    )
    assert schema_ac_scalar.temporal_anchors == ["2023"]
    assert schema_ac_scalar.numerical_values == ["500"]
    assert schema_ac_scalar.spatial_anchors == ["Berlin"]

    # 6. Parity on ClaimAnalysisRequest
    req_from_list = ClaimAnalysisRequest(
        claim_text="Apollo 11 landed in 1969.",
        temporal_anchors=["1969"],
        numerical_values=["11"],
        spatial_anchors=["Sea of Tranquility"],
    )
    assert req_from_list.temporal_anchor == "1969"
    assert req_from_list.numerical_value == "11"
    assert req_from_list.spatial_anchor == "Sea of Tranquility"

    req_from_scalar = ClaimAnalysisRequest(
        claim_text="Apollo 11 landed in 1969.",
        temporal_anchor="1969",
        numerical_value="11",
        spatial_anchor="Sea of Tranquility",
    )
    assert req_from_scalar.temporal_anchors == ["1969"]
    assert req_from_scalar.numerical_values == ["11"]
    assert req_from_scalar.spatial_anchors == ["Sea of Tranquility"]

    req_mismatch = ClaimAnalysisRequest(
        claim_text="Apollo 11 landed in 1969.",
        temporal_anchor="outdated",
        temporal_anchors=["1969"],
    )
    assert req_mismatch.temporal_anchor == "1969"

    # 7. Verify OpenAPI deprecation metadata across models
    for model_cls in (Claim, SchemaAtomicClaim, ServiceAtomicClaim, ClaimAnalysisRequest):
        schema = model_cls.model_json_schema()
        props = schema["properties"]
        assert props["temporal_anchor"].get("deprecated") is True, f"{model_cls.__name__} temporal_anchor not deprecated"
        assert props["numerical_value"].get("deprecated") is True, f"{model_cls.__name__} numerical_value not deprecated"
        assert props["spatial_anchor"].get("deprecated") is True, f"{model_cls.__name__} spatial_anchor not deprecated"

    # 8. SimpleClaimAnalyzer plural anchor compatibility
    analyzer = SimpleClaimAnalyzer()
    analysis = analyzer.analyze(
        claim_text="The mission completed successfully.",
        atomic_claim={
            "temporal_anchors": ["October 2023"],
            "numerical_values": ["12000"],
            "spatial_anchors": ["in Geneva"],
        },
    )
    assert any("October 2023" in q for q in analysis.ablation_log.dynamic_queries)
    assert any("12000" in q for q in analysis.ablation_log.dynamic_queries)
    assert any("in Geneva" in q for q in analysis.ablation_log.dynamic_queries)
    assert any("numerical or date-specific" in step.lower() for step in analysis.verification_plan)
    assert any("geographical and spatial" in step.lower() for step in analysis.verification_plan)



def test_all_subsystem_imports_backward_compatibility():
    """Verify all project modules and components import cleanly without circular dependency."""
    modules = [
        "app.main",
        "app.config",
        "app.dependencies",
        "app.schemas",
        "app.services.llm_client",
        "app.services.real_llm_client",
        "app.services.mock_llm_client",
        "app.services.llm_service",
        "app.services.claim_extractor",
        "app.services.simple_claim_extractor",
        "app.services.claim_analyzer",
        "app.services.simple_claim_analyzer",
        "app.services.corpus_schema",
        "app.services.corpus_loader",
        "app.services.retrieval_models",
        "app.services.bm25_retriever",
        "app.services.dense_retriever",
        "app.services.hybrid_retriever",
        "app.services.nlp_claim_extractor",
        "app.services.llm_claim_extractor",
        "app.services.claim_extraction_service",
        "app.services.claim_models",
    ]
    import ctypes

    for mod in modules:
        ctypes.pythonapi.PyErr_Clear()
        imported = __import__(mod, fromlist=["*"])
        assert imported is not None
