"""Tests for Submodule 3.6: Self-Containment Revision Engine.

Verifies that extracted atomic claims are rewritten post-SPO parsing to be fully
understandable and verifiable in isolation by resolving third-person pronouns,
implicit antecedents, and generic definite noun phrases (e.g. 'the company' -> 'Apple Inc.').
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.self_containment_reviser import AntecedentMemory, SelfContainmentReviser


@pytest.fixture
def client():
    return TestClient(app)


def test_self_containment_reviser_unit_company_resolution():
    """Verify that 'the company' resolves to the active organization entity."""
    reviser = SelfContainmentReviser()
    memory = AntecedentMemory()
    memory.register_entity("Apple Inc.", category="organization")

    claim_data = {
        "claim_text": "The company was founded in 1976.",
        "subject": "The company",
        "predicate": "was founded in",
        "object": "1976",
        "temporal_anchor": "1976",
    }

    revised = reviser.revise_claim(claim_data, memory)
    assert "Apple Inc. was founded in 1976." in revised["claim_text"]
    assert revised["subject"] == "Apple Inc."
    assert revised["is_self_contained"] is True
    assert revised["original_clause"] == "The company was founded in 1976."


def test_self_containment_reviser_unit_flight_resolution():
    """Verify that 'the flight' resolves to the active flight/mission entity."""
    reviser = SelfContainmentReviser()
    memory = AntecedentMemory()
    memory.register_entity("The Wright brothers", category="person")
    memory.register_entity("The Wright Flyer", category="vehicle")

    claim_data = {
        "claim_text": "The flight occurred in 1903 at Kitty Hawk.",
        "subject": "The flight",
        "predicate": "occurred in",
        "object": "1903 at Kitty Hawk",
        "temporal_anchor": "1903",
    }

    revised = reviser.revise_claim(claim_data, memory)
    # Must specify whose or what flight occurred in 1903
    assert "The Wright brothers' flight" in revised["claim_text"] or "The Wright Flyer flight" in revised["claim_text"]
    assert revised["is_self_contained"] is True
    assert revised["original_clause"] == "The flight occurred in 1903 at Kitty Hawk."


def test_self_containment_reviser_unit_pronoun_resolution():
    """Verify that subject pronoun 'He' and object pronoun 'it' resolve to proper antecedents."""
    reviser = SelfContainmentReviser()
    memory = AntecedentMemory()
    memory.register_entity("Guido van Rossum", category="person")
    memory.register_entity("Python", category="work")

    claim_data = {
        "claim_text": "He released it publicly in 1991.",
        "subject": "He",
        "predicate": "released",
        "object": "it publicly in 1991",
        "temporal_anchor": "1991",
    }

    revised = reviser.revise_claim(claim_data, memory)
    assert revised["claim_text"].startswith("Guido van Rossum released Python")
    assert revised["subject"] == "Guido van Rossum"
    assert revised["is_self_contained"] is True
    assert revised["original_clause"] == "He released it publicly in 1991."


def test_self_containment_nlp_extractor_pipeline():
    """Verify end-to-end atomic claim extraction with self-containment rewriting."""
    extractor = NLPClaimExtractor()
    text = (
        "The Wright brothers built the Wright Flyer in Dayton. "
        "The flight occurred in 1903 at Kitty Hawk. "
        "It lasted 12 seconds."
    )

    result = extractor.extract_atomic_claims(text)
    claims = result.claims
    assert len(claims) >= 3

    # First claim: Wright brothers built Wright Flyer
    assert "Wright brothers" in claims[0].claim_text
    assert claims[0].is_self_contained is True

    # Second claim: "The flight" should be rewritten to be self-contained
    c2 = claims[1]
    assert "Wright" in c2.claim_text
    assert "1903" in c2.claim_text
    assert c2.is_self_contained is True
    assert c2.original_clause is not None
    assert "The flight occurred in 1903" in c2.original_clause

    # Third claim: "It lasted 12 seconds" -> "It" resolved to vehicle / flight
    c3 = claims[2]
    assert not c3.claim_text.startswith("It lasted")
    assert "12 seconds" in c3.claim_text
    assert c3.is_self_contained is True


def test_self_containment_preserves_clean_claims():
    """Verify that already self-contained claims are preserved without corruption."""
    extractor = NLPClaimExtractor()
    text = "Neil Armstrong walked on the Moon on July 20, 1969."
    result = extractor.extract_atomic_claims(text)
    assert len(result.claims) == 1
    c = result.claims[0]
    assert c.claim_text == "Neil Armstrong walked on the Moon on July 20, 1969."
    assert c.subject == "Neil Armstrong"
    assert c.original_clause is None  # No revision was needed
    assert c.is_self_contained is True


def test_self_containment_api_endpoint(client):
    """Verify that POST /api/v1/extract-claims returns original_clause and is_self_contained."""
    payload = {
        "query": "Who founded Apple?",
        "response_text": (
            "Apple Inc. was founded in 1976 by Steve Jobs. "
            "The company introduced the Macintosh in 1984."
        ),
    }

    response = client.post("/api/v1/extract-claims", json=payload)
    assert response.status_code == 200
    data = response.json()
    claims = data["claims"]
    assert len(claims) >= 2

    # Second claim should have resolved "The company" -> "Apple Inc."
    c2 = claims[1]
    assert "Apple Inc." in c2["text"]
    assert c2["is_self_contained"] is True
    assert c2["original_clause"] is not None
    assert "The company introduced" in c2["original_clause"]


def test_person_misclassification_heuristics_prevent_fabrication():
    """Verify that multi-word non-person entities are never misclassified as persons (Rank 1)."""
    memory = AntecedentMemory()

    # Geopolitical and spatial entities
    assert memory._infer_category("United States") == "location"
    assert memory._infer_category("Kennedy Space Center") == "location"
    assert memory._infer_category("Cape Canaveral") == "location"

    # Judicial and government institutions
    assert memory._infer_category("Supreme Court") == "organization"
    assert memory._infer_category("White House") == "organization"

    # Historical events
    assert memory._infer_category("World War II") == "event"

    # Actual people
    assert memory._infer_category("Orville Wright") == "person"
    assert memory._infer_category("Wilbur Wright") == "person"
    assert memory._infer_category("Albert Einstein") == "person"
    assert memory._infer_category("Guido van Rossum") == "person"
    assert memory._infer_category("Dr. Jane Goodall") == "person"


def test_expanded_organizations_and_acronyms():
    """Verify that modern tech, aerospace, and acronym entities are recognized as organizations (Rank 9)."""
    memory = AntecedentMemory()

    assert memory._infer_category("Tesla") == "organization"
    assert memory._infer_category("Nvidia") == "organization"
    assert memory._infer_category("Pfizer") == "organization"
    assert memory._infer_category("Moderna") == "organization"
    assert memory._infer_category("CERN") == "organization"
    assert memory._infer_category("MIT") == "organization"
    assert memory._infer_category("WHO") == "organization"
    assert memory._infer_category("DeepMind Technologies") == "organization"

