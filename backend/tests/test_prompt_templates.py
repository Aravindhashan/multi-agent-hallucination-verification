"""Unit tests for prompt template formatting, variable substitution, and boundaries."""

import pytest

from app.prompts import (
    CLAIM_EXTRACTION_PROMPT,
    EVIDENCE_REASONING_PROMPT,
    GENERAL_QUERY_PROMPT,
    PromptTemplate,
)


def test_custom_prompt_template_formatting():
    """Verify custom prompt template format method substitutes variables."""
    template = PromptTemplate(
        template="Hello {name}, welcome to {place}!",
        input_variables=["name", "place"],
    )
    formatted = template.format(name="Alice", place="Wonderland")
    assert formatted == "Hello Alice, welcome to Wonderland!"


def test_prompt_template_missing_variable_raises():
    """Verify missing required variables raise ValueError."""
    template = PromptTemplate(
        template="Hello {name}, welcome to {place}!",
        input_variables=["name", "place"],
    )
    with pytest.raises(ValueError) as excinfo:
        template.format(name="Alice")
    assert "Missing required prompt template variables" in str(excinfo.value)


def test_prompt_template_empty_variable_raises():
    """Verify whitespace-only variable content raises ValueError."""
    template = PromptTemplate(
        template="Topic: {topic}",
        input_variables=["topic"],
    )
    with pytest.raises(ValueError) as excinfo:
        template.format(topic="   \t\n  ")
    assert "must contain at least one non-whitespace character" in str(excinfo.value)


def test_general_query_prompt_formatting():
    """Verify GENERAL_QUERY_PROMPT formats query within data boundaries."""
    formatted = GENERAL_QUERY_PROMPT.format(query="What is photosynthesis?")
    assert "[BEGIN USER QUERY]" in formatted
    assert "What is photosynthesis?" in formatted
    assert "[END USER QUERY]" in formatted


def test_claim_extraction_prompt_formatting():
    """Verify CLAIM_EXTRACTION_PROMPT formats query, response text, and structured schema."""
    formatted = CLAIM_EXTRACTION_PROMPT.format(
        query="Tell me about Paris.",
        response_text="Paris is the capital of France. It was founded in the 3rd century BC.",
    )
    assert "[BEGIN USER QUERY]" in formatted
    assert "Tell me about Paris." in formatted
    assert "[END USER QUERY]" in formatted

    assert "[BEGIN RESPONSE DATA]" in formatted
    assert "Paris is the capital of France." in formatted
    assert "[END RESPONSE DATA]" in formatted

    assert '"claims": [' in formatted


def test_evidence_reasoning_prompt_formatting():
    """Verify EVIDENCE_REASONING_PROMPT formats claim and evidence with explicit boundaries."""
    formatted = EVIDENCE_REASONING_PROMPT.format(
        claim="Paris was founded in the 3rd century BC.",
        evidence="Historical records state that Paris emerged around 250 BC along the Seine River.",
    )
    assert "[BEGIN CLAIM DATA]" in formatted
    assert "Paris was founded in the 3rd century BC." in formatted
    assert "[END CLAIM DATA]" in formatted

    assert "[BEGIN EVIDENCE DATA]" in formatted
    assert "Historical records state that Paris emerged around 250 BC" in formatted
    assert "[END EVIDENCE DATA]" in formatted
