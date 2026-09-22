"""Unit tests for provenance completeness scoring and weight re-normalization."""

from datetime import date
import pytest

from app.services.source_credibility_config import (
    SourceCredibilityConfigError,
    SourceCredibilitySettings,
    SourceCredibilityWeights,
)
from app.services.source_credibility_scorer import SourceCredibilityScorer


@pytest.fixture
def scorer() -> SourceCredibilityScorer:
    return SourceCredibilityScorer()


# ---------------------------------------------------------------------------
# Provenance Completeness Tests
# ---------------------------------------------------------------------------

def test_provenance_full_completeness(scorer: SourceCredibilityScorer):
    """Verify P = 1.0 when all 6 schema-driven fields are present."""
    doc = {
        "document_id": "doc_101",
        "title": "Quantum Computation",
        "source": "Nature Physics",
        "source_url": "https://nature.com/articles/qp101",
        "domain": "nature.com",
        "published_date": date(2023, 1, 15),
    }
    result = scorer.score(doc)
    assert result.provenance_score == 1.0


def test_provenance_partial_completeness(scorer: SourceCredibilityScorer):
    """Verify intermediate P = 3 / 6 = 0.50 when 3 fields are present."""
    doc = {
        "document_id": "doc_102",
        "title": "Quantum Computation",
        "source": "Nature Physics",
        "source_url": None,
        "domain": "",
        "published_date": None,
    }
    result = scorer.score(doc)
    assert result.provenance_score == 0.50


def test_provenance_minimal_completeness(scorer: SourceCredibilityScorer):
    """Verify minimal P = 1 / 6 when only ID is present."""
    doc = {
        "document_id": "doc_103",
        "title": None,
        "source": "   ",
        "source_url": "",
        "domain": None,
        "published_date": None,
    }
    result = scorer.score(doc)
    assert abs(result.provenance_score - (1.0 / 6.0)) < 1e-4


def test_provenance_empty_metadata(scorer: SourceCredibilityScorer):
    """Verify P = 0.0 when no provenance fields are present."""
    doc = {
        "text": "Some text with no metadata at all.",
    }
    result = scorer.score(doc)
    assert result.provenance_score == 0.0


def test_provenance_empty_strings_and_none_not_counted(scorer: SourceCredibilityScorer):
    """Ensure whitespace, None, and empty lists do not increment provenance count."""
    doc = {
        "document_id": "doc_104",
        "title": "  \t\n  ",
        "source": None,
        "source_url": "",
        "domain": "   ",
        "published_date": None,
        "authors": [],
    }
    result = scorer.score(doc)
    assert result.provenance_score == round(1.0 / 6.0, 4)


# ---------------------------------------------------------------------------
# Source Type & Weight Normalization Tests
# ---------------------------------------------------------------------------

def test_source_type_absent_renormalizes_weights(scorer: SourceCredibilityScorer):
    """When source_type is absent, w_type=0.0 and remaining weights renormalize to sum to 1.0."""
    doc = {
        "document_id": "doc_201",
        "title": "Sample Title",
        "source": "Sample Source",
        "source_url": "https://nature.com/article",
    }
    result = scorer.score(doc)
    assert result.source_type_score is None
    assert "w_type" not in result.weights or result.weights.get("w_type", 0.0) == 0.0
    # Expected re-normalization: 0.50 / 0.80 = 0.625, 0.30 / 0.80 = 0.375
    assert abs(result.weights["w_domain"] - 0.625) < 1e-4
    assert abs(result.weights["w_provenance"] - 0.375) < 1e-4
    assert abs(sum(result.weights.values()) - 1.0) < 1e-4


def test_source_type_present_uses_full_weights(scorer: SourceCredibilityScorer):
    """When source_type is present, all three weights are applied and sum to 1.0."""
    doc = {
        "document_id": "doc_202",
        "title": "Sample Title",
        "source": "Sample Source",
        "source_url": "https://nature.com/article",
        "source_type": "peer-reviewed",
    }
    result = scorer.score(doc)
    assert result.source_type_score == 1.0
    assert result.weights["w_domain"] == 0.50
    assert result.weights["w_type"] == 0.20
    assert result.weights["w_provenance"] == 0.30
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Weight Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_weights_must_sum_to_one():
    with pytest.raises(SourceCredibilityConfigError, match="must sum to 1.0"):
        SourceCredibilityWeights(w_domain=0.50, w_type=0.20, w_provenance=0.20)


def test_weights_out_of_range():
    with pytest.raises(SourceCredibilityConfigError, match="must be in"):
        SourceCredibilityWeights(w_domain=1.20, w_type=-0.10, w_provenance=-0.10)


def test_weights_nan_or_inf():
    with pytest.raises(SourceCredibilityConfigError, match="finite float"):
        SourceCredibilityWeights(w_domain=float("nan"), w_type=0.20, w_provenance=0.30)
    with pytest.raises(SourceCredibilityConfigError, match="finite float"):
        SourceCredibilityWeights(w_domain=float("inf"), w_type=0.20, w_provenance=0.30)
