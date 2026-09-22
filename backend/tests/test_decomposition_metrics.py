"""Tests for Submodule 3.7: Decomposition Quality Metrics Engine.

Verifies that coverage, atomicity, entailment-preservation, dropped entities,
and high quality flags are accurately evaluated and returned via the API and session store.
"""

from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.claim_models import AtomicClaim
from app.services.decomposition_evaluator import DecompositionQualityEvaluator
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture(autouse=True)
def clean_state():
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    return TestClient(app)


def test_decomposition_metrics_unit_high_quality():
    """Verify that clean single-proposition claims achieve high coverage and atomicity."""
    evaluator = DecompositionQualityEvaluator()
    source_text = "Apollo 11 was launched on July 16, 1969. Neil Armstrong became the first person to walk on the Moon."
    claims = [
        AtomicClaim(claim_id="c1", claim_text="Apollo 11 was launched on July 16, 1969.", subject="Apollo 11"),
        AtomicClaim(claim_id="c2", claim_text="Neil Armstrong became the first person to walk on the Moon.", subject="Neil Armstrong"),
    ]

    metrics = evaluator.evaluate(source_text, claims)
    assert metrics.coverage_score >= 0.70
    assert metrics.atomicity_score >= 0.80
    assert metrics.entailment_preservation_score >= 0.70
    assert len(metrics.dropped_entities) == 0
    assert metrics.is_high_quality is True


def test_decomposition_metrics_detects_dropped_entities():
    """Verify that dropped entities and dropped coverage are flagged when facts are omitted."""
    evaluator = DecompositionQualityEvaluator()
    source_text = (
        "Apollo 11 was launched on July 16, 1969. "
        "The Saturn V rocket generated 7.5 million pounds of thrust."
    )
    # Intentionally only extract the Apollo 11 claim, completely omitting Saturn V
    claims = [
        AtomicClaim(claim_id="c1", claim_text="Apollo 11 was launched on July 16, 1969.", subject="Apollo 11"),
    ]

    metrics = evaluator.evaluate(source_text, claims)
    assert metrics.coverage_score < 0.80
    assert any("Saturn V" in ent for ent in metrics.dropped_entities)
    assert metrics.is_high_quality is False


def test_decomposition_metrics_atomicity_penalty_on_compound_clauses():
    """Verify that claims with embedded coordinating conjunctions receive atomicity penalties."""
    evaluator = DecompositionQualityEvaluator()
    source_text = "Apollo 11 landed on the Moon, and Neil Armstrong walked on the surface, while Michael Collins orbited above."
    
    # Compound non-atomic claim
    claims = [
        AtomicClaim(
            claim_id="c1",
            claim_text="Apollo 11 landed on the Moon, and Neil Armstrong walked on the surface, while Michael Collins orbited above.",
        )
    ]

    metrics = evaluator.evaluate(source_text, claims)
    assert metrics.atomicity_score <= 0.60
    assert metrics.is_high_quality is False


def test_decomposition_metrics_extractor_integration():
    """Verify that NLPClaimExtractor attaches decomposition_metrics to ClaimExtractionResult."""
    extractor = NLPClaimExtractor()
    text = (
        "Python was created by Guido van Rossum. "
        "It was first released in 1991."
    )

    result = extractor.extract_atomic_claims(text)
    assert result.decomposition_metrics is not None
    m = result.decomposition_metrics
    assert m.coverage_score > 0.0
    assert m.atomicity_score > 0.0
    assert m.is_high_quality is True


def test_decomposition_metrics_api_and_session_persistence(client):
    """Verify that POST /api/v1/extract-claims returns metrics and stores them in SQLite."""
    query_id = str(uuid4())
    payload = {
        "query": "When was Python released?",
        "query_id": query_id,
        "response_text": (
            "Python was developed by Guido van Rossum in the late 1980s. "
            "The language was released in 1991."
        ),
    }

    # First register session
    client.post("/api/v1/query", json={"query": payload["query"], "query_id": query_id})

    res = client.post("/api/v1/extract-claims", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "decomposition_metrics" in data
    metrics = data["decomposition_metrics"]
    assert metrics is not None
    assert "coverage_score" in metrics
    assert "atomicity_score" in metrics
    assert "entailment_preservation_score" in metrics
    assert "is_high_quality" in metrics
    assert metrics["coverage_score"] > 0.60

    # Verify session store recorded decomposition_metrics in metadata
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert "decomposition_metrics" in session.metadata
    stored_m = session.metadata["decomposition_metrics"]
    assert stored_m["coverage_score"] == metrics["coverage_score"]
    assert stored_m["is_high_quality"] == metrics["is_high_quality"]
