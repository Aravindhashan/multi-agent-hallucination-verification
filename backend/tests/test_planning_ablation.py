"""Tests for Submodule 4.5: Planning Ablation Logging.

Verifies that for every analyzed claim, both the dynamically-planned query strategy
and fixed-baseline strategy are logged, difference rationales and empirical lifts are
recorded, and these logs persist cleanly into SQLite session store metadata.
"""

from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ClaimAnalysisRequest, PlanningAblationLog
from app.services.claim_models import AtomicClaim
from app.services.session_store import get_session_store, reset_session_store
from app.services.simple_claim_analyzer import SimpleClaimAnalyzer


@pytest.fixture(autouse=True)
def clean_state():
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    return TestClient(app)


def test_ablation_logging_simple_claim():
    """Verify ablation log for a basic claim without complex anchors."""
    analyzer = SimpleClaimAnalyzer()
    claim_text = "The Earth orbits the Sun."
    result = analyzer.analyze(claim_text)

    assert result.ablation_log is not None
    log = result.ablation_log
    assert isinstance(log, PlanningAblationLog)
    assert log.fixed_baseline_queries == [claim_text]
    assert len(log.dynamic_queries) >= 1
    assert "Naive single-turn" in log.fixed_baseline_strategy
    assert "Adaptive multi-anchor" in log.dynamic_strategy
    assert len(log.difference_rationale) > 0
    assert len(log.empirical_lift_expected) > 0


def test_ablation_logging_anchored_claim():
    """Verify that dynamic queries target temporal, numerical, and spatial anchors while baseline stays verbatim."""
    analyzer = SimpleClaimAnalyzer()
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Apollo 11 was launched on July 16, 1969 from Kennedy Space Center with 7.5 million pounds of thrust.",
        subject="Apollo 11",
        predicate="was launched",
        object="Kennedy Space Center",
        temporal_anchor="July 16, 1969",
        numerical_value="7.5 million pounds",
        spatial_anchor="Kennedy Space Center",
        qualifiers=["successfully"],
    )

    result = analyzer.analyze(claim.claim_text, atomic_claim=claim)
    assert result.ablation_log is not None
    log = result.ablation_log

    # Fixed baseline must be only the verbatim text
    assert log.fixed_baseline_queries == [claim.claim_text]

    # Dynamic queries must include targeted queries for temporal, numerical, spatial anchors
    assert any("July 16, 1969" in q for q in log.dynamic_queries)
    assert any("7.5 million pounds" in q for q in log.dynamic_queries)
    assert any("Kennedy Space Center" in q for q in log.dynamic_queries)
    assert any("successfully" in q for q in log.dynamic_queries)

    # Rationale and lift must detail the isolated anchors
    assert "temporal" in log.difference_rationale.lower() or "numeric" in log.difference_rationale.lower()
    assert "precision" in log.empirical_lift_expected.lower()


def test_ablation_logging_adversarial_counter_queries():
    """Verify that claims with challenge terms trigger adversarial queries in dynamic strategy."""
    analyzer = SimpleClaimAnalyzer()
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Neil Armstrong was the only astronaut on Apollo 11.",
        subject="Neil Armstrong",
        predicate="was",
        object="only astronaut",
        qualifiers=["only"],
    )

    result = analyzer.analyze(claim.claim_text, atomic_claim=claim)
    assert result.requires_deep_verification is True
    assert result.ablation_log is not None
    log = result.ablation_log

    # Must contain adversarial prosecutor counter-query
    assert any("counter-evidence" in q or "contradiction" in q for q in log.dynamic_queries)
    assert "adversarial" in log.dynamic_strategy.lower()
    assert "counter-queries" in log.difference_rationale.lower()
    assert "confirmation bias" in log.empirical_lift_expected.lower()


def test_analyze_claim_api_returns_ablation_log(client):
    """Verify that POST /api/v1/analyze-claim returns the ablation_log payload."""
    payload = {
        "claim_text": "Python 1.0 was released in January 1994.",
        "subject": "Python 1.0",
        "temporal_anchor": "January 1994",
    }
    response = client.post("/api/v1/analyze-claim", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "ablation_log" in data
    ablation = data["ablation_log"]
    assert ablation is not None
    assert "dynamic_strategy" in ablation
    assert "dynamic_queries" in ablation
    assert "fixed_baseline_strategy" in ablation
    assert "fixed_baseline_queries" in ablation
    assert "difference_rationale" in ablation
    assert "empirical_lift_expected" in ablation

    assert ablation["fixed_baseline_queries"] == [payload["claim_text"]]
    assert any("January 1994" in q for q in ablation["dynamic_queries"])


def test_analyze_claim_persists_ablation_in_session_store(client):
    """Verify that session store records planning_ablation_log in SQLite metadata."""
    query_id = str(uuid4())
    # Create session first
    client.post("/api/v1/query", json={"query": "When was Python released?", "query_id": query_id})

    payload = {
        "query_id": query_id,
        "claim_text": "Python was released in 1991.",
        "subject": "Python",
        "temporal_anchor": "1991",
    }
    response = client.post("/api/v1/analyze-claim", json=payload)
    assert response.status_code == 200

    # Retrieve session directly from SQLite session store
    store = get_session_store()
    session = store.get_session(UUID(query_id))
    assert session is not None
    assert "planning_ablation_log" in session.metadata

    persisted_log = session.metadata["planning_ablation_log"]
    assert persisted_log["fixed_baseline_queries"] == [payload["claim_text"]]
    assert any("1991" in q for q in persisted_log["dynamic_queries"])
    assert len(persisted_log["difference_rationale"]) > 0
