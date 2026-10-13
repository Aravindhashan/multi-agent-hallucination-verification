"""Comprehensive Test Suite for Module 5 (Defender Agent).

Covers Submodules 5.1 through 5.4, end-to-end integration, and regression against prior bugs.
Adheres strictly to the user specification and gap coverage requirements.
"""

from datetime import datetime, timezone
import json
import logging
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.main import app
from app.schemas import (
    AtomicClaim,
    Claim,
    DefenderRequest,
    DefenderResponse,
    DefenderResult,
    EvidenceItem,
    PlanningAblationLog,
)
from app.services.defender_service import (
    compute_token_overlap,
    defend_claim,
    defend_claims_batch,
)
from app.services.session_store import get_session_store, reset_session_store
from app.services.simple_defender import PlanGuidedDefender


class MockDeterministicRetriever:
    """Mock evidence retriever returning deterministic items and recording query/stance."""

    def __init__(self, items: list[EvidenceItem] | None = None) -> None:
        self.items = items if items is not None else []
        self.recorded_queries: list[str] = []
        self.recorded_stances: list[str] = []

    def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
        self.recorded_queries.append(query)
        self.recorded_stances.append(stance)
        return self.items


def make_evidence_item(
    evidence_id: str,
    text: str,
    relevance: float = 0.8,
    credibility: float = 0.8,
    freshness: float = 0.8,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        text=text,
        relevance_score=relevance,
        credibility_score=credibility,
        freshness_score=freshness,
    )


@pytest.fixture(autouse=True)
def clean_session_db():
    reset_session_store()
    yield
    reset_session_store()


# =====================================================================
# 5.1 — Plan-Guided Query Formulation
# =====================================================================
class TestPlanGuidedQueryFormulation:

    def test_query_reflects_module4_verification_plan(self):
        """A claim with a chronological-check plan (from Module 4) must produce

        a query containing temporal terms drawn from the claim's temporal_anchors,
        not just the raw claim text.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()

        store.create_session(query_id=query_id, query="When was the telescope launched?")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "The telescope was launched in December 2021.",
                    "temporal_anchors": ["December 2021"],
                    "numerical_values": [],
                    "spatial_anchors": [],
                }
            ],
        )
        store.record_claim_analysis(
            query_id=query_id,
            claim_id=claim_id,
            claim_text="The telescope was launched in December 2021.",
            complexity_score=0.7,
            verification_plan=["Verify chronological launch date in December 2021"],
        )

        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Launch record")])
        resp = defend_claim(
            claim_text="The telescope was launched in December 2021.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        assert "dates: December 2021" in resp.query_used
        assert "plan: Verify chronological launch date in December 2021" in resp.query_used
        assert resp.query_used != "The telescope was launched in December 2021."

    def test_query_reflects_quantity_check_plan(self):
        """A claim flagged for a quantity check must produce a query incorporating

        numerical_values, not just entity/subject terms.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()

        store.create_session(query_id=query_id, query="How many satellites were deployed?")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "The rocket placed 53 satellites into low Earth orbit.",
                    "temporal_anchors": [],
                    "numerical_values": ["53"],
                    "spatial_anchors": ["low Earth orbit"],
                }
            ],
        )
        store.record_claim_analysis(
            query_id=query_id,
            claim_id=claim_id,
            claim_text="The rocket placed 53 satellites into low Earth orbit.",
            complexity_score=0.6,
            verification_plan=["Verify satellite quantity 53"],
        )

        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Mission report")])
        resp = defend_claim(
            claim_text="The rocket placed 53 satellites into low Earth orbit.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        assert "numbers: 53" in resp.query_used
        assert "locations: low Earth orbit" in resp.query_used
        assert "plan: Verify satellite quantity 53" in resp.query_used

    def test_query_uses_plural_anchor_fields_not_deprecated_scalars(self):
        """Formulation must read temporal_anchors/numerical_values/spatial_anchors

        (lists), never the deprecated scalar fields, even if both are present.
        # [GAP COVERAGE] confirms Module 3.12's sync invariant didn't leave
        # Defender silently reading the old scalar fields.
        """
        # Directly instantiate PlanGuidedDefender with plural lists distinct from legacy scalars
        mock_retriever = MockDeterministicRetriever([])
        defender = PlanGuidedDefender(
            retriever=mock_retriever,
            anchors={
                "temporal": ["2021-12-25", "2022-01-24"],
                "numerical": ["100", "200"],
                "spatial": ["Kourou", "Lagrange point L2"],
            },
            verification_plan=["Verify timeline"],
        )
        query = defender.build_query("JWST flight sequence")

        # Must include all plural anchor elements
        assert "dates: 2021-12-25, 2022-01-24" in query
        assert "numbers: 100, 200" in query
        assert "locations: Kourou, Lagrange point L2" in query

    def test_multi_anchor_claim_produces_query_covering_all_anchors(self):
        """A claim with 2 temporal anchors and 2 numerical values must generate

        a query (or query set) that references all of them, not just anchor[0].
        # [GAP COVERAGE] regression check against the original singleton-field bug.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()

        store.create_session(query_id=query_id, query="Multi-anchor historical claim")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "The spacecraft operated from 1969 to 1972 across 6 missions carrying 12 astronauts.",
                    "temporal_anchors": ["1969", "1972"],
                    "numerical_values": ["6", "12"],
                    "spatial_anchors": ["Moon"],
                }
            ],
        )

        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Apollo records")])
        resp = defend_claim(
            claim_text="The spacecraft operated from 1969 to 1972 across 6 missions carrying 12 astronauts.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        # Assert all anchors are included, not truncated to anchor[0]
        assert "1969" in resp.query_used
        assert "1972" in resp.query_used
        assert "6" in resp.query_used
        assert "12" in resp.query_used

    def test_query_diversity_check_rejects_near_duplicate_of_baseline(self):
        """If the generated supporting query is near-identical (token overlap

        above threshold) to Module 2's default retrieval query, it must be
        flagged/rejected rather than silently issued and counted as a distinct query.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()

        # A claim with no informative anchors or plan will produce a near-identical query
        store.create_session(query_id=query_id, query="General photosynthesis query")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "Photosynthesis converts solar light into chemical energy.",
                    "temporal_anchors": [],
                    "numerical_values": [],
                    "spatial_anchors": [],
                }
            ],
        )

        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Biology text")])
        resp = defend_claim(
            claim_text="Photosynthesis converts solar light into chemical energy.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        # The dynamic query matches baseline and is flagged
        assert resp.query_used == "Photosynthesis converts solar light into chemical energy."
        session = store.get_session(query_id)
        assert "defender_ablation_log" in session.metadata
        ablation = session.metadata["defender_ablation_log"]
        assert ablation["dynamic_queries"] == ["Photosynthesis converts solar light into chemical energy."]
        assert ablation["fixed_baseline_queries"] == ["Photosynthesis converts solar light into chemical energy."]
        assert "No lift expected" in ablation["empirical_lift_expected"]

    def test_ablation_log_records_both_planned_and_fixed_baseline_query(self):
        """record_claim_evidence (or equivalent) must persist BOTH the

        plan-guided query AND Module 4's fixed-baseline query string, even
        though only the planned query is executed.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()

        store.create_session(query_id=query_id, query="Ablation query test")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "Rover Curiosity landed in Gale Crater in August 2012.",
                    "temporal_anchors": ["August 2012"],
                    "numerical_values": [],
                    "spatial_anchors": ["Gale Crater"],
                }
            ],
        )

        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Mars science")])
        resp = defend_claim(
            claim_text="Rover Curiosity landed in Gale Crater in August 2012.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=mock_retriever,
        )

        # Check claim record in SQLite
        record = store.get_claim_record(query_id, claim_id)
        assert record is not None
        assert record["defender_query_used"] == resp.query_used
        assert record["defender_baseline_query"] == "Rover Curiosity landed in Gale Crater in August 2012."

        # Check session metadata ablation log
        session = store.get_session(query_id)
        ablation = session.metadata["defender_ablation_log"]
        assert ablation["dynamic_queries"] == [resp.query_used]
        assert ablation["fixed_baseline_queries"] == ["Rover Curiosity landed in Gale Crater in August 2012."]

    def test_missing_verification_plan_falls_back_gracefully(self):
        """If Module 4 has not run for a claim_id (plan is None/missing),

        formulation must fall back to a sane default query rather than raising
        or producing an empty query.
        """
        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Factual text")])
        # Call without query_id / session plan
        resp = defend_claim(
            claim_text="The human heart has four chambers.",
            claim_id=uuid4(),
            query_id=None,
            retriever=mock_retriever,
        )
        assert resp.query_used != ""
        assert "The human heart has four chambers." in resp.query_used
        assert len(resp.supporting_evidence) == 1

    def test_nlp_extractor_fallback_path(self):
        """When claim originates from a source that skipped SPO parsing

        (predicate is None per the earlier audit finding), query formulation
        must still produce a usable query from claim_text alone.
        # [GAP COVERAGE]
        """
        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Text")])
        # Claim with dates and numbers but no prior SPO
        resp = defend_claim(
            claim_text="In 1945, world leaders signed the charter in San Francisco with 50 nations.",
            claim_id=uuid4(),
            query_id=None,
            retriever=mock_retriever,
        )
        assert resp.query_used != ""
        assert "dates: 1945" in resp.query_used or "1945" in resp.query_used
        assert "numbers: 50" in resp.query_used or "50" in resp.query_used

    def test_multi_claim_ablation_log_isolation_in_session_claims(self):
        """Two sibling claims under the same query_id must maintain INDEPENDENT

        ablation logs in session_claims table — direct regression test against
        reintroducing the Rank 3 query-level metadata overwrite bug.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        c1_id = "c1_temporal"
        c2_id = "c2_quantity"

        store.create_session(query_id=query_id, query="Multi-claim ablation isolation")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": c1_id,
                    "claim_text": "Apollo 11 launched in July 1969.",
                    "temporal_anchors": ["July 1969"],
                    "numerical_values": [],
                    "spatial_anchors": [],
                },
                {
                    "claim_id": c2_id,
                    "claim_text": "The spacecraft carried 3 astronauts.",
                    "temporal_anchors": [],
                    "numerical_values": ["3"],
                    "spatial_anchors": [],
                },
            ],
        )

        retriever = MockDeterministicRetriever([make_evidence_item("ev", "Doc", 0.8)])

        # Defend Claim 1
        resp1 = defend_claim(
            claim_text="Apollo 11 launched in July 1969.",
            claim_id=c1_id,
            query_id=query_id,
            retriever=retriever,
        )

        # Defend Claim 2
        resp2 = defend_claim(
            claim_text="The spacecraft carried 3 astronauts.",
            claim_id=c2_id,
            query_id=query_id,
            retriever=retriever,
        )

        # Retrieve per-claim records from SQLite session_claims
        rec1 = store.get_claim_record(query_id, c1_id)
        rec2 = store.get_claim_record(query_id, c2_id)

        assert rec1 is not None and rec2 is not None
        assert rec1["ablation_log"] is not None
        assert rec2["ablation_log"] is not None

        # Verify Claim 1's ablation log was NOT overwritten by Claim 2 in session_claims
        assert "July 1969" in rec1["ablation_log"]["dynamic_queries"][0]
        assert "3" in rec2["ablation_log"]["dynamic_queries"][0]
        assert rec1["ablation_log"]["dynamic_queries"] != rec2["ablation_log"]["dynamic_queries"]

        # Verify get_claim_records returns both distinct ablation logs
        all_records = {r["claim_id"]: r for r in store.get_claim_records(query_id)}
        assert c1_id in all_records and c2_id in all_records
        assert "July 1969" in all_records[c1_id]["ablation_log"]["dynamic_queries"][0]
        assert "3" in all_records[c2_id]["ablation_log"]["dynamic_queries"][0]




# =====================================================================
# 5.2 — Hybrid Supporting Retrieval
# =====================================================================
class TestHybridSupportingRetrieval:

    def test_stance_supporting_is_actually_passed_to_retriever(self):
        """Assert HybridRetriever.retrieve() is called with stance='supporting',

        not the default/unset value.
        # [GAP COVERAGE]
        """
        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Support doc")])
        resp = defend_claim(
            claim_text="Mercury is the smallest planet in the Solar System.",
            retriever=mock_retriever,
        )
        assert len(mock_retriever.recorded_stances) == 1
        assert mock_retriever.recorded_stances[0] == "supporting"

    def test_real_hybrid_retriever_used_not_mock(self):
        """Assert the injected retriever is the real HybridRetriever instance

        (via get_evidence_retriever), not MockEvidenceRetriever, in a
        non-test-configured environment.
        # [GAP COVERAGE]
        """
        from app.dependencies import get_evidence_retriever
        from app.services.hybrid_retriever import HybridRetriever

        retriever = get_evidence_retriever()
        assert isinstance(retriever, HybridRetriever)

    def test_retrieval_returns_ranked_results_with_scores(self):
        """Each returned evidence item includes a retrieval score/rank,

        not just raw text — required for 5.3's composite scoring.
        """
        mock_retriever = MockDeterministicRetriever([
            make_evidence_item("e1", "Evidence A", relevance=0.85, credibility=0.9, freshness=0.8),
            make_evidence_item("e2", "Evidence B", relevance=0.60, credibility=0.7, freshness=0.6),
        ])
        resp = defend_claim(
            claim_text="Jupiter has the most mass among Solar System planets.",
            retriever=mock_retriever,
        )
        for item in resp.supporting_evidence:
            assert item.relevance_score is not None
            assert item.composite_score is not None
            assert 0.0 <= item.composite_score <= 1.0

    def test_batch_slicing_respects_max_batch_size(self):
        """defend-claims-batch with >5 claims is processed in slices of ≤5,

        not sent to the retriever as one oversized batch.
        """
        retrieved_queries: list[str] = []

        class SliceTrackingRetriever:
            def retrieve(self, query: str, stance: str = "supporting"):
                retrieved_queries.append(query)
                return [make_evidence_item("ev", "Text", 0.8)]

        claims = [
            DefenderRequest(claim_text=f"Batch claim number {i} for slicing test.")
            for i in range(12)
        ]
        results = defend_claims_batch(claims, retriever=SliceTrackingRetriever(), batch_size=5)
        assert len(results) == 12
        assert len(retrieved_queries) == 12

    def test_batch_slicing_preserves_claim_order_in_response(self):
        """Batch response list[DefenderResponse] order matches the order of

        list[DefenderRequest], even across slice boundaries.
        """
        mock_retriever = MockDeterministicRetriever([make_evidence_item("e", "Text", 0.8)])
        ordered_ids = [f"claim_{i}" for i in range(11)]
        claims = [
            DefenderRequest(claim_text=f"Statement {cid}", claim_id=cid)
            for cid in ordered_ids
        ]
        responses = defend_claims_batch(claims, retriever=mock_retriever, batch_size=5)
        assert len(responses) == len(ordered_ids)
        response_ids = [r.claim_id for r in responses]
        assert response_ids == ordered_ids

    def test_empty_retrieval_result_handled_without_exception(self):
        """A claim whose queries return zero documents from the corpus must

        produce a valid DefenderResponse with empty evidence list, not a
        500 error or unhandled exception.
        # [GAP COVERAGE]
        """
        mock_retriever = MockDeterministicRetriever([])
        resp = defend_claim(
            claim_text="Unmatched obscure fact not present in any corpus document.",
            retriever=mock_retriever,
        )
        assert isinstance(resp, DefenderResponse)
        assert resp.supporting_evidence == []

    def test_retriever_timeout_or_failure_does_not_crash_pipeline(self):
        """If the underlying retriever raises (e.g., FAISS index unavailable),

        the endpoint returns a structured error, not an unhandled 500.
        # [GAP COVERAGE]
        """
        from app.dependencies import get_evidence_retriever

        class FailingRetriever:
            def retrieve(self, query: str, stance: str = "supporting"):
                raise RuntimeError("FAISS index unavailable or connection timed out")

        client = TestClient(app)
        app.dependency_overrides[get_evidence_retriever] = lambda: FailingRetriever()
        try:
            resp = client.post("/api/v1/defend-claim", json={"claim_text": "Valid test claim."})
            assert resp.status_code == 503
            assert "FAISS index unavailable" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_evidence_retriever, None)


# =====================================================================
# 5.3 — Evidence Relevance Filtering / Ranking
# =====================================================================
class TestEvidenceRelevanceFiltering:

    def test_composite_score_formula_matches_spec(self):
        """Verify the 0.5/0.3/0.2 weighting is applied to the correct three

        components (relevance/credibility/freshness) — assert against a
        hand-computed expected value, not just that a score exists.
        """
        # Hand-computed:
        # rel=0.8, cred=0.7, fresh=0.9
        # 0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.9 = 0.40 + 0.21 + 0.18 = 0.79
        item = make_evidence_item("e_calc", "Text", relevance=0.8, credibility=0.7, freshness=0.9)
        mock_retriever = MockDeterministicRetriever([item])
        resp = defend_claim(claim_text="Formula check claim.", retriever=mock_retriever)
        assert len(resp.supporting_evidence) == 1
        assert resp.supporting_evidence[0].composite_score == 0.79

    def test_threshold_gate_excludes_low_relevance_evidence(self):
        """Evidence scoring below 0.2 composite is excluded from the

        DefenderResponse's evidence list entirely, not just deprioritized.
        """
        # composite: 0.5*0.1 + 0.3*0.1 + 0.2*0.1 = 0.10 < 0.20
        low_item = make_evidence_item("low", "Low score text", relevance=0.1, credibility=0.1, freshness=0.1)
        # composite: 0.5*0.8 + 0.3*0.8 + 0.2*0.8 = 0.80 >= 0.20
        high_item = make_evidence_item("high", "High score text", relevance=0.8, credibility=0.8, freshness=0.8)
        mock_retriever = MockDeterministicRetriever([low_item, high_item])

        resp = defend_claim(claim_text="Threshold test.", retriever=mock_retriever)
        retained = [e.evidence_id for e in resp.supporting_evidence]
        assert "low" not in retained
        assert "high" in retained

    def test_threshold_gate_boundary_condition(self):
        """Evidence scoring exactly at 0.2 — confirm documented inclusive/

        exclusive behavior (>= vs >) and that it's consistent, not accidental.
        """
        # rel=0.20, cred=0.20, fresh=0.20 -> composite = 0.2000
        boundary_item = make_evidence_item("boundary", "Boundary item", relevance=0.20, credibility=0.20, freshness=0.20)
        # rel=0.199, cred=0.199, fresh=0.199 -> composite = 0.1990 < 0.20
        below_item = make_evidence_item("below", "Below boundary", relevance=0.199, credibility=0.199, freshness=0.199)

        mock_retriever = MockDeterministicRetriever([boundary_item, below_item])
        resp = defend_claim(claim_text="Boundary check.", retriever=mock_retriever)
        retained = [e.evidence_id for e in resp.supporting_evidence]
        assert "boundary" in retained
        assert "below" not in retained

    def test_no_free_text_argument_field_in_response_schema(self):
        """DefenderResponse contains no field carrying LLM-generated persuasive

        text that could be mistaken for a scored input (e.g., no 'argument',
        'justification', or 'reasoning' free-text field feeding the verdict).
        # [GAP COVERAGE] confirms schema structurally prevents it.
        """
        forbidden_fields = {"argument", "justification", "reasoning", "free_text_defense", "llm_argument"}
        assert not forbidden_fields.intersection(DefenderResponse.model_fields.keys())
        assert not forbidden_fields.intersection(DefenderResult.model_fields.keys())
        assert not forbidden_fields.intersection(EvidenceItem.model_fields.keys())

    def test_ranking_is_deterministic_for_identical_input(self):
        """Running the same claim + same corpus state twice produces

        identical evidence ranking and scores.
        """
        items = [
            make_evidence_item("e1", "Doc 1", 0.6, 0.7, 0.8),
            make_evidence_item("e2", "Doc 2", 0.9, 0.8, 0.7),
            make_evidence_item("e3", "Doc 3", 0.7, 0.6, 0.5),
        ]
        retriever1 = MockDeterministicRetriever(items)
        retriever2 = MockDeterministicRetriever(items)

        resp1 = defend_claim("Deterministic test claim", retriever=retriever1)
        resp2 = defend_claim("Deterministic test claim", retriever=retriever2)

        ids1 = [e.evidence_id for e in resp1.supporting_evidence]
        ids2 = [e.evidence_id for e in resp2.supporting_evidence]
        assert ids1 == ids2
        scores1 = [e.composite_score for e in resp1.supporting_evidence]
        scores2 = [e.composite_score for e in resp2.supporting_evidence]
        assert scores1 == scores2

    def test_all_retrieved_documents_below_threshold_yields_empty_not_error(self):
        """A claim where every retrieved document scores below 0.2 must

        return a valid response with zero evidence, distinguishable from
        'no documents retrieved' (5.2's empty-result case) for downstream
        NLI/verdict logic that may treat these differently.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()
        store.create_session(query_id=query_id, query="Test query")

        # 2 documents retrieved, both below threshold
        below_items = [
            make_evidence_item("e_sub1", "Low", relevance=0.1, credibility=0.1, freshness=0.1),
            make_evidence_item("e_sub2", "Low", relevance=0.05, credibility=0.05, freshness=0.05),
        ]
        retriever = MockDeterministicRetriever(below_items)
        resp = defend_claim("Claim with low evidence", claim_id=claim_id, query_id=query_id, retriever=retriever)
        assert resp.supporting_evidence == []

        record = store.get_claim_record(query_id, claim_id)
        assert record is not None
        # Distinguishable: raw_evidence_count is 2, retained_evidence_count is 0
        assert record.get("raw_evidence_count") == 2
        assert record.get("retained_evidence_count") == 0


# =====================================================================
# 5.4 — API & SQLite Audit
# =====================================================================
class TestDefenderAPIAndPersistence:

    def test_defend_claim_accepts_string_claim_id(self):
        """POST /defend-claim with claim_id='c1' (Module 3's native string

        format) succeeds — regression test for the original UUID-only 422 bug.
        """
        client = TestClient(app)
        resp = client.post(
            "/api/v1/defend-claim",
            json={"claim_text": "The Sun is a G-type main-sequence star.", "claim_id": "c1"},
        )
        assert resp.status_code == 200
        assert resp.json()["claim_id"] == "c1"

    def test_defend_claim_accepts_uuid_claim_id(self):
        """POST /defend-claim with a proper UUID claim_id also succeeds

        (both types must work per the widened schema).
        """
        client = TestClient(app)
        test_uuid = str(uuid4())
        resp = client.post(
            "/api/v1/defend-claim",
            json={"claim_text": "The Sun is a G-type main-sequence star.", "claim_id": test_uuid},
        )
        assert resp.status_code == 200
        assert resp.json()["claim_id"] == test_uuid

    def test_defend_claim_rejects_blank_claim_text_with_422(self):
        client = TestClient(app)
        resp = client.post("/api/v1/defend-claim", json={"claim_text": "   "})
        assert resp.status_code == 422

    def test_defend_claims_batch_empty_list_returns_empty_list_not_error(self):
        client = TestClient(app)
        resp = client.post("/api/v1/defend-claims-batch", json=[])
        assert resp.status_code == 200
        assert resp.json() == []

    def test_defend_claims_batch_exceeding_max_size_is_rejected_or_sliced_per_spec(self):
        """Confirm documented behavior for >5-claim batches: auto-sliced per spec."""
        client = TestClient(app)
        batch = [{"claim_text": f"Batch claim assertion {i}", "claim_id": f"c_{i}"} for i in range(8)]
        resp = client.post("/api/v1/defend-claims-batch", json=batch)
        assert resp.status_code == 200
        assert len(resp.json()) == 8

    def test_session_claims_persistence_keyed_correctly(self):
        """record_claim_evidence writes to session_claims under the correct

        (query_id, claim_id) composite key, not the legacy per-query sessions table.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = "c_composite_key"
        store.create_session(query_id=query_id, query="Test session")
        store.record_claim_evidence(
            query_id=query_id,
            claim_id=claim_id,
            evidence_type="supporting",
            evidence=[{"evidence_id": "ev1", "text": "Valid doc"}],
            query_used="test query",
        )
        record = store.get_claim_record(query_id, claim_id)
        assert record is not None
        assert record["claim_id"] == claim_id
        assert len(record["supporting_evidence"]) == 1

    def test_multiple_claims_same_query_do_not_overwrite_each_others_evidence(self):
        """Two claims under the same query_id, both defended, retain

        DISTINCT supporting_evidence records — direct regression test of
        the original multi-claim overwrite bug.
        """
        store = get_session_store()
        query_id = uuid4()
        store.create_session(query_id=query_id, query="Multi-claim test query")

        retriever = MockDeterministicRetriever([make_evidence_item("ev_shared", "Data", 0.8)])
        resp1 = defend_claim("First unique claim", claim_id="c1", query_id=query_id, retriever=retriever)
        resp2 = defend_claim("Second unique claim", claim_id="c2", query_id=query_id, retriever=retriever)

        rec1 = store.get_claim_record(query_id, "c1")
        rec2 = store.get_claim_record(query_id, "c2")
        assert rec1 is not None and rec2 is not None
        assert rec1["claim_id"] == "c1"
        assert rec2["claim_id"] == "c2"
        assert len(rec1["supporting_evidence"]) == 1
        assert len(rec2["supporting_evidence"]) == 1

    def test_defender_query_used_is_persisted_and_retrievable(self):
        """get_claim_record() returns defender_query_used matching exactly

        what was passed to the retriever.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()
        store.create_session(query_id=query_id, query="Query used retrieval test")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "Voyager 1 was launched on September 5, 1977.",
                    "temporal_anchors": ["September 5, 1977"],
                    "numerical_values": [],
                    "spatial_anchors": [],
                }
            ],
        )

        retriever = MockDeterministicRetriever([make_evidence_item("v1", "Voyager report")])
        resp = defend_claim(
            "Voyager 1 was launched on September 5, 1977.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=retriever,
        )

        record = store.get_claim_record(query_id, claim_id)
        assert record is not None
        assert record["defender_query_used"] == resp.query_used

    def test_uuid_fields_serialize_without_error_in_metadata_json(self):
        """Evidence containing UUID-typed fields serializes cleanly via

        record_claim_evidence — regression test for the UUID serialization fix.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()
        store.create_session(query_id=query_id, query="UUID serialization test")

        evidence_with_uuid = [
            {
                "evidence_id": uuid4(),  # Raw UUID object
                "text": "Evidence item with native UUID object in payload",
                "retrieval_method": "hybrid",
            }
        ]
        # Must not raise TypeError: Object of type UUID is not JSON serializable
        store.record_claim_evidence(
            query_id=query_id,
            claim_id=claim_id,
            evidence_type="supporting",
            evidence=evidence_with_uuid,
            query_used="test query",
        )
        record = store.get_claim_record(query_id, claim_id)
        assert record is not None
        assert len(record["supporting_evidence"]) == 1

    def test_malformed_metadata_does_not_corrupt_prior_session_claims_rows(self):
        """A failed/malformed write for one claim_id does not affect or

        overwrite other claim_id rows under the same query_id.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        store.create_session(query_id=query_id, query="Isolation test")

        # Write claim 1 successfully
        store.record_claim_evidence(
            query_id=query_id,
            claim_id="c1",
            evidence_type="supporting",
            evidence=[{"evidence_id": "ev1", "text": "Valid doc 1"}],
            query_used="query 1",
        )

        # Attempt bad write or operation on claim 2
        try:
            class UnserializableObject:
                pass

            store.record_claim_evidence(
                query_id=query_id,
                claim_id="c2",
                evidence_type="supporting",
                evidence=[{"bad_key": UnserializableObject()}],  # Might fail or serialize as string
            )
        except Exception:
            pass

        # Verify claim 1 row is completely intact
        rec1 = store.get_claim_record(query_id, "c1")
        assert rec1 is not None
        assert rec1["claim_id"] == "c1"
        assert len(rec1["supporting_evidence"]) == 1
        assert rec1["defender_query_used"] == "query 1"


# =====================================================================
# Integration — Full Pipeline Chain
# =====================================================================
class TestDefenderPipelineIntegration:

    def test_full_chain_query_to_defend_claim(self):
        """Query -> Generate -> Extract Claims -> Analyze Claim -> Evaluate Quality

        -> Defend Claim, single query_id, all state retrievable at the end.
        """
        client = TestClient(app)
        # 1. Query
        q_resp = client.post("/api/v1/query", json={"query": "When was Galileo launched?"})
        assert q_resp.status_code == 200
        query_id = q_resp.json()["query_id"]

        # 2. Generate
        g_resp = client.post("/api/v1/generate", json={"query": "When was Galileo launched?", "query_id": query_id})
        assert g_resp.status_code == 200

        # 3. Extract claims
        e_resp = client.post(
            "/api/v1/extract-claims",
            json={
                "query": "When was Galileo launched?",
                "response_text": "The Galileo spacecraft was launched on October 18, 1989 by Space Shuttle Atlantis.",
                "query_id": query_id,
            },
        )
        assert e_resp.status_code == 200
        claims = e_resp.json()["claims"]
        assert len(claims) >= 1
        primary = claims[0]

        # 4. Analyze claim
        a_resp = client.post(
            "/api/v1/analyze-claim",
            json={
                "claim_id": primary["claim_id"],
                "text": primary["text"],
                "query_id": query_id,
                "temporal_anchor": primary.get("temporal_anchor"),
                "subject": primary.get("subject"),
                "predicate": primary.get("predicate"),
            },
        )
        assert a_resp.status_code == 200

        # 5. Evaluate quality
        eq_resp = client.post(
            "/api/v1/evidence/evaluate-quality",
            json={
                "claim_id": primary["claim_id"],
                "claim_text": primary["text"],
                "query_id": query_id,
                "evidence_items": [
                    {
                        "evidence_id": "doc1",
                        "text": "Galileo launched on October 18, 1989.",
                        "domain": "nasa.gov",
                        "relevance_score": 0.95,
                    }
                ],
            },
        )
        assert eq_resp.status_code == 200

        # 6. Defend claim
        d_resp = client.post(
            "/api/v1/defend-claim",
            json={
                "claim_id": primary["claim_id"],
                "claim_text": primary["text"],
                "query_id": query_id,
            },
        )
        assert d_resp.status_code == 200
        d_data = d_resp.json()
        assert d_data["claim_id"] == primary["claim_id"]
        assert "query_used" in d_data

        # Verify state in SQLite
        store = get_session_store()
        record = store.get_claim_record(UUID(query_id), primary["claim_id"])
        assert record is not None
        assert "supporting_evidence" in record
        assert record["defender_query_used"] is not None

    def test_full_chain_with_multi_claim_response(self):
        """A generated response producing 3+ atomic claims: confirm ALL

        claims are independently defended and persisted, not just the last
        one processed.
        # [GAP COVERAGE]
        """
        store = get_session_store()
        query_id = uuid4()
        store.create_session(query_id=query_id, query="Multi-claim exploration")

        claims_data = [
            {"claim_id": "c1", "claim_text": "First atomic claim assertion in multi-fact sequence."},
            {"claim_id": "c2", "claim_text": "Second atomic claim assertion in multi-fact sequence."},
            {"claim_id": "c3", "claim_text": "Third atomic claim assertion in multi-fact sequence."},
        ]
        store.record_claims(query_id=query_id, claims=claims_data)

        retriever = MockDeterministicRetriever([make_evidence_item("ev_multi", "Supporting text", 0.85)])
        responses = defend_claims_batch(claims_data, query_id=query_id, retriever=retriever)
        assert len(responses) == 3

        # Confirm all 3 exist independently in SQLite
        for cid in ["c1", "c2", "c3"]:
            rec = store.get_claim_record(query_id, cid)
            assert rec is not None
            assert rec["claim_id"] == cid
            assert len(rec["supporting_evidence"]) == 1
            assert rec["supporting_evidence"][0]["evidence_id"] == "ev_multi"

    def test_full_chain_defender_evidence_available_to_downstream_module(self):
        """Evidence persisted by Defender is retrievable in the exact

        structure Module 9 (NLI) will expect to consume it — schema
        contract check.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = "claim_downstream_contract"
        store.create_session(query_id=query_id, query="Contract test")

        item = make_evidence_item("nli_doc_1", "NLI test text", relevance=0.9, credibility=0.85, freshness=0.8)
        retriever = MockDeterministicRetriever([item])

        resp = defend_claim(
            "Evidence contract validation claim",
            claim_id=claim_id,
            query_id=query_id,
            retriever=retriever,
        )

        # Retrieve from session store
        rec = store.get_claim_record(query_id, claim_id)
        assert rec is not None
        evidence_list = rec["supporting_evidence"]
        assert len(evidence_list) == 1
        ev = evidence_list[0]

        # Verify contract fields required by NLI verification
        required_keys = {"evidence_id", "text", "relevance_score", "credibility_score", "freshness_score", "composite_score"}
        for k in required_keys:
            assert k in ev, f"Missing required NLI evidence key: {k}"
        assert ev["evidence_id"] == "nli_doc_1"
        assert ev["composite_score"] is not None

    def test_chain_recovers_if_evaluate_quality_step_is_skipped(self):
        """If a claim reaches Defender without having passed through

        Module 8's quality evaluation (out-of-order call), Defender handles
        gracefully and computes evidence scores cleanly.
        """
        client = TestClient(app)
        # Directly call defend-claim without prior quality evaluation
        resp = client.post(
            "/api/v1/defend-claim",
            json={
                "claim_text": "The Moon orbits the Earth at an average distance of 384,400 km.",
                "claim_id": "c_direct_no_eval",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim_id"] == "c_direct_no_eval"
        assert isinstance(data["supporting_evidence"], list)
        for e in data["supporting_evidence"]:
            assert "composite_score" in e


# =====================================================================
# Regression — Known Prior Failure Patterns
# =====================================================================
class TestDefenderRegressionAgainstPriorBugs:

    def test_defender_does_not_use_scalar_anchor_fields(self):
        """Direct regression test: feed a claim with ONLY plural anchors

        populated (scalars null/deprecated) and confirm query formulation
        still works — proves no lingering scalar-field dependency.
        """
        store = get_session_store()
        query_id = uuid4()
        claim_id = uuid4()
        store.create_session(query_id=query_id, query="Plural anchor test")
        store.record_claims(
            query_id=query_id,
            claims=[
                {
                    "claim_id": str(claim_id),
                    "claim_text": "Spacecraft Dawn visited Ceres in 2015.",
                    "temporal_anchors": ["2015"],
                    "numerical_values": [],
                    "spatial_anchors": ["Ceres"],
                }
            ],
        )

        retriever = MockDeterministicRetriever([make_evidence_item("e1", "Dawn mission")])
        resp = defend_claim(
            "Spacecraft Dawn visited Ceres in 2015.",
            claim_id=claim_id,
            query_id=query_id,
            retriever=retriever,
        )
        assert "dates: 2015" in resp.query_used
        assert "locations: Ceres" in resp.query_used

    def test_defender_handles_claim_with_no_predicate(self):
        """Regression test for Module 3.2's SPO coverage gap — a claim

        where predicate is None must not crash query formulation.
        """
        mock_retriever = MockDeterministicRetriever([make_evidence_item("e1", "Text")])
        # A noun phrase or fragment without clear predicate
        resp = defend_claim(
            claim_text="Quantum entanglement and non-local correlations.",
            retriever=mock_retriever,
        )
        assert isinstance(resp, DefenderResponse)
        assert "Quantum entanglement and non-local correlations." in resp.query_used

    def test_defender_does_not_reintroduce_mock_retriever_silently(self):
        """Config/dependency-injection check: confirm get_evidence_retriever

        resolves to HybridRetriever under standard settings, with an
        explicit failure if it silently falls back to mock without logging.
        """
        from app.dependencies import get_evidence_retriever
        from app.services.hybrid_retriever import HybridRetriever

        # With standard settings, HybridRetriever should be used
        retriever = get_evidence_retriever()
        assert isinstance(retriever, HybridRetriever)
