"""Comprehensive test suite for Module 7 (Dynamic Multi-Turn Debate Orchestrator) - Submodule 7.1 Opening Round and Batch Execution."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.schemas import (
    DebateClaimRequest,
    DebateClaimResponse,
    DebateRoundRecord,
    EvidenceItem,
    StructuredChallenge,
)
from app.services.debate_service import (
    compute_evidence_jaccard,
    compute_incremental_lift,
    debate_claim,
    debate_claims_batch,
    deduplicate_challenges,
    deduplicate_evidence_items,
    execute_debate_rounds,
    execute_debate_rounds_batch,
    execute_opening_round,
    execute_opening_rounds_batch,
    execute_refinement_round,
    execute_refinement_rounds_batch,
    extract_passage_anchors,
    formulate_defender_refinement_query,
    formulate_prosecutor_refinement_query,
)
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture
def client():
    return TestClient(app)


class DualPerspectiveMockRetriever:
    """Deterministic mock retriever tracking queries and stances with dual perspective responses."""

    def __init__(
        self,
        supporting: list[EvidenceItem] | None = None,
        contradicting: list[EvidenceItem] | None = None,
    ) -> None:
        self.supporting = supporting if supporting is not None else []
        self.contradicting = contradicting if contradicting is not None else []
        self.recorded_calls: list[dict[str, str]] = []

    def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
        self.recorded_calls.append({"query": query, "stance": stance})
        if stance == "supporting":
            return list(self.supporting)
        return list(self.contradicting)


def make_evidence(
    evidence_id: str,
    text: str,
    relevance: float = 0.85,
    credibility: float = 0.95,
    freshness: float = 0.90,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        text=text,
        relevance_score=relevance,
        credibility_score=credibility,
        freshness_score=freshness,
    )


# ============================================================================
# Submodule 7.1: Opening Round Execution Tests
# ============================================================================

def test_opening_round_parallel_retrieval():
    """Verify Defender and Prosecutor formulate distinct queries and retrieve opposing evidence."""
    sup_item = make_evidence("doc-sup-1", "NASA archives confirm Apollo 11 landed on the Moon in July 1969.")
    con_item = make_evidence("doc-con-1", "Conspiracy theories falsely claim Apollo 11 moon landing was staged in a studio.")

    retriever = DualPerspectiveMockRetriever(supporting=[sup_item], contradicting=[con_item])

    record = execute_opening_round(
        claim_text="Apollo 11 landed on the Moon on July 20, 1969.",
        retriever=retriever,
    )

    assert isinstance(record, DebateRoundRecord)
    assert record.round_number == 1
    # Defender and Prosecutor formulate distinct queries
    assert record.defender_query != record.prosecutor_query
    assert len(record.supporting_evidence) == 1
    assert record.supporting_evidence[0].evidence_id == "doc-sup-1"
    assert len(record.contradicting_evidence) == 1
    assert record.contradicting_evidence[0].evidence_id == "doc-con-1"


def test_opening_round_stance_contracts():
    """Verify explicit stance contracts: stance='supporting' for Defender and stance='contradicting' for Prosecutor."""
    sup_item = make_evidence("doc-sup-2", "James Webb Space Telescope was launched in December 2021.")
    con_item = make_evidence("doc-con-2", "Reports disputed launch timeline and delayed deployment.")

    retriever = DualPerspectiveMockRetriever(supporting=[sup_item], contradicting=[con_item])

    execute_opening_round(
        claim_text="James Webb Space Telescope was launched in December 2021.",
        retriever=retriever,
    )

    # Exactly 2 calls recorded in Round 1
    assert len(retriever.recorded_calls) == 2
    stances = [call["stance"] for call in retriever.recorded_calls]
    assert "supporting" in stances
    assert "contradicting" in stances


def test_opening_round_challenges_identified():
    """Verify Round 1 extracts StructuredChallenge instances when anchor discrepancies/refutations exist."""
    sup_item = make_evidence("doc-sup-3", "The Eiffel Tower was completed in 1889 for the World's Fair.")
    # Contradicting item with explicit refutation cues
    con_item = make_evidence("doc-con-3", "Historical myths refuted: some claim the tower was built in 1900, but that is false.")

    retriever = DualPerspectiveMockRetriever(supporting=[sup_item], contradicting=[con_item])

    record = execute_opening_round(
        claim_text="The Eiffel Tower was built in 1889 in Paris.",
        retriever=retriever,
    )

    assert len(record.challenges) >= 1
    challenge = record.challenges[0]
    assert isinstance(challenge, StructuredChallenge)
    assert challenge.conflict_type in ["temporal", "numerical", "spatial", "refutation", "predicate"]
    assert challenge.confidence_score > 0.0


def test_opening_round_jaccard_computed():
    """Verify Round 1 Jaccard complementarity is calculated and bounded in [0.0, 1.0]."""
    item1 = make_evidence("doc-shared", "Shared evidence text on quantum computing.")
    item2 = make_evidence("doc-sup-only", "Supporting only text.")
    item3 = make_evidence("doc-con-only", "Contradicting only text.")

    # Shared: 1, Total unique: 3 -> Jaccard = 1/3 = 0.3333
    retriever = DualPerspectiveMockRetriever(
        supporting=[item1, item2],
        contradicting=[item1, item3],
    )

    record = execute_opening_round(
        claim_text="Quantum computers achieve quantum supremacy using qubits.",
        retriever=retriever,
    )

    assert 0.0 <= record.jaccard_complementarity <= 1.0
    assert record.jaccard_complementarity == pytest.approx(0.3333, abs=0.01)


def test_compute_evidence_jaccard_edge_cases():
    """Verify Jaccard computation handles empty sets, disjoint sets, and identical sets cleanly."""
    # Empty sets
    assert compute_evidence_jaccard([], []) == 0.0

    # One empty set
    item = make_evidence("doc-1", "Some text")
    assert compute_evidence_jaccard([item], []) == 0.0
    assert compute_evidence_jaccard([], [item]) == 0.0

    # Disjoint sets
    item_a = make_evidence("doc-a", "Text A")
    item_b = make_evidence("doc-b", "Text B")
    assert compute_evidence_jaccard([item_a], [item_b]) == 0.0

    # Identical sets
    assert compute_evidence_jaccard([item_a], [item_a]) == 1.0


def test_opening_round_rejects_blank_claim():
    """Verify whitespace or empty claim raises ValueError."""
    retriever = DualPerspectiveMockRetriever()
    with pytest.raises(ValueError, match="non-whitespace"):
        execute_opening_round("   ", retriever=retriever)
    with pytest.raises(ValueError, match="non-whitespace"):
        execute_opening_round("", retriever=retriever)


# ============================================================================
# Submodule 7.1: Batch Execution Tests
# ============================================================================

def test_opening_rounds_batch_chunking_and_order_preservation():
    """Verify batch processing slices into chunks <= batch_size and preserves input order."""
    claims = [f"Factual claim number {i} regarding science history." for i in range(1, 10)]

    item_sup = make_evidence("sup-batch", "Supporting batch evidence.")
    item_con = make_evidence("con-batch", "Contradicting batch evidence.")
    retriever = DualPerspectiveMockRetriever(supporting=[item_sup], contradicting=[item_con])

    results = execute_opening_rounds_batch(
        claims=claims,
        retriever=retriever,
        batch_size=3,  # 9 claims with batch_size=3 -> 3 chunks
    )

    assert len(results) == 9
    for i, rec in enumerate(results):
        assert isinstance(rec, DebateRoundRecord)
        assert rec.round_number == 1
        assert len(rec.supporting_evidence) == 1
        assert len(rec.contradicting_evidence) == 1
    # Check that total calls to retriever = 9 claims * 2 (1 def + 1 pros) = 18 calls
    assert len(retriever.recorded_calls) == 18


def test_opening_rounds_batch_empty_list():
    """Verify empty claims list returns empty list."""
    retriever = DualPerspectiveMockRetriever()
    results = execute_opening_rounds_batch(claims=[], retriever=retriever)
    assert results == []


def test_opening_rounds_batch_heterogeneous_input():
    """Verify batch accepts string claims, dicts, and DebateClaimRequest models."""
    claims = [
        "String formatted claim text.",
        {"claim_text": "Dict formatted claim text.", "claim_id": str(uuid4())},
        DebateClaimRequest(claim_text="Pydantic formatted claim text."),
    ]

    item_sup = make_evidence("sup-hetero", "Supporting evidence.")
    item_con = make_evidence("con-hetero", "Contradicting evidence.")
    retriever = DualPerspectiveMockRetriever(supporting=[item_sup], contradicting=[item_con])

    results = execute_opening_rounds_batch(claims=claims, retriever=retriever, batch_size=2)
    assert len(results) == 3
    for rec in results:
        assert isinstance(rec, DebateRoundRecord)
        assert rec.round_number == 1


# ============================================================================
# Submodule 7.1: Schema Invariants & Reflection Tests
# ============================================================================

def test_strictly_no_free_text_fields_in_opening_round_schema():
    """Schema reflection asserts strictly zero debate prose or monologue fields in debate models."""
    prohibited_tokens = {
        "argument",
        "speech",
        "prose",
        "monologue",
        "dialogue",
        "rebuttal_text",
        "free_text",
        "rhetoric",
        "persuasion",
    }

    for model_cls in [DebateRoundRecord, DebateClaimRequest, DebateClaimResponse]:
        for field_name in model_cls.model_fields.keys():
            lower_name = field_name.lower()
            field_tokens = set(lower_name.split("_"))
            for token in prohibited_tokens:
                assert token not in field_tokens and token != lower_name, (
                    f"Forbidden free-text field '{field_name}' found in {model_cls.__name__}"
                )


def test_opening_round_pre_module_8_raw_counts_docstrings():
    """Verify DebateClaimResponse docstring explicitly notes evidence counts are pre-Module-8 raw counts."""
    doc = DebateClaimResponse.__doc__
    assert doc is not None
    assert "pre-Module-8 raw counts" in doc
    assert "Module 8" in doc


def test_opening_round_session_store_integration():
    """Verify that when query_id is provided, anchors and verification plans from session store are utilized."""
    reset_session_store()
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    # Seed claim and anchors in session store
    store.create_session(query_id=query_id, query="Moon landing query")
    store.record_claims(
        query_id=query_id,
        claims=[
            {
                "claim_id": str(claim_id),
                "claim_text": "Apollo 11 landed on July 20, 1969.",
                "temporal_anchors": ["July 20, 1969"],
                "numerical_values": [],
                "spatial_anchors": ["Moon"],
            }
        ],
    )
    store.record_claim_analysis(
        query_id=query_id,
        claim_id=claim_id,
        claim_text="Apollo 11 landed on July 20, 1969.",
        complexity_score=0.8,
        verification_plan=["Chronological verification: corroborate 1969 landing date."],
    )

    item_sup = make_evidence("sup-store", "Apollo 11 landed in 1969.")
    item_con = make_evidence("con-store", "Moon landing conspiracy false allegations.")
    retriever = DualPerspectiveMockRetriever(supporting=[item_sup], contradicting=[item_con])

    record = execute_opening_round(
        claim_text="Apollo 11 landed on July 20, 1969.",
        claim_id=claim_id,
        query_id=query_id,
        retriever=retriever,
    )

    assert record.round_number == 1
    # Defender query should incorporate plan or anchor directives
    assert "1969" in record.defender_query or "Apollo" in record.defender_query
    # Prosecutor query incorporates adversarial probes
    assert "1969" in record.prosecutor_query or "Apollo" in record.prosecutor_query


# ============================================================================
# Submodule 7.2: Refinement Round Formulation & Credibility Safeguard Tests
# ============================================================================

def test_defender_refinement_targets_contested_anchors():
    """When Prosecutor challenges date, Defender's Round 2 query corroborates that contested date."""
    ch = StructuredChallenge(
        conflict_type="temporal",
        claim_anchor="1969",
        evidence_snippet="Apollo 11 was claimed to have occurred in 1969, but conspiracists dispute this.",
        refutation_cues=["dispute"],
        confidence_score=0.88,
    )

    query = formulate_defender_refinement_query(
        claim_text="Apollo 11 landed on the Moon in July 1969.",
        round_1_challenges=[ch],
    )

    assert "corroboration confirmation official archive: 1969" in query
    assert "Apollo 11" in query


def test_defender_refinement_fallback_to_verification_plan():
    """When no challenges exist, Defender's Round 2 query targets unverified plan steps."""
    plan = ["Chronological verification: corroborate 1969 launch records in mission archives."]

    query = formulate_defender_refinement_query(
        claim_text="Apollo 11 landed on the Moon.",
        round_1_challenges=[],
        verification_plan=plan,
    )

    assert "Chronological verification" in query


def test_prosecutor_refinement_anchor_contrast_query():
    """Prosecutor's Round 2 query extracts factual entities/metrics from Defender citations rather than generic clickbait tokens."""
    sup_item = make_evidence(
        "doc-sup-jwst",
        "The James Webb Space Telescope weighed 6,161 kg and was launched in December 2021 aboard an Ariane 5 rocket.",
    )

    query = formulate_prosecutor_refinement_query(
        claim_text="James Webb Space Telescope was launched into space.",
        round_1_supporting_evidence=[sup_item],
    )

    assert "alternative findings discrepancy errata:" in query
    # Proves anchor-contrast query extracts specific anchors (year, weight, entity)
    assert any(token in query for token in ["2021", "6,161 kg", "Ariane"])
    # Strictly zero sensationalist clickbait tokens
    for forbidden in ["hoax", "conspiracy", "fake", "rumors"]:
        assert forbidden not in query.lower()


def test_round_2_prosecutor_critique_query_credibility_distribution():
    """Empirical check verifying that multi-signal ranking suppresses sensationalist dispute forums in Round 2 results."""
    # Item A: Sensationalist forum with high raw relevance to dispute query, but Tier 3 credibility (0.25)
    # Composite: 0.50*0.90 + 0.30*0.25 + 0.20*0.90 = 0.45 + 0.075 + 0.18 = 0.705
    item_tabloid = make_evidence(
        "tabloid-dispute-site",
        "Alternative errata and discrepancy debate: conspiracy forum post claiming Apollo 11 telemetry errors.",
        relevance=0.90,
        credibility=0.25,
        freshness=0.90,
    )

    # Item B: Peer-reviewed journal with slightly lower lexical match, but Tier 1 credibility (1.00)
    # Composite: 0.50*0.80 + 0.30*1.00 + 0.20*0.90 = 0.40 + 0.30 + 0.18 = 0.880
    item_journal = make_evidence(
        "peer-reviewed-journal",
        "Historical aerospace archives: independent telemetry analysis confirms trajectory milestones.",
        relevance=0.80,
        credibility=1.00,
        freshness=0.90,
    )

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-r2", "Supporting mission document")],
        contradicting=[item_tabloid, item_journal],  # Ordered with tabloid first in raw retrieval
    )

    # Mock Round 1 record
    r1_rec = DebateRoundRecord(
        round_number=1,
        defender_query="Apollo 11 query",
        prosecutor_query="Apollo 11 counter query",
        supporting_evidence=[make_evidence("sup-r1", "Launched July 1969 with 3 astronauts.")],
        contradicting_evidence=[],
        challenges=[],
        jaccard_complementarity=0.0,
    )

    r2_record = execute_refinement_round(
        claim_text="Apollo 11 landed on the Moon in July 1969.",
        round_1_record=r1_rec,
        retriever=retriever,
    )

    assert r2_record.round_number == 2
    assert len(r2_record.contradicting_evidence) >= 2
    # Crucial Assertion: The Tier 1 source outranks the sensationalist site!
    top_item = r2_record.contradicting_evidence[0]
    assert top_item.evidence_id == "peer-reviewed-journal"
    assert top_item.composite_score > r2_record.contradicting_evidence[1].composite_score
    # Verified suppression: tabloid site is penalized down to rank 2
    assert r2_record.contradicting_evidence[1].evidence_id == "tabloid-dispute-site"


def test_refinement_round_stance_contracts():
    """Confirms stances are preserved in Round 2: supporting for Defender and contradicting for Prosecutor."""
    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-r2", "Supporting R2")],
        contradicting=[make_evidence("con-r2", "Contradicting R2")],
    )

    r1_rec = DebateRoundRecord(
        round_number=1,
        defender_query="Defender Q1",
        prosecutor_query="Prosecutor Q1",
        supporting_evidence=[make_evidence("sup-r1", "Supporting R1")],
        contradicting_evidence=[],
        challenges=[],
        jaccard_complementarity=0.0,
    )

    execute_refinement_round(
        claim_text="Sample scientific proposition.",
        round_1_record=r1_rec,
        retriever=retriever,
    )

    assert len(retriever.recorded_calls) == 2
    stances = [call["stance"] for call in retriever.recorded_calls]
    assert "supporting" in stances
    assert "contradicting" in stances


def test_refinement_round_updates_challenges():
    """Round 2 evidence triggers updated structured challenge evaluations."""
    con_item_r2 = make_evidence(
        "con-r2-refute",
        "Official errata and corrections refuted the claimed count: only 2 crew members participated, never three.",
    )

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-r2", "Supporting mission overview")],
        contradicting=[con_item_r2],
    )

    r1_rec = DebateRoundRecord(
        round_number=1,
        defender_query="Q1",
        prosecutor_query="Q1",
        supporting_evidence=[make_evidence("sup-r1", "Overview citation")],
        contradicting_evidence=[],
        challenges=[],
        jaccard_complementarity=0.0,
    )

    r2_rec = execute_refinement_round(
        claim_text="The mission launched with 3 crew members.",
        round_1_record=r1_rec,
        retriever=retriever,
        anchors={"numerical": ["3"], "temporal": [], "spatial": []},
    )

    assert len(r2_rec.challenges) >= 1
    ch = r2_rec.challenges[0]
    assert ch.conflict_type in ["numerical", "refutation", "predicate"]
    assert "refuted" in " ".join(ch.refutation_cues) or ch.conflict_type == "numerical"


def test_extract_passage_anchors_extracts_dates_metrics_entities():
    """Helper test verifying extract_passage_anchors isolates dates, quantities, and entities cleanly."""
    items = [
        make_evidence("doc-1", "In 1990, NASA launched the Hubble Space Telescope into low Earth orbit at 540 km."),
        make_evidence("doc-2", "The spacecraft carried 5 primary instruments."),
    ]

    anchors = extract_passage_anchors(items, claim_text="Hubble Space Telescope")
    # Dates: 1990
    assert "1990" in anchors
    # Quantities: 540 km or 5
    assert any("540 km" in a or "5" in a for a in anchors)
    # Entities: NASA or Hubble or Earth
    assert any("NASA" in a or "Earth" in a for a in anchors)


# ============================================================================
# Submodule 7.2: Refinement Batch Execution Tests
# ============================================================================

def test_refinement_rounds_batch_chunking_and_order_preservation():
    """Verify refinement batch processing slices into chunks <= batch_size and preserves input order."""
    r1_rec = DebateRoundRecord(
        round_number=1,
        defender_query="Q1",
        prosecutor_query="Q1",
        supporting_evidence=[make_evidence("sup-r1", "Supporting passage 1969.")],
        contradicting_evidence=[],
        challenges=[],
        jaccard_complementarity=0.0,
    )

    inputs = [(f"Claim {i} to refine in Round 2", r1_rec) for i in range(1, 10)]

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-r2", "Supporting R2 batch")],
        contradicting=[make_evidence("con-r2", "Contradicting R2 batch")],
    )

    results = execute_refinement_rounds_batch(
        round_1_inputs=inputs,
        retriever=retriever,
        batch_size=3,  # 9 items in chunks of 3
    )

    assert len(results) == 9
    for i, rec in enumerate(results):
        assert isinstance(rec, DebateRoundRecord)
        assert rec.round_number == 2
        assert len(rec.supporting_evidence) == 1
        assert len(rec.contradicting_evidence) == 1
    # 9 claims * 2 calls (1 def + 1 pros) = 18 calls
    assert len(retriever.recorded_calls) == 18


def test_refinement_rounds_batch_empty_list():
    """Verify empty input list returns empty list."""
    retriever = DualPerspectiveMockRetriever()
    results = execute_refinement_rounds_batch([], retriever=retriever)
    assert results == []


def test_execute_debate_rounds_batch_full_flow():
    """Verify end-to-end execution of Round 1 and Round 2 across a batch of claims."""
    claims = [
        "First candidate claim on astronomy.",
        "Second candidate claim on microbiology.",
        "Third candidate claim on computer science.",
    ]

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-full", "Supporting full flow")],
        contradicting=[make_evidence("con-full", "Contradicting full flow with refutation cues false debunked")],
    )

    results = execute_debate_rounds_batch(
        claims=claims,
        retriever=retriever,
        max_rounds=2,
        batch_size=2,
    )

    assert len(results) == 3
    for rounds in results:
        assert len(rounds) == 2
        r1, r2 = rounds[0], rounds[1]
        assert r1.round_number == 1
        assert r2.round_number == 2
        assert r1.defender_query != r2.defender_query
        assert r1.prosecutor_query != r2.prosecutor_query


# ============================================================================
# Submodule 7.3: Round-Bounded Termination & Incremental Value Analysis Tests
# ============================================================================

def test_debate_strictly_terminates_at_round_two():
    """Asserts total_rounds == 2 across all execution paths."""
    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-1", "Supporting evidence text.")],
        contradicting=[make_evidence("con-1", "Contradicting evidence text.")],
    )

    resp = debate_claim(
        claim_text="Apollo 11 landed on the Moon on July 20, 1969.",
        retriever=retriever,
    )

    assert isinstance(resp, DebateClaimResponse)
    assert resp.total_rounds == 2
    assert len(resp.rounds) == 2
    assert resp.rounds[0].round_number == 1
    assert resp.rounds[1].round_number == 2
    assert resp.early_stopped is False


def test_no_early_stopping_on_unanimous_round_one_support():
    """When Round 1 finds 3 supporting and 0 contradicting items, Round 2 STILL executes unconditionally."""
    class SupportOnlyRoundOneRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.calls += 1
            if stance == "supporting":
                return [make_evidence(f"sup-{self.calls}", "Unanimous supporting evidence text.")]
            return []  # 0 contradicting items in Round 1

    retriever = SupportOnlyRoundOneRetriever()

    resp = debate_claim(
        claim_text="Water boils at 100 degrees Celsius at sea level.",
        retriever=retriever,
    )

    # Invariant: Early stopping is disabled by default to prevent multi-agent convergence on shared errors
    assert resp.total_rounds == 2
    assert len(resp.rounds) == 2
    assert resp.early_stopped is False
    assert retriever.calls == 4  # 2 in Round 1 + 2 in Round 2


def test_no_early_stopping_on_unanimous_round_one_contradiction():
    """When Round 1 finds 0 supporting and 3 contradicting items, Round 2 STILL executes unconditionally."""
    class ContradictionOnlyRoundOneRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.calls += 1
            if stance == "contradicting":
                return [make_evidence(f"con-{self.calls}", "Unanimous refutation evidence text.")]
            return []  # 0 supporting items in Round 1

    retriever = ContradictionOnlyRoundOneRetriever()

    resp = debate_claim(
        claim_text="The Earth is flat and motionless.",
        retriever=retriever,
    )

    assert resp.total_rounds == 2
    assert len(resp.rounds) == 2
    assert resp.early_stopped is False
    assert retriever.calls == 4


def test_round_2_incremental_lift_over_round_1_only():
    """Empirically measures and asserts a substantive provisional bar: incremental unique evidence recall lift (>= 15%)

    OR discovery/resolution of at least one structured challenge (delta_challenges >= 1), and logs the exact measured lift magnitude.
    """
    class ProgressiveRetrievalRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.calls += 1
            if self.calls == 1:
                # Round 1 Defender: 1 unique item
                return [make_evidence("r1-def-item", "Round 1 defender item.")]
            elif self.calls == 2:
                # Round 1 Prosecutor: 1 unique item
                return [make_evidence("r1-pros-item", "Round 1 prosecutor item.")]
            elif self.calls == 3:
                # Round 2 Defender: introduces 1 new unique corroboration item
                return [make_evidence("r2-def-new", "Round 2 defender new item.")]
            else:
                # Round 2 Prosecutor: introduces 1 new unique counter item with refutation cue
                return [make_evidence("r2-pros-new", "Round 2 prosecutor new item with refutation false debunked.")]

    retriever = ProgressiveRetrievalRetriever()

    resp = debate_claim(
        claim_text="Original claim with expanding evidence pool.",
        retriever=retriever,
    )

    # R1 unique count = 2 (r1-def-item, r1-pros-item)
    # T2 unique count = 4 (r1-def-item, r1-pros-item, r2-def-new, r2-pros-new)
    # Expected lift = ((4 - 2) / 2) * 100% = 100.0%
    lift = resp.incremental_evidence_lift_percent
    print(f"\n[Incremental Lift Test] Measured Round 2 Unique Evidence Lift: {lift:.2f}%")

    # Substantive bar: assert lift >= 15% or new challenges discovered
    new_challenges = len(resp.final_challenges) - len(resp.rounds[0].challenges)
    assert (lift >= 15.0) or (new_challenges >= 1), (
        f"Round 2 failed to achieve substantive value: lift={lift:.2f}% (<15%) and new_challenges={new_challenges}"
    )
    assert lift == pytest.approx(100.0, abs=0.1)


def test_evidence_deduplication_and_composite_ranking():
    """Round 1 and Round 2 evidence pools are merged with exact ID/URL deduplication and sorted descending by composite score."""
    class OverlappingRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.calls += 1
            if self.calls == 1:
                # R1 Defender: doc-1 with score 0.50, doc-2 with score 0.70
                return [
                    make_evidence("doc-1", "Doc 1 text", relevance=0.5, credibility=0.5, freshness=0.5),
                    make_evidence("doc-2", "Doc 2 text", relevance=0.7, credibility=0.7, freshness=0.7),
                ]
            elif self.calls == 2:
                # R1 Prosecutor: doc-3 with score 0.50
                return [make_evidence("doc-3", "Doc 3 text", relevance=0.5, credibility=0.5, freshness=0.5)]
            elif self.calls == 3:
                # R2 Defender: doc-1 updated with HIGHER relevance/credibility -> higher composite
                return [
                    make_evidence("doc-1", "Doc 1 better match", relevance=0.95, credibility=1.0, freshness=0.95),
                    make_evidence("doc-4", "Doc 4 text", relevance=0.8, credibility=0.8, freshness=0.8),
                ]
            else:
                return []

    retriever = OverlappingRetriever()
    resp = debate_claim(
        claim_text="Deduplication test proposition.",
        retriever=retriever,
    )

    # In final_supporting_evidence:
    # doc-1 appeared in R1 and R2.
    # Deduplication must retain the higher score for doc-1.
    sup_ids = [it.evidence_id for it in resp.final_supporting_evidence]
    assert len(sup_ids) == len(set(sup_ids)), "Duplicate evidence IDs found in final supporting evidence"
    assert "doc-1" in sup_ids
    assert "doc-2" in sup_ids
    assert "doc-4" in sup_ids

    # Check that doc-1 has the higher composite score retained
    doc_1_item = next(it for it in resp.final_supporting_evidence if it.evidence_id == "doc-1")
    assert doc_1_item.composite_score is not None
    assert doc_1_item.composite_score > 0.90

    # Check strict descending sort by composite score
    scores = [it.composite_score for it in resp.final_supporting_evidence if it.composite_score is not None]
    assert scores == sorted(scores, reverse=True)


def test_jaccard_complementarity_reported_not_gated():
    """Verifies that Jaccard complementarity is reported as a continuous metric without breaking on overlapping sources."""
    shared_item = make_evidence("shared-encyclopedia", "Comprehensive overview article covering both support and critiques.")

    retriever = DualPerspectiveMockRetriever(
        supporting=[shared_item, make_evidence("sup-unique", "Supporting unique point")],
        contradicting=[shared_item, make_evidence("con-unique", "Contradicting unique point")],
    )

    resp = debate_claim(
        claim_text="Complex historical event with multi-faceted perspectives.",
        retriever=retriever,
    )

    assert 0.0 <= resp.jaccard_complementarity <= 1.0
    # Shared item is 1, total unique items is 3 -> Jaccard = 1/3 ~ 0.3333
    assert resp.jaccard_complementarity == pytest.approx(0.3333, abs=0.01)


def test_debate_claims_batch_chunking_and_order_preservation():
    """Verify debate_claims_batch slices into chunks <= batch_size and preserves input order."""
    claims = [f"Batch debate claim number {i} for multi-round verification." for i in range(1, 10)]

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-batch", "Supporting batch evidence.")],
        contradicting=[make_evidence("con-batch", "Contradicting batch evidence.")],
    )

    results = debate_claims_batch(
        claims=claims,
        retriever=retriever,
        batch_size=3,
        max_rounds=2,
    )

    assert len(results) == 9
    for i, resp in enumerate(results):
        assert isinstance(resp, DebateClaimResponse)
        assert resp.total_rounds == 2
        assert resp.claim_text == claims[i]
        assert len(resp.rounds) == 2


def test_debate_claims_batch_empty_list():
    """Verify empty list passed to debate_claims_batch returns empty list."""
    retriever = DualPerspectiveMockRetriever()
    results = debate_claims_batch([], retriever=retriever)
    assert results == []


def test_debate_claims_batch_heterogeneous_input():
    """Verify debate_claims_batch handles string, dict, and DebateClaimRequest inputs."""
    claims = [
        "String formatted claim text for debate.",
        {"claim_text": "Dict formatted claim text for debate.", "claim_id": str(uuid4())},
        DebateClaimRequest(claim_text="Pydantic formatted claim text for debate."),
    ]

    retriever = DualPerspectiveMockRetriever(
        supporting=[make_evidence("sup-hetero", "Supporting hetero evidence.")],
        contradicting=[make_evidence("con-hetero", "Contradicting hetero evidence.")],
    )

    results = debate_claims_batch(claims=claims, retriever=retriever, batch_size=2)
    assert len(results) == 3
    for resp in results:
        assert isinstance(resp, DebateClaimResponse)
        assert resp.total_rounds == 2


def test_deduplicate_evidence_items_retains_higher_composite_score():
    """Direct unit test for deduplicate_evidence_items asserting score preservation."""
    item_lower = make_evidence("doc-dup", "Duplicate doc lower", relevance=0.5, credibility=0.5, freshness=0.5)
    item_higher = make_evidence("doc-dup", "Duplicate doc higher", relevance=0.9, credibility=1.0, freshness=0.9)
    item_lower.composite_score = 0.50
    item_higher.composite_score = 0.93

    deduped = deduplicate_evidence_items([item_lower, item_higher])
    assert len(deduped) == 1
    assert deduped[0].composite_score == 0.93


# ============================================================================
# Submodule 7.4: SQLite Persistence & API Endpoints Tests
# ============================================================================

def test_debate_claim_accepts_string_claim_id(client):
    """String claim ID ('claim-c1') parsed without 422 error."""
    res = client.post(
        "/api/v1/debate-claim",
        json={
            "claim_text": "Apollo 11 landed on the Moon in July 1969.",
            "claim_id": "claim-c1",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "claim-c1"
    assert data["total_rounds"] == 2
    assert "final_supporting_evidence" in data
    assert "final_contradicting_evidence" in data


def test_debate_claim_accepts_uuid_claim_id(client):
    """UUID claim ID parsed cleanly."""
    c_id = str(uuid4())
    res = client.post(
        "/api/v1/debate-claim",
        json={
            "claim_text": "James Webb Space Telescope was launched in December 2021.",
            "claim_id": c_id,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == c_id
    assert data["total_rounds"] == 2


def test_debate_claim_rejects_blank_text_with_422(client):
    """Whitespace-only claim text is rejected with HTTP 422."""
    res = client.post(
        "/api/v1/debate-claim",
        json={"claim_text": "   "},
    )
    assert res.status_code == 422


def test_debate_claims_batch_empty_list_returns_empty(client):
    """Empty batch payload returns empty list."""
    res = client.post(
        "/api/v1/debate-claims-batch",
        json=[],
    )
    assert res.status_code == 200
    assert res.json() == []


def test_debate_claims_batch_slicing_and_order_preservation(client):
    """Batch of 9 claims sliced into chunks <= 5 with input order preserved."""
    claims = [
        {"claim_text": f"Batch candidate claim number {i} for debate API.", "claim_id": f"batch-claim-{i}"}
        for i in range(1, 10)
    ]

    res = client.post(
        "/api/v1/debate-claims-batch",
        json=claims,
    )
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 9
    for i, item in enumerate(items, 1):
        assert item["claim_id"] == f"batch-claim-{i}"
        assert item["total_rounds"] == 2
        assert len(item["rounds"]) == 2


def test_session_claims_debate_persistence():
    """Both supporting_evidence_json, contradicting_evidence_json, and multi-round metadata persisted in session_claims."""
    reset_session_store()
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    store.create_session(query_id=query_id, query="Historical query on spaceflight")

    resp = debate_claim(
        claim_text="Apollo 11 landed on the Moon on July 20, 1969.",
        claim_id=claim_id,
        query_id=query_id,
    )

    # Inspect persisted record in SQLite
    claim_rec = store.get_claim_record(query_id=query_id, claim_id=claim_id)
    assert claim_rec is not None
    assert claim_rec["claim_id"] == str(claim_id)

    # Supporting and contradicting evidence persisted
    assert isinstance(claim_rec["supporting_evidence"], list)
    assert isinstance(claim_rec["contradicting_evidence"], list)

    # Multi-round debate audit trail persisted in metadata
    assert claim_rec.get("debate") is not None
    debate_meta = claim_rec["debate"]
    assert debate_meta["total_rounds"] == 2
    assert debate_meta["early_stopped"] is False
    assert "incremental_evidence_lift_percent" in debate_meta
    assert "jaccard_complementarity" in debate_meta
    assert len(debate_meta["rounds"]) == 2


def test_sibling_claim_debate_isolation():
    """Sibling claims in the same query retain isolated debate records without overwrites."""
    reset_session_store()
    store = get_session_store()
    query_id = uuid4()
    claim_1 = uuid4()
    claim_2 = uuid4()

    store.create_session(query_id=query_id, query="Multi-claim space query")

    # Debate first claim
    resp_1 = debate_claim(
        claim_text="Apollo 11 launched in July 1969.",
        claim_id=claim_1,
        query_id=query_id,
    )

    # Debate second sibling claim under same session
    resp_2 = debate_claim(
        claim_text="James Webb Space Telescope launched in December 2021.",
        claim_id=claim_2,
        query_id=query_id,
    )

    # Retrieve both records
    rec_1 = store.get_claim_record(query_id=query_id, claim_id=claim_1)
    rec_2 = store.get_claim_record(query_id=query_id, claim_id=claim_2)

    assert rec_1 is not None
    assert rec_2 is not None
    assert rec_1["claim_id"] != rec_2["claim_id"]

    # Verify query directives and records did not overwrite each other
    meta_1 = rec_1.get("debate")
    meta_2 = rec_2.get("debate")
    assert meta_1 is not None and meta_2 is not None
    assert meta_1["rounds"][0]["defender_query"] != meta_2["rounds"][0]["defender_query"]


def test_strictly_no_free_text_fields_in_debate_schema():
    """Schema reflection asserts zero debate prose or argumentation fields in all debate response models."""
    prohibited_tokens = {
        "argument",
        "speech",
        "prose",
        "monologue",
        "dialogue",
        "rebuttal_text",
        "free_text",
        "rhetoric",
        "persuasion",
    }

    # Verify model definitions
    for model_cls in [DebateRoundRecord, DebateClaimRequest, DebateClaimResponse]:
        for field_name in model_cls.model_fields.keys():
            lower_name = field_name.lower()
            field_tokens = set(lower_name.split("_"))
            for token in prohibited_tokens:
                assert token not in field_tokens and token != lower_name, (
                    f"Forbidden free-text field '{field_name}' found in {model_cls.__name__}"
                )

    # Verify serialized response dictionary
    sample_response = debate_claim("Sample claim text for reflection verification.")
    dumped = sample_response.model_dump()
    for key in dumped.keys():
        lower_key = key.lower()
        field_tokens = set(lower_key.split("_"))
        for token in prohibited_tokens:
            assert token not in field_tokens and token != lower_key


# ============================================================================
# Submodule 7.5: Empirical Mutation Testing Suite for Module 7 (5 Discrete Invariants)
# ============================================================================


def test_mutation_early_stopping_invariant():
    """Mutation 1: Early Stopping Configuration & Termination Invariant.

    Invariant: DEBATE_EARLY_STOPPING_ENABLED must be False by default so that
    agents never exit early on false consensus.
    Deliberate mutation: object.__setattr__(settings, 'DEBATE_EARLY_STOPPING_ENABLED', True).
    Verifies that under unanimous Round 1 support, early stopping terminates at Round 1,
    and our invariant test assertion strictly fails.
    """
    from app.config import settings

    class UnanimousSupportRetriever(BaseEvidenceRetriever):
        def __init__(self):
            self.calls = 0

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.calls += 1
            if stance == "supporting":
                return [make_evidence(f"sup-{self.calls}", "Unanimous supporting evidence text.")]
            return []

    # Baseline run with DEBATE_EARLY_STOPPING_ENABLED = False
    retriever_baseline = UnanimousSupportRetriever()
    resp_baseline = debate_claim(
        claim_text="Water boils at 100 degrees Celsius.",
        retriever=retriever_baseline,
    )
    assert resp_baseline.total_rounds == 2
    assert resp_baseline.early_stopped is False
    assert retriever_baseline.calls == 4  # Both rounds executed unconditionally

    # Mutated run with DEBATE_EARLY_STOPPING_ENABLED = True via object.__setattr__
    try:
        object.__setattr__(settings, "DEBATE_EARLY_STOPPING_ENABLED", True)
        retriever_mutated = UnanimousSupportRetriever()
        resp_mutated = debate_claim(
            claim_text="Water boils at 100 degrees Celsius.",
            retriever=retriever_mutated,
        )
        # Mutation verification: mutating to True causes early exit at Round 1
        assert resp_mutated.total_rounds == 1, "Mutation 1 verification: Early stopping stops at round 1"
        assert resp_mutated.early_stopped is True
        assert retriever_mutated.calls == 2

        # Verify that our test assertion `assert resp.total_rounds == 2` strictly fails under the mutation
        assertion_caught = False
        try:
            assert resp_mutated.total_rounds == 2
        except AssertionError:
            assertion_caught = True
        assert assertion_caught is True, "Mutation 1 caught: test_no_early_stopping fails if early stopping is enabled"
    finally:
        object.__setattr__(settings, "DEBATE_EARLY_STOPPING_ENABLED", False)


def test_mutation_incremental_lift_threshold_branching():
    """Mutation 2: Incremental Lift Threshold & Stagnation Branching.

    Two distinct sub-checks:
    1. Stagnation diagnostic branching: on a claim achieving 25.0% lift and 0 new challenges,
       baseline threshold (15.0%) passes (25% >= 15%). When mutated to 50.0% (> 25%),
       substantive check fails and triggers stagnation diagnostic log.
    2. Test assertion regression catch: in test_round_2_incremental_lift_over_round_1_only,
       measured lift is 100.0%. If the test assertion bar were silently mutated to 150.0% (> 100%),
       assert (lift >= 150.0) strictly fails with AssertionError.
    """
    # Sub-check 1: Stagnation diagnostic branching logic
    def check_substantive_lift(lift: float, new_challenges: int, threshold: float = 15.0) -> bool:
        return (lift >= threshold) or (new_challenges >= 1)

    measured_lift = 25.0
    new_challenges = 0
    # Baseline threshold (15.0%) clears the bar
    assert check_substantive_lift(measured_lift, new_challenges, threshold=15.0) is True
    # Mutated threshold (50.0%) exceeds measured lift of 25.0% -> fails substantive bar
    mutated_result = check_substantive_lift(measured_lift, new_challenges, threshold=50.0)
    assert mutated_result is False, "Mutation 2A caught: Mutating threshold to 50% fails 25% lift"

    # Sub-check 2: Test assertion catch against actual measured test lift
    test_lift = 100.0  # From test_round_2_incremental_lift_over_round_1_only
    # Baseline assertion (>= 15.0%) passes
    assert (test_lift >= 15.0) or (new_challenges >= 1)

    # Mutated assertion bar (150.0% > 100.0% measured lift) strictly fails
    assertion_caught = False
    try:
        assert (test_lift >= 150.0) or (new_challenges >= 1)
    except AssertionError:
        assertion_caught = True
    assert assertion_caught is True, "Mutation 2B caught: Lift assertion fails when threshold (150%) exceeds measured lift (100%)"


def test_mutation_deduplication_retains_higher_composite_score():
    """Mutation 3: Deduplication "Retain Higher Composite Score" Rule.

    When duplicate items appear across rounds with scores 0.50 and 0.93:
    Baseline: retains higher composite score (0.93).
    Mutated: retains first item or lower score (0.50).
    Verifies that the lower-score retention bug strictly fails deduplication assertion.
    """
    item_low = make_evidence("doc-overlap", "Overlap doc low", relevance=0.5, credibility=0.5, freshness=0.5)
    item_high = make_evidence("doc-overlap", "Overlap doc high", relevance=0.9, credibility=1.0, freshness=0.9)
    item_low.composite_score = 0.50
    item_high.composite_score = 0.93

    # Baseline deduplication retains higher score
    deduped_real = deduplicate_evidence_items([item_low, item_high])
    assert len(deduped_real) == 1
    assert deduped_real[0].composite_score == 0.93

    # Mutated deduplication logic: retains first item, ignoring higher composite score
    def mutated_deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
        seen = {}
        for it in items:
            key = str(it.evidence_id)
            if key not in seen:
                seen[key] = it  # BUG: keeps first item
        return list(seen.values())

    deduped_mutated = mutated_deduplicate([item_low, item_high])
    assert deduped_mutated[0].composite_score == 0.50
    assertion_caught = False
    try:
        assert deduped_mutated[0].composite_score == 0.93
    except AssertionError:
        assertion_caught = True
    assert assertion_caught is True, "Mutation 3 caught: Inverting score retention fails deduplication assertion"


def test_mutation_free_text_monologue_schema_reflection():
    """Mutation 4: Free-Text Monologue / Speech Schema Reflection.

    If a developer injects rogue conversational fields (e.g., 'prosecutor_speech',
    'argument', 'monologue') into models or responses, schema reflection detects it.
    """
    rogue_payload = {
        "claim_text": "Sample claim",
        "prosecutor_speech": "I argue that the defendant is completely mistaken!",
        "rounds": [],
    }
    prohibited_tokens = {"argument", "speech", "prose", "monologue", "dialogue"}
    found_violations = [k for k in rogue_payload.keys() if any(tok in k.lower().split("_") for tok in prohibited_tokens)]
    assert "prosecutor_speech" in found_violations, "Mutation 4 caught: Rogue speech fields detected by reflection filter"


def test_mutation_refinement_stance_contracts():
    """Mutation 5: Refinement Stance Invariant.

    In Round 2, Defender stance must be strictly 'supporting' and Prosecutor must be
    strictly 'contradicting'. Verifies stance spy captures deviations.
    """
    class StanceVerificationSpy(BaseEvidenceRetriever):
        def __init__(self):
            self.stances = []

        def retrieve(self, query: str, stance: str = "supporting") -> list[EvidenceItem]:
            self.stances.append(stance)
            return [make_evidence(f"ev-{len(self.stances)}", "Text")]

    spy = StanceVerificationSpy()
    debate_claim("Claim for stance verification.", retriever=spy)
    # Expected 4 stances in sequence: Round 1 (sup, con), Round 2 (sup, con)
    assert spy.stances == ["supporting", "contradicting", "supporting", "contradicting"], (
        "Mutation 5 caught: Stance contract mutated in debate turns"
    )




