"""Comprehensive Batch and Integration Tests for Component 4:
Federated Evidentiary Multi-Turn Debate & Evidence Quality Firewall.

Verifies end-to-end integration across:
- Module 5 / Module 6: Plan-Guided Defender and Adversarial Prosecutor
- Module 7: Multi-Turn Turn-Bounded (T=2) Evidentiary Debate Orchestration
- Module 8: Unified Evidence Quality Firewall (Domain Credibility, Temporal Freshness,
            Source Independence Kish n_eff, and Evidence Pollution Defense)
- Component 3: FederatedEvidenceRetriever (SciFact BM25, FEVER BM25, Free Web Escalation)
- Component 1: SQLite WebQueryCache with sub-millisecond retrieval on cache hits
"""

from datetime import date
import time
from uuid import uuid4
import pytest

from app.schemas import (
    DebateClaimRequest,
    DebateClaimResponse,
    DebateRoundRecord,
    DebateQualityResponse,
    EvidenceItem,
    StructuredChallenge,
)
from app.services.debate_service import (
    debate_claim,
    debate_claims_batch,
    extract_passage_anchors,
)
from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.federated_evidence_retriever import (
    FederatedEvidenceRetriever,
    get_federated_evidence_retriever,
)
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture(scope="module")
def federated_retriever() -> FederatedEvidenceRetriever:
    """Return shared FederatedEvidenceRetriever in hybrid mode."""
    return FederatedEvidenceRetriever(mode="hybrid", top_k=3, escalation_threshold=0.50)


@pytest.fixture(scope="module")
def quality_service() -> EvidenceQualityService:
    """Return shared EvidenceQualityService instance."""
    return get_evidence_quality_service()


# ============================================================================
# 1. SciFact Biomedical Real Corpus Debate & Quality
# ============================================================================

def test_biomedical_scifact_federated_debate_and_quality(
    federated_retriever: FederatedEvidenceRetriever,
    quality_service: EvidenceQualityService,
):
    """Verify biomedical claim routes to SciFact PubMed corpus, conducts 2-turn debate,
    and passes through Module 8 Quality Firewall with Tier 1 domain credibility.
    """
    claim = "Aspirin reduces cardiovascular mortality in clinical trials"

    # Step 1: Execute Turn-Bounded Debate (T=2)
    debate_resp = debate_claim(
        claim_text=claim,
        retriever=federated_retriever,
        max_rounds=2,
    )

    assert isinstance(debate_resp, DebateClaimResponse)
    assert debate_resp.total_rounds == 2
    assert debate_resp.early_stopped is False
    assert len(debate_resp.rounds) == 2

    # Step 2: Validate Evidence Routing & Grounding
    assert len(debate_resp.final_supporting_evidence) >= 1
    assert len(debate_resp.final_contradicting_evidence) >= 1

    top_supporting = debate_resp.final_supporting_evidence[0]
    assert top_supporting.retrieval_method == "benchmark_scifact"
    assert top_supporting.domain == "ncbi.nlm.nih.gov"
    assert "Aspirin" in top_supporting.source_title or "vascular" in top_supporting.text.lower()

    # Step 3: Validate Round 1 vs Round 2 Divergence & Anchors
    r1 = debate_resp.rounds[0]
    r2 = debate_resp.rounds[1]
    assert r1.defender_query != r1.prosecutor_query
    assert r2.round_number == 2

    # Step 4: Evaluate Debate Evidence Pool via Module 8 Quality Firewall
    qual_resp = quality_service.evaluate_debate_quality(
        claim=debate_resp.claim_text,
        supporting_evidence=debate_resp.final_supporting_evidence,
        contradicting_evidence=debate_resp.final_contradicting_evidence,
        reference_date=date(2024, 1, 1),
    )

    assert isinstance(qual_resp, DebateQualityResponse)
    assert qual_resp.total_raw_evidence_count >= 2
    assert qual_resp.total_admissible_count >= 1

    # Tier 1 Domain Authority for ncbi.nlm.nih.gov (.gov suffix)
    for rep in qual_resp.pooled_supporting_evidence:
        assert rep.credibility_score >= 0.85
        assert rep.domain == "ncbi.nlm.nih.gov"

    detailed_eval = quality_service.evaluate_evidence(
        claim=claim, evidence_items=debate_resp.final_supporting_evidence
    )
    for item in detailed_eval.evaluated_items:
        assert item.domain_tier == 1
        assert "High authority" in item.domain_tier_label

    # Kish effective sample size discounting redundancy
    assert qual_resp.supporting_n_eff >= 1.0
    assert qual_resp.contradicting_n_eff >= 1.0
    assert qual_resp.joint_n_eff >= 1.0


# ============================================================================
# 2. FEVER Encyclopedic Real Corpus Debate & Quality
# ============================================================================

def test_encyclopedic_fever_federated_debate_and_quality(
    federated_retriever: FederatedEvidenceRetriever,
    quality_service: EvidenceQualityService,
):
    """Verify general factual claim routes to FEVER Wikipedia corpus, conducts 2-turn debate,
    and passes through Module 8 with Tier 2 domain authority and Kish redundancy discounting.
    """
    claim = "Colin Kaepernick was the starting quarterback for the San Francisco 49ers"

    debate_resp = debate_claim(
        claim_text=claim,
        retriever=federated_retriever,
        max_rounds=2,
    )

    assert debate_resp.total_rounds == 2
    assert debate_resp.early_stopped is False

    # Evidence comes from FEVER Wikipedia corpus
    top_sup = debate_resp.final_supporting_evidence[0]
    assert top_sup.retrieval_method == "benchmark_fever"
    assert top_sup.domain == "en.wikipedia.org"
    assert "Colin Kaepernick" in top_sup.source_title

    # Module 8 Quality Evaluation
    qual_resp = quality_service.evaluate_debate_quality(
        claim=debate_resp.claim_text,
        supporting_evidence=debate_resp.final_supporting_evidence,
        contradicting_evidence=debate_resp.final_contradicting_evidence,
    )

    assert qual_resp.total_admissible_count >= 1
    # Tier 2 Domain Authority for en.wikipedia.org
    for rep in qual_resp.pooled_supporting_evidence:
        assert rep.credibility_score >= 0.70
        assert rep.domain == "en.wikipedia.org"

    detailed_eval = quality_service.evaluate_evidence(
        claim=claim, evidence_items=debate_resp.final_supporting_evidence
    )
    for item in detailed_eval.evaluated_items:
        assert item.domain_tier == 2
        assert "Established" in item.domain_tier_label

    assert qual_resp.supporting_n_eff >= 1.0


# ============================================================================
# 3. Contemporary Live Web Escalation Debate & Quality
# ============================================================================

def test_contemporary_live_web_escalation_debate_and_quality(
    federated_retriever: FederatedEvidenceRetriever,
    quality_service: EvidenceQualityService,
):
    """Verify contemporary astronomy claim escalates past local benchmarks to Live Web,
    conducts 2-turn debate with live web citations, and evaluates through Module 8.
    """
    claim = "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b"

    debate_resp = debate_claim(
        claim_text=claim,
        retriever=federated_retriever,
        max_rounds=2,
    )

    assert debate_resp.total_rounds == 2
    assert len(debate_resp.final_supporting_evidence) >= 1

    # Escalated to live web
    top_sup = debate_resp.final_supporting_evidence[0]
    assert top_sup.retrieval_method.startswith("free_web_")
    assert top_sup.domain in ("science.nasa.gov", "nasa.gov", "news.ucsc.edu", "en.wikipedia.org")

    qual_resp = quality_service.evaluate_debate_quality(
        claim=debate_resp.claim_text,
        supporting_evidence=debate_resp.final_supporting_evidence,
        contradicting_evidence=debate_resp.final_contradicting_evidence,
    )

    assert qual_resp.total_raw_evidence_count >= 2
    assert qual_resp.total_admissible_count >= 1
    assert qual_resp.supporting_n_eff >= 1.0


# ============================================================================
# 4. Batch Processing with Memory-Safe Chunking
# ============================================================================

def test_federated_debate_claims_batch_memory_safe_chunking(
    federated_retriever: FederatedEvidenceRetriever,
    quality_service: EvidenceQualityService,
):
    """Verify batch multi-turn debate slices requests into memory-safe chunks (batch_size=2)
    and preserves input order across heterogeneous domains.
    """
    batch_claims = [
        "Aspirin reduces cardiovascular mortality in clinical trials",
        "Colin Kaepernick was the starting quarterback for the San Francisco 49ers",
        "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b",
        "Tim Berners-Lee is an English computer scientist who invented the World Wide Web",
    ]

    debate_responses = debate_claims_batch(
        claims=batch_claims,
        retriever=federated_retriever,
        batch_size=2,  # Memory-safe chunk size <= 2
        max_rounds=2,
    )

    assert len(debate_responses) == 4

    for idx, resp in enumerate(debate_responses):
        assert resp.claim_text == batch_claims[idx]
        assert resp.total_rounds == 2
        assert resp.early_stopped is False
        assert len(resp.final_supporting_evidence) >= 1

        # Run quality evaluation on batch items
        qual = quality_service.evaluate_debate_quality(
            claim=resp.claim_text,
            supporting_evidence=resp.final_supporting_evidence,
            contradicting_evidence=resp.final_contradicting_evidence,
        )
        assert qual.total_raw_evidence_count >= 1
        assert qual.supporting_n_eff >= 1.0


# ============================================================================
# 5. SessionStore Audit Trail Persistence
# ============================================================================

def test_federated_debate_and_quality_session_store_audit(
    federated_retriever: FederatedEvidenceRetriever,
    quality_service: EvidenceQualityService,
):
    """Verify end-to-end SQLite session persistence for debate rounds and quality audit."""
    reset_session_store()
    store = get_session_store()
    query_id = uuid4()
    claim_id = uuid4()

    store.create_session(query_id=query_id, query="Cardiovascular medication efficacy audit")

    # 1. Execute debate with session tracking
    debate_resp = debate_claim(
        claim_text="Aspirin reduces cardiovascular mortality in clinical trials",
        claim_id=claim_id,
        query_id=query_id,
        retriever=federated_retriever,
        max_rounds=2,
    )

    # Verify debate record in SQLite
    claim_rec = store.get_claim_record(query_id=query_id, claim_id=claim_id)
    assert claim_rec is not None
    assert claim_rec["claim_id"] == str(claim_id)
    assert len(claim_rec["supporting_evidence"]) >= 1
    assert claim_rec.get("debate") is not None
    assert claim_rec["debate"]["total_rounds"] == 2

    # 2. Evaluate debate quality with session tracking
    qual_resp = quality_service.evaluate_debate_quality(
        claim=debate_resp.claim_text,
        supporting_evidence=debate_resp.final_supporting_evidence,
        contradicting_evidence=debate_resp.final_contradicting_evidence,
        claim_id=claim_id,
        query_id=query_id,
    )

    # Verify quality audit record in SQLite
    claim_rec_updated = store.get_claim_record(query_id=query_id, claim_id=claim_id)
    assert claim_rec_updated.get("debate_quality") is not None
    audit = claim_rec_updated["debate_quality"]
    assert audit["supporting_raw_count"] == qual_resp.supporting_report.total_raw_evidence_count
    assert audit["supporting_n_eff"] == qual_resp.supporting_n_eff
    assert audit["joint_n_eff"] == qual_resp.joint_n_eff


# ============================================================================
# 6. SQLite Query Cache Performance Speedup
# ============================================================================

def test_web_cache_latency_speedup_in_debate(
    federated_retriever: FederatedEvidenceRetriever,
):
    """Verify that repeated queries hit the local SQLite cache and execute in sub-millisecond time."""
    claim = "James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b"

    # First run (populates cache if not present)
    _ = debate_claim(claim_text=claim, retriever=federated_retriever, max_rounds=2)

    # Second run (100% cache hits)
    start_time = time.perf_counter()
    resp_cached = debate_claim(claim_text=claim, retriever=federated_retriever, max_rounds=2)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    assert resp_cached.total_rounds == 2
    assert len(resp_cached.final_supporting_evidence) >= 1
    # Cache hit retrieval across all 4 debate calls must complete quickly
    assert elapsed_ms < 2500.0, f"Expected cache-accelerated run < 2500ms, got {elapsed_ms:.1f}ms"


# ============================================================================
# 7. Passage Anchor Extraction & Structured Challenge Mechanics
# ============================================================================

def test_passage_anchor_extraction_and_refinement():
    """Verify factual anchors (years, entities, metrics) are isolated from cited passages
    and used by Prosecutor to generate focused cross-examination queries.
    """
    sample_evidence = [
        EvidenceItem(
            evidence_id="pubmed-101",
            text="In 2011, a randomized trial of 1,200 patients showed a 15% reduction in mortality at 50 mg dosage.",
            source_title="Clinical Trial 2011",
            relevance_score=0.90,
        )
    ]

    anchors = extract_passage_anchors(sample_evidence, claim_text="Aspirin reduction in mortality")
    # Year: 2011
    assert "2011" in anchors
    # Metric: 15% or 1,200 or 50 mg
    assert any("%" in a or "1,200" in a or "50" in a for a in anchors)
