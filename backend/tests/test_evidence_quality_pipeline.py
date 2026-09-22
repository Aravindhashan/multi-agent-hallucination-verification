"""Tests for EvidenceQualityService and POST /api/v1/evidence/evaluate-quality endpoint."""

from datetime import date, timedelta
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.evidence_quality_service import (
    EvidenceQualityService,
    get_evidence_quality_service,
)
from app.services.session_store import get_session_store, reset_session_store


@pytest.fixture(autouse=True)
def clean_state():
    reset_session_store()
    yield
    reset_session_store()


@pytest.fixture
def client():
    return TestClient(app)


def test_evidence_quality_service_unit():
    """Verify that EvidenceQualityService orchestrates credibility, freshness, clustering, and pollution."""
    service = EvidenceQualityService()
    claim = "Apollo 11 was the first crewed mission to land on the Moon on July 20, 1969."
    today = date(2023, 6, 1)

    items = [
        # Item 1: High authority, fresh, clean
        {
            "evidence_id": "nasa_01",
            "text": "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
            "source_url": "https://www.nasa.gov/mission_pages/apollo/apollo11.html",
            "domain": "nasa.gov",
            "source_title": "NASA Historical Archives",
            "published_date": date(2023, 1, 15),
        },
        # Item 2: Duplicate of Item 1 (near-duplicate text from a mirror domain)
        {
            "evidence_id": "mirror_01",
            "text": "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
            "source_url": "https://mirror-news.com/apollo11.html",
            "domain": "mirror-news.com",
            "source_title": "Syndicated Mirror",
            "published_date": date(2023, 1, 15),
        },
        # Item 3: Older news passage
        {
            "evidence_id": "nyt_01",
            "text": "Men walk on Moon as Apollo 11 astronauts complete lunar landing mission.",
            "source_url": "https://www.nytimes.com/1969/07/21/archives.html",
            "domain": "nytimes.com",
            "source_title": "The New York Times",
            "published_date": date(2020, 7, 20),
        },
        # Item 4: Ungrounded / completely irrelevant passage
        {
            "evidence_id": "unrelated_01",
            "text": "The recipe requires two cups of flour, three eggs, and fresh vanilla extract.",
            "source_url": "https://cooking-blog.example.com/pancakes",
            "domain": "cooking-blog.example.com",
            "source_title": "Cooking Blog",
            "published_date": date(2023, 5, 1),
        },
    ]

    report = service.evaluate_evidence(
        claim=claim,
        evidence_items=items,
        reference_date=today,
    )

    assert report.total_evidence_count == 4
    assert len(report.evaluated_items) == 4

    # High authority NASA item
    nasa_rep = next(it for it in report.evaluated_items if it.evidence_id == "nasa_01")
    assert nasa_rep.credibility_score >= 0.80
    assert nasa_rep.is_admissible
    assert nasa_rep.pollution_score == 0.0

    # Redundancy cluster check: items 1 and 2 should share the same cluster
    mirror_rep = next(it for it in report.evaluated_items if it.evidence_id == "mirror_01")
    assert nasa_rep.cluster_id == mirror_rep.cluster_id
    assert nasa_rep.cluster_size == 2
    assert mirror_rep.cluster_size == 2

    # Kish effective count should be less than raw count of 4 due to duplication
    assert report.kish_effective_count < 4.0

    # Ungrounded item check: should be flagged with high pollution score
    unrelated_rep = next(it for it in report.evaluated_items if it.evidence_id == "unrelated_01")
    assert unrelated_rep.pollution_score >= 0.80
    assert not unrelated_rep.is_admissible
    assert "Rejected" in unrelated_rep.admissibility_reason


def test_evidence_quality_route_api(client):
    """Verify POST /api/v1/evidence/evaluate-quality endpoint."""
    claim = "The James Webb Space Telescope operates at Sun-Earth Lagrange Point 2."
    query_id = str(uuid4())

    payload = {
        "claim_text": claim,
        "query_id": query_id,
        "evidence_items": [
            {
                "evidence_id": "jwst_doc_1",
                "text": "The James Webb Space Telescope is located at the Sun-Earth L2 Lagrange point 1.5 million kilometers from Earth.",
                "source_url": "https://jwst.nasa.gov/content/about/orbit.html",
                "domain": "nasa.gov",
                "source_title": "NASA JWST Orbit Guide",
                "retrieval_method": "hybrid",
                "relevance_score": 0.95,
            },
            {
                "evidence_id": "jwst_doc_2",
                "text": "The James Webb Space Telescope operates in halo orbit around Sun-Earth Lagrange Point 2.",
                "source_url": "https://esa.int/Science_Exploration/Space_Science/Webb",
                "domain": "esa.int",
                "source_title": "ESA Space Science",
                "retrieval_method": "dense",
                "relevance_score": 0.88,
            },
        ],
    }

    # First register the session
    client.post("/api/v1/query", json={"query": "Where is JWST located?", "query_id": query_id})

    response = client.post("/api/v1/evidence/evaluate-quality", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["claim_text"] == claim
    assert data["query_id"] == query_id
    assert data["total_evidence_count"] == 2
    assert data["admissible_count"] == 2
    assert data["polluted_count"] == 0
    assert len(data["evaluated_items"]) == 2

    # Verify session store recorded the evaluation metadata
    session = get_session_store().get_session(uuid4().__class__(query_id))
    assert session is not None
    assert "evidence_quality_evaluation" in session.metadata
    assert session.metadata["evidence_quality_evaluation"]["total_evidence_count"] == 2


def test_evidence_quality_empty_items_rejected(client):
    """Verify that empty evidence list returns HTTP 422 Unprocessable Entity."""
    response = client.post(
        "/api/v1/evidence/evaluate-quality",
        json={"claim_text": "Some claim", "evidence_items": []},
    )
    assert response.status_code == 422
