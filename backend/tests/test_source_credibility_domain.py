"""Unit tests for domain normalization and authority tier classification."""

import pytest

from app.services.source_credibility_scorer import SourceCredibilityScorer


@pytest.fixture
def scorer() -> SourceCredibilityScorer:
    """Fixture providing a default SourceCredibilityScorer instance."""
    return SourceCredibilityScorer()


# ---------------------------------------------------------------------------
# Section 8.2 Table Tests (Explicit test per table row)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_input,expected_domain,expected_tier",
    [
        (None, "unknown", 4),
        ("", "unknown", 4),
        (" ", "unknown", 4),
        ("not a url", "unknown", 4),
        ("http://", "unknown", 4),
        ("https://EXAMPLE.COM/path?x=1#f", "example.com", 4),
        ("example.com", "example.com", 4),
        ("https://www.example.com:8080/a", "example.com", 4),
        ("127.0.0.1", "127.0.0.1", 4),
        ("::1", "::1", 4),
    ],
)
def test_section_8_2_decision_table(
    scorer: SourceCredibilityScorer,
    raw_input: str | None,
    expected_domain: str,
    expected_tier: int,
):
    """Verify exact domain normalization and tier assignment for all Section 8.2 cases."""
    normalized = scorer.normalize_domain(raw_input)
    assert normalized == expected_domain
    tier = scorer.classify_tier(normalized)
    assert tier == expected_tier


# ---------------------------------------------------------------------------
# Domain Normalization Grammar Tests
# ---------------------------------------------------------------------------

def test_domain_normalization_schemes(scorer: SourceCredibilityScorer):
    assert scorer.normalize_domain("http://nature.com") == "nature.com"
    assert scorer.normalize_domain("https://nature.com") == "nature.com"
    assert scorer.normalize_domain("ftp://ftp.nature.com") == "ftp.nature.com"


def test_domain_normalization_www_prefix(scorer: SourceCredibilityScorer):
    # Stripped once
    assert scorer.normalize_domain("www.nature.com") == "nature.com"
    assert scorer.normalize_domain("https://www.nature.com") == "nature.com"
    # Not repeatedly stripped
    assert scorer.normalize_domain("www.www.nature.com") == "www.nature.com"


def test_domain_normalization_path_query_fragment(scorer: SourceCredibilityScorer):
    assert (
        scorer.normalize_domain("https://nature.com/articles/s41586-020-2649-2?search=1#figure1")
        == "nature.com"
    )


def test_domain_normalization_case_insensitivity(scorer: SourceCredibilityScorer):
    assert scorer.normalize_domain("HTTPS://NATURE.COM/INDEX.HTML") == "nature.com"
    assert scorer.normalize_domain("wWw.SciEnce.Org") == "science.org"


def test_domain_normalization_port_stripping(scorer: SourceCredibilityScorer):
    assert scorer.normalize_domain("https://nature.com:8443/article") == "nature.com"
    assert scorer.normalize_domain("reuters.com:80") == "reuters.com"


def test_domain_normalization_localhost(scorer: SourceCredibilityScorer):
    assert scorer.normalize_domain("localhost") == "localhost"
    assert scorer.normalize_domain("http://localhost:8000/api") == "localhost"


# ---------------------------------------------------------------------------
# Tier Classification Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "domain,expected_tier,expected_score",
    [
        # Tier 1 (1.00)
        ("nih.gov", 1, 1.00),
        ("subdomain.nasa.gov", 1, 1.00),
        ("mit.edu", 1, 1.00),
        ("cs.stanford.edu", 1, 1.00),
        ("who.int", 1, 1.00),
        ("army.mil", 1, 1.00),
        ("cam.ac.uk", 1, 1.00),
        ("service.gov.uk", 1, 1.00),
        # Tier 2 (0.75)
        ("wikipedia.org", 2, 0.75),
        ("en.wikipedia.org", 2, 0.75),
        ("nature.com", 2, 0.75),
        ("science.org", 2, 0.75),
        ("reuters.com", 2, 0.75),
        ("apnews.com", 2, 0.75),
        ("arxiv.org", 2, 0.75),
        ("britannica.com", 2, 0.75),
        # Tier 3 (0.50)
        ("nytimes.com", 3, 0.50),
        ("theguardian.com", 3, 0.50),
        ("forbes.com", 3, 0.50),
        ("bloomberg.com", 3, 0.50),
        ("medium.com", 3, 0.50),
        ("github.com", 3, 0.50),
        # Tier 4 (0.25)
        ("unregistered-blog-123.xyz", 4, 0.25),
        ("randomsite.net", 4, 0.25),
        ("test.invalid", 4, 0.25),
        ("my-local.test", 4, 0.25),
        ("mock.example", 4, 0.25),
        ("192.168.1.1", 4, 0.25),
        ("localhost", 4, 0.25),
        ("unknown", 4, 0.25),
    ],
)
def test_tier_classification_and_scores(
    scorer: SourceCredibilityScorer,
    domain: str,
    expected_tier: int,
    expected_score: float,
):
    tier = scorer.classify_tier(domain)
    assert tier == expected_tier
    assert scorer.settings.tier_scores[tier] == expected_score
