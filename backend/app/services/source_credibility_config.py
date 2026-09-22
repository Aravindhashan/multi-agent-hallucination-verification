"""Configuration and default parameters for Source Credibility and Domain Authority Scoring."""

from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# Default Constants & Lookup Sets (Config-driven only)
# ---------------------------------------------------------------------------

SOURCE_CRED_W_DOMAIN: float = 0.50
SOURCE_CRED_W_TYPE: float = 0.20
SOURCE_CRED_W_PROVENANCE: float = 0.30

# Tier 1: High authority domains and official public sector/educational TLDs
SOURCE_CRED_TIER1_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".mil",
    ".edu",
    ".int",
    ".ac.uk",
    ".gov.uk",
)

# Tier 2: Established professional/scholarly orgs, major institutional domains, top encyclopedias
SOURCE_CRED_TIER2_DOMAINS: tuple[str, ...] = (
    "wikipedia.org",
    "nature.com",
    "science.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "un.org",
    "arxiv.org",
    "ieee.org",
    "acm.org",
    "britannica.com",
    "plos.org",
    "frontiersin.org",
    "springer.com",
    "wiley.com",
)

# Tier 3: General informational, commercial, news, blog platforms
SOURCE_CRED_TIER3_DOMAINS: tuple[str, ...] = (
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "forbes.com",
    "bloomberg.com",
    "economist.com",
    "medium.com",
    "substack.com",
    "cnn.com",
    "techcrunch.com",
    "theverge.com",
    "github.com",
    "gitlab.com",
    "stackoverflow.com",
)

# Reserved and placeholder domains per RFC 2606 / RFC 6761
SOURCE_CRED_PLACEHOLDER_SUFFIXES: tuple[str, ...] = (
    ".example",
    ".test",
    ".invalid",
    ".localhost",
)

# Schema-driven provenance fields evaluated for completeness
SOURCE_CRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "document_id",
    "title",
    "source",
    "source_url",
    "domain",
    "published_date",
)

# Source type scoring map (when source_type is present)
SOURCE_CRED_SOURCE_TYPE_MAP: dict[str, float] = {
    "government": 1.00,
    "academic": 1.00,
    "peer-reviewed": 1.00,
    "academic / peer-reviewed": 1.00,
    "institutional": 0.75,
    "professional": 0.75,
    "institutional / professional": 0.75,
    "technical": 0.65,
    "news": 0.55,
    "commercial": 0.40,
    "blog": 0.30,
    "unknown": 0.25,
}

SOURCE_CRED_TIER_SCORES: dict[int, float] = {
    1: 1.00,
    2: 0.75,
    3: 0.50,
    4: 0.25,
}

SOURCE_CRED_TIER_LABELS: dict[int, str] = {
    1: "High authority",
    2: "Established",
    3: "General",
    4: "Unknown / unverified",
}

DISCLAIMER_PHRASE: str = "Domain tier is a provenance heuristic, not proof of factual correctness."


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class SourceCredibilityError(Exception):
    """Base exception for all Source Credibility errors."""

    pass


class SourceCredibilityConfigError(SourceCredibilityError):
    """Raised when configuration weights or parameters are invalid."""

    pass


class SourceCredibilityInputError(SourceCredibilityError):
    """Raised when document input cannot be parsed or is an unsupported type."""

    pass


class SourceCredibilityScoringError(SourceCredibilityError):
    """Raised when an internal scoring invariant is violated."""

    pass


# ---------------------------------------------------------------------------
# Config Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCredibilityWeights:
    """Weights for the composite source credibility scoring formula."""

    w_domain: float = SOURCE_CRED_W_DOMAIN
    w_type: float = SOURCE_CRED_W_TYPE
    w_provenance: float = SOURCE_CRED_W_PROVENANCE

    def __post_init__(self) -> None:
        """Validate weight ranges, finiteness, and sum constraint."""
        for name, w in [
            ("w_domain", self.w_domain),
            ("w_type", self.w_type),
            ("w_provenance", self.w_provenance),
        ]:
            if not isinstance(w, (int, float)) or math.isnan(w) or math.isinf(w):
                raise SourceCredibilityConfigError(f"Weight '{name}' must be a finite float, got {w}.")
            if w < 0.0 or w > 1.0:
                raise SourceCredibilityConfigError(f"Weight '{name}' must be in [0.0, 1.0], got {w}.")

        total = self.w_domain + self.w_type + self.w_provenance
        if abs(total - 1.0) > 1e-9:
            raise SourceCredibilityConfigError(
                f"Source credibility weights must sum to 1.0, got {total:.10f}."
            )


@dataclass
class SourceCredibilitySettings:
    """Runtime configuration settings for Source Credibility and Domain Authority Scoring."""

    weights: SourceCredibilityWeights = field(default_factory=SourceCredibilityWeights)
    tier1_suffixes: tuple[str, ...] = field(default_factory=lambda: SOURCE_CRED_TIER1_SUFFIXES)
    tier2_domains: tuple[str, ...] = field(default_factory=lambda: SOURCE_CRED_TIER2_DOMAINS)
    tier3_domains: tuple[str, ...] = field(default_factory=lambda: SOURCE_CRED_TIER3_DOMAINS)
    placeholder_suffixes: tuple[str, ...] = field(default_factory=lambda: SOURCE_CRED_PLACEHOLDER_SUFFIXES)
    provenance_fields: tuple[str, ...] = field(default_factory=lambda: SOURCE_CRED_PROVENANCE_FIELDS)
    source_type_map: dict[str, float] = field(default_factory=lambda: dict(SOURCE_CRED_SOURCE_TYPE_MAP))
    tier_scores: dict[int, float] = field(default_factory=lambda: dict(SOURCE_CRED_TIER_SCORES))
    tier_labels: dict[int, str] = field(default_factory=lambda: dict(SOURCE_CRED_TIER_LABELS))
    model_version: str = "source_credibility_v1"
    disclaimer: str = DISCLAIMER_PHRASE
