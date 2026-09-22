"""Data models and structured output schemas for Source Credibility Scoring."""

import math
from pydantic import BaseModel, Field, model_validator


class CredibilityFactor(BaseModel):
    """Structured breakdown factor contributing to the overall credibility score."""

    name: str = Field(..., description="Component name ('domain', 'provenance', or 'source_type').")
    raw_value: float = Field(..., ge=0.0, le=1.0, description="Raw normalized component score in [0.0, 1.0].")
    weight: float = Field(..., ge=0.0, le=1.0, description="Effective weight applied to this component.")
    contribution: float = Field(..., ge=0.0, le=1.0, description="Weighted contribution to final credibility score.")
    note: str = Field(..., description="Deterministic explanation note for this factor.")


class SourceCredibilityResult(BaseModel):
    """Comprehensive, explainable credibility assessment for an evidence document."""

    document_id: str = Field(..., description="Identifier of the scored document.")
    normalized_domain: str = Field(..., description="Canonical hostname extracted from document metadata.")
    domain_tier: int = Field(..., ge=1, le=4, description="Categorical authority tier (1 to 4).")
    domain_tier_label: str = Field(..., description="Descriptive label for the authority tier.")
    domain_score: float = Field(..., ge=0.0, le=1.0, description="Domain authority score component.")
    provenance_score: float = Field(..., ge=0.0, le=1.0, description="Completeness of document provenance metadata.")
    source_type_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Source type score component (None if source_type is absent in document).",
    )
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Overall clamped composite credibility score.")
    weights: dict[str, float] = Field(..., description="Effective weights applied in composite scoring.")
    factors: list[CredibilityFactor] = Field(..., description="Ordered list of structured explanation factors.")
    explanation: str = Field(..., description="Deterministic human-readable explanation.")
    model_version: str = Field(default="source_credibility_v1", description="Model version tag.")

    @model_validator(mode="after")
    def validate_invariants(self) -> "SourceCredibilityResult":
        """Validate finiteness and mathematical consistency."""
        for score_name, score_val in [
            ("domain_score", self.domain_score),
            ("provenance_score", self.provenance_score),
            ("credibility_score", self.credibility_score),
        ]:
            if math.isnan(score_val) or math.isinf(score_val):
                raise ValueError(f"Field '{score_name}' must be finite, got {score_val}.")

        if self.source_type_score is not None:
            if math.isnan(self.source_type_score) or math.isinf(self.source_type_score):
                raise ValueError(f"Field 'source_type_score' must be finite, got {self.source_type_score}.")

        weights_total = sum(self.weights.values())
        if abs(weights_total - 1.0) > 1e-4:
            raise ValueError(f"Effective weights must sum to 1.0, got {weights_total:.6f}.")

        return self
