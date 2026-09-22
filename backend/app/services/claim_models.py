"""Pydantic data models for atomic claims and extraction results."""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class AtomicClaim(BaseModel):
    """Represent an independently verifiable atomic factual claim."""

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Deterministic unique identifier for this claim (e.g., 'c1', 'c2').",
    )
    claim_text: str = Field(
        ...,
        min_length=1,
        description="Normalized atomic factual claim text.",
    )
    subject: str | None = Field(
        default=None,
        description="Extracted subject entity of the claim.",
    )
    predicate: str | None = Field(
        default=None,
        description="Extracted action, relation, or state predicate.",
    )
    object: str | None = Field(
        default=None,
        description="Extracted object entity, complement, or target value.",
    )
    temporal_anchor: str | None = Field(
        default=None,
        description="Deprecated: use temporal_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    numerical_value: str | None = Field(
        default=None,
        description="Deprecated: use numerical_values as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    spatial_anchor: str | None = Field(
        default=None,
        description="Deprecated: use spatial_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    temporal_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted dates, years, or temporal constraints.",
    )
    numerical_values: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted numerical quantities, measurements, or metrics.",
    )
    spatial_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted spatial locations or geographical/astronomical entities.",
    )
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Extracted factual qualifiers, restrictive conditions, or modifiers.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_anchor_invariants(cls, data: Any) -> Any:
        """Enforce scalar == plural[0] synchronization invariant."""
        if isinstance(data, dict):
            # Sync temporal
            t_list = data.get("temporal_anchors")
            t_scalar = data.get("temporal_anchor")
            if t_list and not t_scalar:
                data["temporal_anchor"] = t_list[0]
            elif t_scalar and not t_list:
                data["temporal_anchors"] = [t_scalar]
            elif t_list and t_scalar and t_scalar != t_list[0]:
                data["temporal_anchor"] = t_list[0]

            # Sync numerical
            n_list = data.get("numerical_values")
            n_scalar = data.get("numerical_value")
            if n_list and not n_scalar:
                data["numerical_value"] = n_list[0]
            elif n_scalar and not n_list:
                data["numerical_values"] = [n_scalar]
            elif n_list and n_scalar and n_scalar != n_list[0]:
                data["numerical_value"] = n_list[0]

            # Sync spatial
            s_list = data.get("spatial_anchors")
            s_scalar = data.get("spatial_anchor")
            if s_list and not s_scalar:
                data["spatial_anchor"] = s_list[0]
            elif s_scalar and not s_list:
                data["spatial_anchors"] = [s_scalar]
            elif s_list and s_scalar and s_scalar != s_list[0]:
                data["spatial_anchor"] = s_list[0]

        return data
    original_clause: str | None = Field(
        default=None,
        description="Original un-revised clause text prior to self-containment revision.",
    )
    is_self_contained: bool = Field(
        default=True,
        description="Whether this claim has been resolved to be verifiable in isolation.",
    )

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        """Reject empty or whitespace-only claim IDs."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("claim_id must contain at least one non-whitespace character.")
        return trimmed

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, value: str) -> str:
        """Reject empty or whitespace-only claim text."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("claim_text must contain at least one non-whitespace character.")
        return trimmed


class DecompositionMetrics(BaseModel):
    """Quality metrics evaluating the factual decomposition of text into atomic claims."""

    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall of key content tokens and named entities from the source text (did extraction drop info?).",
    )
    atomicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion of extracted claims that are truly indivisible single propositions.",
    )
    entailment_preservation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Degree to which the extracted claims faithfully preserve source facts without distortion.",
    )
    dropped_entities: list[str] = Field(
        default_factory=list,
        description="Salient named entities or quantities present in source text but omitted from claims.",
    )
    is_high_quality: bool = Field(
        description="True if coverage >= 0.70 and atomicity >= 0.80.",
    )


class ClaimExtractionResult(BaseModel):
    """Represent the result of decomposing text into atomic claims."""

    claims: list[AtomicClaim] = Field(
        default_factory=list,
        description="Ordered list of extracted atomic claims.",
    )
    extraction_method: str = Field(
        default="nlp",
        description="Method used for extraction (e.g., 'nlp', 'llm', 'hybrid').",
    )
    raw_response_text: str = Field(
        ...,
        description="The original unparsed response text.",
    )
    decomposition_metrics: DecompositionMetrics | None = Field(
        default=None,
        description="Automated quality evaluation metrics for the decomposition.",
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware UTC timestamp of the extraction operation.",
    )
