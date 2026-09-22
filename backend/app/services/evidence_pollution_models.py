"""Pydantic data models for Step 4.4 Evidence Pollution & Circularity Defense.

Provides typed models for individual claim-evidence pollution evaluations and batch set results,
strictly honoring the Anti-Fabrication Contract (§5).
"""

from __future__ import annotations

import math
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.evidence_pollution_config import DISCLAIMER_PHRASE


class PollutionClass(str, Enum):
    """Categorical classification of evidence pollution signal (§9).

    Only classes with deterministic triggers in the current repository schema are included.
    Full citation graph circularity is unsupported by the schema and disabled per §5.
    """

    SELF_REFERENTIAL = "SELF_REFERENTIAL"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    UNGROUNDED = "UNGROUNDED"
    WEAK_POLLUTION_SIGNAL = "WEAK_POLLUTION_SIGNAL"
    CLEAN = "CLEAN"


class EvidencePollutionResult(BaseModel):
    """Structured result of claim-evidence pollution and grounding evaluation (§14).

    Attributes:
        document_id: Identifier of the evaluated evidence document.
        pollution_score: Normalized heuristic pollution score in [0.0, 1.0].
        classification: Categorical classification of pollution severity.
        claim_token_count: Number of meaningful non-stopword tokens in the claim.
        evidence_token_count: Number of meaningful non-stopword tokens in the evidence.
        claim_evidence_overlap: Directional overlap of claim tokens in evidence: |C ∩ E| / |C|.
        evidence_claim_overlap: Directional overlap of evidence tokens in claim: |E ∩ C| / |E|.
        grounding_score: Directional lexical grounding score of claim in evidence.
        is_exact_duplicate: True if evidence duplicates claim text, matching canonical URL, or repeated doc ID.
        is_self_referential: True if evidence appears to merely repeat or self-reference the claim origin.
        is_circular: Always False in current repository schema (no citation graph exists).
        circularity_supported: False (Anti-Fabrication Contract §5).
        explanation: Deterministic explanation detailing signals, precedence, and disclaimer.
        disclaimer: Mandatory research boundary disclaimer (§1.1, §23).
        model_version: Semantic version of this component.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Identifier of the evaluated evidence document.")
    pollution_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized heuristic pollution score in [0.0, 1.0].",
    )
    classification: PollutionClass = Field(
        description="Categorical classification of evidence pollution."
    )
    claim_token_count: int = Field(
        ge=0,
        description="Number of meaningful non-stopword tokens in the claim.",
    )
    evidence_token_count: int = Field(
        ge=0,
        description="Number of meaningful non-stopword tokens in the evidence.",
    )
    claim_evidence_overlap: float = Field(
        ge=0.0,
        le=1.0,
        description="Directional token overlap: |Tokens(C) ∩ Tokens(E)| / |Tokens(C)|.",
    )
    evidence_claim_overlap: float = Field(
        ge=0.0,
        le=1.0,
        description="Directional token overlap: |Tokens(E) ∩ Tokens(C)| / |Tokens(E)|.",
    )
    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Directional lexical grounding of claim in evidence.",
    )
    is_exact_duplicate: bool = Field(
        description="Flag indicating exact content, URL, or cross-evidence repetition."
    )
    is_self_referential: bool = Field(
        description="Flag indicating evidence repeats claim or matches claim origin."
    )
    is_circular: bool = Field(
        default=False,
        description="Circularity flag (False in current schema per Anti-Fabrication Contract).",
    )
    circularity_supported: bool = Field(
        default=False,
        description="Indicates whether repository schema supports citation graph circularity.",
    )
    explanation: str = Field(
        description="Deterministic human-readable explanation with embedded disclaimer."
    )
    disclaimer: str = Field(
        default=DISCLAIMER_PHRASE,
        description="Mandatory heuristic boundary notice.",
    )
    model_version: str = Field(
        default="4.4.0",
        description="Component semantic version.",
    )

    @field_validator("pollution_score", "claim_evidence_overlap", "evidence_claim_overlap", "grounding_score")
    @classmethod
    def check_finite(cls, v: float) -> float:
        """Ensure score fields are finite and not NaN or Inf."""
        if not math.isfinite(v):
            raise ValueError(f"Score value must be finite, got {v}")
        return v


class BatchPollutionResult(BaseModel):
    """Batch-level summary across a set of retrieved evidence items evaluated against a claim (§15)."""

    model_config = ConfigDict(frozen=True)

    total_items: int = Field(description="Total number of evaluated evidence items.")
    evaluations: list[EvidencePollutionResult] = Field(
        description="Ordered list of individual evidence evaluations."
    )
    polluted_document_ids: list[str] = Field(
        description="Document IDs classified as SELF_REFERENTIAL, EXACT_DUPLICATE, or UNGROUNDED."
    )
    ungrounded_document_ids: list[str] = Field(
        description="Document IDs specifically classified as UNGROUNDED."
    )
    clean_document_ids: list[str] = Field(
        description="Document IDs classified as CLEAN."
    )
    disclaimer: str = Field(
        default=DISCLAIMER_PHRASE,
        description="Mandatory heuristic boundary notice.",
    )
    model_version: str = Field(
        default="4.4.0",
        description="Component semantic version.",
    )
