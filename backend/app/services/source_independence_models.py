"""Pydantic data models for Step 4.3 Source Independence & Redundancy Detector.

Provides typed models for pairwise comparison results, redundancy classification categories,
and group/set-level complete-linkage clustering results.
"""

from __future__ import annotations

import math
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.source_independence_config import DISCLAIMER_PHRASE


class RedundancyClass(str, Enum):
    """Categorical classification of evidence redundancy relationship (§20)."""

    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    REDUNDANT = "REDUNDANT"
    WEAKLY_RELATED = "WEAKLY_RELATED"
    INDEPENDENT = "INDEPENDENT"


class IndependencePairResult(BaseModel):
    """Structured result of pairwise source independence and redundancy comparison (§19).

    Attributes:
        document_id_a: Identifier of first evidence document.
        document_id_b: Identifier of second evidence document.
        text_similarity: Normalized token Jaccard similarity in [0.0, 1.0].
        is_exact_duplicate: True if texts or canonical URLs match identically.
        is_near_duplicate: True if lexical similarity exceeds near_duplicate_threshold (with sufficient tokens).
        same_domain: True if both evidence items originate from the same normalized domain.
        same_source: True if both evidence items share the same normalized source string.
        same_url: True if both evidence items share the same normalized source URL.
        shared_metadata_signal: True if domain, source, or URL metadata indicate non-independence.
        redundancy_class: Categorical classification (EXACT_DUPLICATE, REDUNDANT, WEAKLY_RELATED, INDEPENDENT).
        independence_score: Normalized independence score in [0.0, 1.0] (0 = redundant, 1 = independent).
        explanation: Deterministic human-readable explanation with embedded disclaimer.
        disclaimer: Standard research heuristic disclaimer.
        model_version: Semantic version of this scoring component.
    """

    model_config = ConfigDict(frozen=True)

    document_id_a: str = Field(description="Identifier of first evidence document.")
    document_id_b: str = Field(description="Identifier of second evidence document.")
    text_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Lexical token Jaccard similarity in [0.0, 1.0].",
    )
    is_exact_duplicate: bool = Field(
        description="Flag indicating exact normalized text or URL identity."
    )
    is_near_duplicate: bool = Field(
        description="Flag indicating lexical similarity meets or exceeds near-duplicate threshold."
    )
    same_domain: bool = Field(description="Flag indicating identical normalized domain.")
    same_source: bool = Field(description="Flag indicating identical normalized source.")
    same_url: bool = Field(description="Flag indicating identical normalized source URL.")
    shared_metadata_signal: bool = Field(
        description="Flag indicating presence of shared metadata relationship."
    )
    redundancy_class: RedundancyClass = Field(
        description="Categorical redundancy classification."
    )
    independence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized independence score in [0.0, 1.0].",
    )
    explanation: str = Field(
        description="Deterministic explanation detailing signals and disclaimer."
    )
    disclaimer: str = Field(
        default=DISCLAIMER_PHRASE,
        description="Heuristic boundary notice.",
    )
    model_version: str = Field(
        default="4.3.0",
        description="Module step semantic version.",
    )

    @field_validator("text_similarity", "independence_score")
    @classmethod
    def check_finite(cls, v: float) -> float:
        """Ensure float fields are finite numbers and not NaN or Inf."""
        if not math.isfinite(v):
            raise ValueError(f"Score value must be finite, got {v}")
        return v


class IndependenceCluster(BaseModel):
    """Cluster of redundant or syndicated evidence documents identified via complete linkage."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int = Field(description="Deterministic 0-indexed cluster identifier.")
    document_ids: list[str] = Field(
        description="List of document IDs belonging to this redundancy cluster."
    )
    cluster_type: RedundancyClass = Field(
        description="Classification of redundancy within this cluster (EXACT_DUPLICATE or REDUNDANT)."
    )


class IndependenceSetResult(BaseModel):
    """Group-level result summarizing evidence set independence and complete-linkage clusters (§23)."""

    model_config = ConfigDict(frozen=True)

    total_documents: int = Field(description="Total number of documents in the evaluated set.")
    pairwise_comparisons: list[IndependencePairResult] = Field(
        description="Canonical pairwise comparison results (A, B) with i < j."
    )
    clusters: list[IndependenceCluster] = Field(
        description="Clusters of redundant documents."
    )
    independent_document_ids: list[str] = Field(
        description="Document IDs that do not belong to any redundancy cluster."
    )
    mean_pairwise_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Average pairwise lexical similarity across all canonical pairs.",
    )
    disclaimer: str = Field(
        default=DISCLAIMER_PHRASE,
        description="Heuristic boundary notice.",
    )
    model_version: str = Field(
        default="4.3.0",
        description="Module step semantic version.",
    )

    @field_validator("mean_pairwise_similarity")
    @classmethod
    def check_finite(cls, v: float) -> float:
        """Ensure mean similarity is finite."""
        if not math.isfinite(v):
            raise ValueError(f"Score value must be finite, got {v}")
        return v
