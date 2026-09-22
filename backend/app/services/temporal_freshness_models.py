"""Pydantic data models for Temporal Freshness Scoring."""

from datetime import date
import math
from pydantic import BaseModel, Field, model_validator


class TemporalFreshnessResult(BaseModel):
    """Structured result of temporal freshness and decay scoring for an evidence item."""

    freshness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized temporal freshness score in [0.0, 1.0].",
    )
    published_date: date | None = Field(
        default=None,
        description="Document publication date if available in metadata.",
    )
    reference_date: date = Field(
        ...,
        description="Explicit baseline reference date against which age was measured.",
    )
    age_days: int | None = Field(
        default=None,
        ge=0,
        description="Calendar days elapsed between publication and reference date (None if missing).",
    )
    half_life_days: float = Field(
        ...,
        gt=0.0,
        description="Configured decay half-life in days.",
    )
    explanation: str = Field(
        ...,
        description="Deterministic human-readable explanation of the score.",
    )
    is_missing_date: bool = Field(
        default=False,
        description="Flag indicating that publication date was unavailable in metadata.",
    )
    is_future_date: bool = Field(
        default=False,
        description="Flag indicating that publication date was after the reference date.",
    )
    model_version: str = Field(
        default="temporal_freshness_v1",
        description="Model version tag.",
    )

    @model_validator(mode="after")
    def validate_invariants(self) -> "TemporalFreshnessResult":
        """Validate mathematical consistency and presence constraints."""
        if math.isnan(self.freshness_score) or math.isinf(self.freshness_score):
            raise ValueError(f"freshness_score must be a finite float, got {self.freshness_score}.")

        if math.isnan(self.half_life_days) or math.isinf(self.half_life_days):
            raise ValueError(f"half_life_days must be a finite float, got {self.half_life_days}.")

        if self.is_missing_date:
            if self.published_date is not None:
                raise ValueError("published_date must be None when is_missing_date is True.")
            if self.age_days is not None:
                raise ValueError("age_days must be None when is_missing_date is True.")
        else:
            if self.published_date is None:
                raise ValueError("published_date cannot be None when is_missing_date is False.")
            if self.age_days is None:
                raise ValueError("age_days cannot be None when is_missing_date is False.")

        return self
