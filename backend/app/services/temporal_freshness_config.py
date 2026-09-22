"""Configuration and default parameters for Temporal Freshness & Decay Scoring."""

from dataclasses import dataclass
import math

DEFAULT_HALF_LIFE_DAYS: float = 365.0
DEFAULT_MISSING_DATE_FALLBACK: float = 0.50
DISCLAIMER_PHRASE: str = "Temporal freshness is a heuristic, not a truth signal."
MODEL_VERSION: str = "temporal_freshness_v1"


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class TemporalFreshnessError(Exception):
    """Base exception for all temporal freshness errors."""

    pass


class TemporalFreshnessConfigError(TemporalFreshnessError):
    """Raised when temporal freshness configuration parameters are invalid."""

    pass


class TemporalFreshnessInputError(TemporalFreshnessError):
    """Raised when dates or inputs fail type/value validation."""

    pass


# ---------------------------------------------------------------------------
# Configuration Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalFreshnessSettings:
    """Settings for exponential decay temporal freshness scoring.

    Half-life is a configurable heuristic parameter representing the assumed
    temporal decay rate of evidence freshness. It is not a scientifically
    validated universal constant.
    """

    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    missing_date_fallback: float = DEFAULT_MISSING_DATE_FALLBACK
    model_version: str = MODEL_VERSION
    disclaimer: str = DISCLAIMER_PHRASE

    def __post_init__(self) -> None:
        """Validate parameter ranges and finiteness."""
        if (
            not isinstance(self.half_life_days, (int, float))
            or math.isnan(self.half_life_days)
            or math.isinf(self.half_life_days)
            or self.half_life_days <= 0.0
        ):
            raise TemporalFreshnessConfigError(
                f"half_life_days must be a finite strictly positive float, got {self.half_life_days}."
            )

        if (
            not isinstance(self.missing_date_fallback, (int, float))
            or math.isnan(self.missing_date_fallback)
            or math.isinf(self.missing_date_fallback)
            or not (0.0 <= self.missing_date_fallback <= 1.0)
        ):
            raise TemporalFreshnessConfigError(
                f"missing_date_fallback must be a finite float in [0.0, 1.0], got {self.missing_date_fallback}."
            )
