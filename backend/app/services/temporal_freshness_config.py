"""Configuration and default parameters for Temporal Freshness & Decay Scoring."""

from dataclasses import dataclass
import math

DEFAULT_HALF_LIFE_DAYS: float = 365.0

# Submodule 8.2 Historical Lookback & Retrospective Decay Constants [PROVISIONAL]
# Rationale: Historical lookback of 5 years separates immediate contemporary news cycles
# from established historical scholarship. Retrospective scholarship decay uses a 10-year
# (3650-day) half-life derived from Garfield (1972) citation decay studies showing historical
# and scientific literature retains authoritative utility over decade timescales.
DEFAULT_HISTORICAL_HALF_LIFE_DAYS: float = 3650.0  # 10 years [PROVISIONAL]
DEFAULT_HISTORICAL_LOOKBACK_YEARS: float = 5.0      # 5 years [PROVISIONAL]

# Contemporaneous documentation window: documents published within 2 years (730 days)
# of an event anchor receive F >= 0.95 under linear slope (0.05 / 730).
DEFAULT_CONTEMPORANEOUS_WINDOW_DAYS: int = 730     # 2 years [PROVISIONAL]
DEFAULT_CONTEMPORANEOUS_SLOPE: float = 0.05        # Slope ensuring F >= 0.95 at 730d [PROVISIONAL]

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
    """Settings for exponential decay and anchor-aware temporal freshness scoring.

    Half-life is a configurable heuristic parameter representing the assumed
    temporal decay rate of evidence freshness. It is not a scientifically
    validated universal constant.
    """

    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    historical_half_life_days: float = DEFAULT_HISTORICAL_HALF_LIFE_DAYS
    historical_lookback_years: float = DEFAULT_HISTORICAL_LOOKBACK_YEARS
    contemporaneous_window_days: int = DEFAULT_CONTEMPORANEOUS_WINDOW_DAYS
    contemporaneous_slope: float = DEFAULT_CONTEMPORANEOUS_SLOPE
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
            not isinstance(self.historical_half_life_days, (int, float))
            or math.isnan(self.historical_half_life_days)
            or math.isinf(self.historical_half_life_days)
            or self.historical_half_life_days <= 0.0
        ):
            raise TemporalFreshnessConfigError(
                f"historical_half_life_days must be a finite strictly positive float, got {self.historical_half_life_days}."
            )

        if (
            not isinstance(self.historical_lookback_years, (int, float))
            or math.isnan(self.historical_lookback_years)
            or math.isinf(self.historical_lookback_years)
            or self.historical_lookback_years <= 0.0
        ):
            raise TemporalFreshnessConfigError(
                f"historical_lookback_years must be a finite strictly positive float, got {self.historical_lookback_years}."
            )

        if (
            isinstance(self.contemporaneous_window_days, bool)
            or not isinstance(self.contemporaneous_window_days, int)
            or self.contemporaneous_window_days < 1
        ):
            raise TemporalFreshnessConfigError(
                f"contemporaneous_window_days must be a positive int >= 1, got {self.contemporaneous_window_days}."
            )

        if (
            not isinstance(self.contemporaneous_slope, (int, float))
            or math.isnan(self.contemporaneous_slope)
            or math.isinf(self.contemporaneous_slope)
            or not (0.0 <= self.contemporaneous_slope <= 1.0)
        ):
            raise TemporalFreshnessConfigError(
                f"contemporaneous_slope must be a finite float in [0.0, 1.0], got {self.contemporaneous_slope}."
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
