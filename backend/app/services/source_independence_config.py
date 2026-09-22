"""Configuration and exception definitions for Step 4.3 Source Independence & Redundancy Detector.

Provides settings, validation rules, default heuristic thresholds, and typed exceptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Documented heuristic thresholds (§15, §26)
DEFAULT_NEAR_DUPLICATE_THRESHOLD: float = 0.85
DEFAULT_WEAK_RELATION_THRESHOLD: float = 0.50
DEFAULT_MINIMUM_MEANINGFUL_TOKENS: int = 5

# Mandatory disclaimer phrase (§32)
DISCLAIMER_PHRASE: str = (
    "Independence and redundancy are heuristic signals, not proof of source independence."
)


class SourceIndependenceError(Exception):
    """Base exception for all Step 4.3 source independence errors."""


class SourceIndependenceConfigError(SourceIndependenceError):
    """Raised when source independence configuration parameters violate domain constraints."""


class SourceIndependenceInputError(SourceIndependenceError):
    """Raised when evidence documents or comparison inputs are invalid, missing, or malformed."""


@dataclass(frozen=True)
class SourceIndependenceSettings:
    """Settings and heuristic thresholds for SourceIndependenceDetector.

    Attributes:
        near_duplicate_threshold: Token Jaccard threshold above which pairs are considered REDUNDANT.
            Must satisfy 0.0 < near_duplicate_threshold <= 1.0. Default: 0.85.
        weak_relation_threshold: Token Jaccard threshold above which pairs are considered WEAKLY_RELATED.
            Must satisfy 0.0 <= weak_relation_threshold < near_duplicate_threshold. Default: 0.50.
        minimum_meaningful_tokens: Minimum token count required on both sides before near-duplicate
            classification can apply without a hard metadata match (e.g. matching URL).
            Must be an integer >= 1. Default: 5.
    """

    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    weak_relation_threshold: float = DEFAULT_WEAK_RELATION_THRESHOLD
    minimum_meaningful_tokens: int = DEFAULT_MINIMUM_MEANINGFUL_TOKENS

    def __post_init__(self) -> None:
        """Validate parameter types, finiteness, and range constraints."""
        # Validate near_duplicate_threshold
        if not isinstance(self.near_duplicate_threshold, (int, float)) or isinstance(
            self.near_duplicate_threshold, bool
        ):
            raise SourceIndependenceConfigError(
                f"near_duplicate_threshold must be a float, got {type(self.near_duplicate_threshold).__name__}"
            )
        if not math.isfinite(self.near_duplicate_threshold):
            raise SourceIndependenceConfigError(
                "near_duplicate_threshold must be finite, got NaN or Inf"
            )
        if not (0.0 < self.near_duplicate_threshold <= 1.0):
            raise SourceIndependenceConfigError(
                f"near_duplicate_threshold must be in (0.0, 1.0], got {self.near_duplicate_threshold}"
            )

        # Validate weak_relation_threshold
        if not isinstance(self.weak_relation_threshold, (int, float)) or isinstance(
            self.weak_relation_threshold, bool
        ):
            raise SourceIndependenceConfigError(
                f"weak_relation_threshold must be a float, got {type(self.weak_relation_threshold).__name__}"
            )
        if not math.isfinite(self.weak_relation_threshold):
            raise SourceIndependenceConfigError(
                "weak_relation_threshold must be finite, got NaN or Inf"
            )
        if not (0.0 <= self.weak_relation_threshold < self.near_duplicate_threshold):
            raise SourceIndependenceConfigError(
                f"weak_relation_threshold must be in [0.0, near_duplicate_threshold), "
                f"got {self.weak_relation_threshold} (near_duplicate_threshold={self.near_duplicate_threshold})"
            )

        # Validate minimum_meaningful_tokens
        if not isinstance(self.minimum_meaningful_tokens, int) or isinstance(
            self.minimum_meaningful_tokens, bool
        ):
            raise SourceIndependenceConfigError(
                f"minimum_meaningful_tokens must be an integer, got {type(self.minimum_meaningful_tokens).__name__}"
            )
        if self.minimum_meaningful_tokens < 1:
            raise SourceIndependenceConfigError(
                f"minimum_meaningful_tokens must be >= 1, got {self.minimum_meaningful_tokens}"
            )
