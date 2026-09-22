"""Configuration, threshold heuristics, and exception definitions for Step 4.4.

Provides settings, validation rules, default heuristic thresholds, and typed exceptions
for Evidence Pollution & Circularity Defense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Documented heuristic thresholds (§24)
DEFAULT_UNGROUNDED_THRESHOLD: float = 0.15
DEFAULT_WEAK_GROUNDING_THRESHOLD: float = 0.35
DEFAULT_MINIMUM_MEANINGFUL_TOKENS: int = 5
DEFAULT_MINIMUM_TOKEN_LENGTH: int = 2

# Mandatory research disclaimer phrase (§1.1, §23)
DISCLAIMER_PHRASE: str = (
    "Evidence pollution and circularity are heuristic signals, not proof of claim falsity."
)

from app.utils.lexical import CANONICAL_STOPWORDS

# Curated English stopword set for lexical grounding filtering (§7)
DEFAULT_STOPWORDS: frozenset[str] = CANONICAL_STOPWORDS


class EvidencePollutionError(Exception):
    """Base exception for all Step 4.4 evidence pollution errors."""


class EvidencePollutionConfigError(EvidencePollutionError):
    """Raised when evidence pollution configuration parameters violate domain constraints."""


class EvidencePollutionInputError(EvidencePollutionError):
    """Raised when claim or evidence inputs are invalid, missing, or malformed."""


@dataclass(frozen=True)
class EvidencePollutionSettings:
    """Settings and heuristic thresholds for EvidencePollutionDetector.

    Attributes:
        ungrounded_threshold: Grounding score below which evidence is considered UNGROUNDED.
            Must satisfy 0.0 < ungrounded_threshold <= 1.0. Default: 0.15.
        weak_grounding_threshold: Grounding score below which evidence has weak grounding signal.
            Must satisfy ungrounded_threshold <= weak_grounding_threshold <= 1.0. Default: 0.35.
        minimum_meaningful_tokens: Minimum meaningful token count required on claim and evidence
            before lexical metrics can trigger strong classifications. Default: 5.
        minimum_token_length: Minimum character length for meaningful tokens. Default: 2.
        stopwords: Set of lowercase stopwords to exclude from lexical grounding calculations.
    """

    ungrounded_threshold: float = DEFAULT_UNGROUNDED_THRESHOLD
    weak_grounding_threshold: float = DEFAULT_WEAK_GROUNDING_THRESHOLD
    minimum_meaningful_tokens: int = DEFAULT_MINIMUM_MEANINGFUL_TOKENS
    minimum_token_length: int = DEFAULT_MINIMUM_TOKEN_LENGTH
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)

    def __post_init__(self) -> None:
        """Validate parameter types, finiteness, and range constraints."""
        # Validate ungrounded_threshold
        if not isinstance(self.ungrounded_threshold, (int, float)) or isinstance(
            self.ungrounded_threshold, bool
        ):
            raise EvidencePollutionConfigError(
                f"ungrounded_threshold must be a float, got {type(self.ungrounded_threshold).__name__}"
            )
        if not math.isfinite(self.ungrounded_threshold):
            raise EvidencePollutionConfigError(
                "ungrounded_threshold must be finite, got NaN or Inf"
            )
        if not (0.0 < self.ungrounded_threshold <= 1.0):
            raise EvidencePollutionConfigError(
                f"ungrounded_threshold must be in (0.0, 1.0], got {self.ungrounded_threshold}"
            )

        # Validate weak_grounding_threshold
        if not isinstance(self.weak_grounding_threshold, (int, float)) or isinstance(
            self.weak_grounding_threshold, bool
        ):
            raise EvidencePollutionConfigError(
                f"weak_grounding_threshold must be a float, got {type(self.weak_grounding_threshold).__name__}"
            )
        if not math.isfinite(self.weak_grounding_threshold):
            raise EvidencePollutionConfigError(
                "weak_grounding_threshold must be finite, got NaN or Inf"
            )
        if not (self.ungrounded_threshold <= self.weak_grounding_threshold <= 1.0):
            raise EvidencePollutionConfigError(
                f"weak_grounding_threshold must satisfy ungrounded_threshold <= weak_grounding_threshold <= 1.0, "
                f"got {self.weak_grounding_threshold} (ungrounded_threshold={self.ungrounded_threshold})"
            )

        # Validate minimum_meaningful_tokens
        if not isinstance(self.minimum_meaningful_tokens, int) or isinstance(
            self.minimum_meaningful_tokens, bool
        ):
            raise EvidencePollutionConfigError(
                f"minimum_meaningful_tokens must be an integer, got {type(self.minimum_meaningful_tokens).__name__}"
            )
        if self.minimum_meaningful_tokens < 1:
            raise EvidencePollutionConfigError(
                f"minimum_meaningful_tokens must be >= 1, got {self.minimum_meaningful_tokens}"
            )

        # Validate minimum_token_length
        if not isinstance(self.minimum_token_length, int) or isinstance(
            self.minimum_token_length, bool
        ):
            raise EvidencePollutionConfigError(
                f"minimum_token_length must be an integer, got {type(self.minimum_token_length).__name__}"
            )
        if self.minimum_token_length < 1:
            raise EvidencePollutionConfigError(
                f"minimum_token_length must be >= 1, got {self.minimum_token_length}"
            )
