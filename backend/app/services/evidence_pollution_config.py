"""Configuration, threshold heuristics, and exception definitions for Step 4.4.

Provides settings, validation rules, default heuristic thresholds, and typed exceptions
for Evidence Pollution & Circularity Defense.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Documented heuristic thresholds (§24, Module 8.4)
# [PROVISIONAL] Derived from empirical lexical overlap distributions on news scrapers where
# boilerplate headers exhibit < 15% content overlap while substantive quotations exceed 50%.
# Clean retention target (>95%) enforces non-destructive retrieval pre-filtering.
# Routed to Submodule 8.6 FEVER/SciFact Calibration Checkpoint.
POLLUTION_CIRCULARITY_THRESHOLD: float = 0.90
POLLUTION_LOW_GROUNDING_THRESHOLD: float = 0.80
POLLUTION_BOILERPLATE_THRESHOLD: float = 0.60
POLLUTION_CLEAN_THRESHOLD: float = 0.20
POLLUTION_CLEAN_RETENTION_TARGET: float = 0.95

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

# Standard regex patterns for scraped navigation / web template boilerplate (§8.4)
DEFAULT_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:all\s+rights?\s+reserved|terms\s+of\s+(?:service|use)|privacy\s+policy)\b", re.IGNORECASE),
    re.compile(r"\b(?:cookie\s+policy|accept\s+(?:all\s+)?cookies?|manage\s+cookie\s+preferences)\b", re.IGNORECASE),
    re.compile(r"\b(?:subscribe\s+(?:to\s+(?:read|continue)|now)|sign\s+up\s+for\s+(?:our\s+)?newsletter)\b", re.IGNORECASE),
    re.compile(r"\b(?:share\s+on\s+(?:facebook|twitter|linkedin)|follow\s+us\s+on)\b", re.IGNORECASE),
    re.compile(r"\b(?:advertisement|sponsored\s+content|click\s+here\s+to\s+(?:read|view))\b", re.IGNORECASE),
    re.compile(r"\b(?:leave\s+a\s+comment|leave\s+a\s+reply|related\s+(?:articles|posts|stories))\b", re.IGNORECASE),
)


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
        circularity_threshold: Pollution score threshold for Circularity / Self-referential / Exact duplicate. Default: 0.90.
        low_grounding_threshold: Pollution score threshold for Ungrounded / Severely ungrounded evidence. Default: 0.80.
        boilerplate_threshold: Pollution score threshold for Boilerplate / Scraped templates. Default: 0.60.
        clean_threshold: Maximum pollution score for Clean evidence. Default: 0.20.
        clean_retention_target: Target retention specificity for non-destructive pre-filtering. Default: 0.95.
        boilerplate_patterns: Tuple of compiled regex patterns to detect web boilerplate headers/footers.
        stopwords: Set of lowercase stopwords to exclude from lexical grounding calculations.
    """

    ungrounded_threshold: float = DEFAULT_UNGROUNDED_THRESHOLD
    weak_grounding_threshold: float = DEFAULT_WEAK_GROUNDING_THRESHOLD
    minimum_meaningful_tokens: int = DEFAULT_MINIMUM_MEANINGFUL_TOKENS
    minimum_token_length: int = DEFAULT_MINIMUM_TOKEN_LENGTH
    circularity_threshold: float = POLLUTION_CIRCULARITY_THRESHOLD
    low_grounding_threshold: float = POLLUTION_LOW_GROUNDING_THRESHOLD
    boilerplate_threshold: float = POLLUTION_BOILERPLATE_THRESHOLD
    clean_threshold: float = POLLUTION_CLEAN_THRESHOLD
    clean_retention_target: float = POLLUTION_CLEAN_RETENTION_TARGET
    boilerplate_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: DEFAULT_BOILERPLATE_PATTERNS
    )
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

        # Validate 4-layer hierarchy thresholds (§8.4)
        for name, val in [
            ("clean_threshold", self.clean_threshold),
            ("boilerplate_threshold", self.boilerplate_threshold),
            ("low_grounding_threshold", self.low_grounding_threshold),
            ("circularity_threshold", self.circularity_threshold),
            ("clean_retention_target", self.clean_retention_target),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise EvidencePollutionConfigError(
                    f"{name} must be a float, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise EvidencePollutionConfigError(f"{name} must be finite, got NaN or Inf")
            if not (0.0 <= val <= 1.0):
                raise EvidencePollutionConfigError(f"{name} must be in [0.0, 1.0], got {val}")

        # Validate monotonicity of the 4-layer hierarchy
        if not (
            self.clean_threshold
            <= self.boilerplate_threshold
            <= self.low_grounding_threshold
            <= self.circularity_threshold
        ):
            raise EvidencePollutionConfigError(
                "4-layer pollution thresholds must satisfy "
                "clean_threshold <= boilerplate_threshold <= low_grounding_threshold <= circularity_threshold, "
                f"got clean={self.clean_threshold}, boilerplate={self.boilerplate_threshold}, "
                f"low_grounding={self.low_grounding_threshold}, circularity={self.circularity_threshold}"
            )
