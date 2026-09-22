"""Temporal Freshness and Decay Scorer.

Computes a deterministic evidence freshness score based on exponential decay
relative to an explicitly provided reference date.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import logging
from typing import Any

from app.services.temporal_freshness_config import (
    TemporalFreshnessInputError,
    TemporalFreshnessSettings,
)
from app.services.temporal_freshness_models import TemporalFreshnessResult

logger = logging.getLogger("hallucination_verifier.temporal_freshness")


class TemporalFreshnessScorer:
    """Stateless, deterministic temporal freshness scorer using exponential decay.

    Formula:
        F_temporal = 2 ** (-age_days / half_life_days)

    Note:
        Temporal freshness is an evidence-quality indicator, not a factual truth signal.
        Fresh evidence is not automatically true; older evidence is not automatically false.
    """

    def __init__(self, settings: TemporalFreshnessSettings | None = None) -> None:
        """Initialize scorer with optional custom configuration settings."""
        self.settings = settings if settings is not None else TemporalFreshnessSettings()

    def score(
        self,
        published_date: date | None,
        reference_date: date,
    ) -> TemporalFreshnessResult:
        """Compute temporal freshness score for a publication date against a reference date.

        Args:
            published_date: Document publication date, or None if unavailable in metadata.
            reference_date: Explicit baseline date (e.g. claim extraction date).

        Returns:
            TemporalFreshnessResult with normalized score, age, flags, and explanation.

        Raises:
            TemporalFreshnessInputError: If date types are invalid or reference_date is None.
        """
        # Validate reference_date
        if reference_date is None:
            raise TemporalFreshnessInputError("reference_date must be provided and cannot be None.")
        if not isinstance(reference_date, date):
            raise TemporalFreshnessInputError(
                f"reference_date must be an instance of datetime.date, got {type(reference_date).__name__}."
            )
        # Normalize if a datetime instance was passed
        ref_d: date = reference_date.date() if isinstance(reference_date, datetime) else reference_date

        # Handle Case D: Missing publication date
        if published_date is None:
            score_val = self.settings.missing_date_fallback
            explanation = (
                f"Publication date is unavailable; temporal freshness cannot be directly estimated, "
                f"so the configured neutral fallback score of {score_val:.2f} was used.\n\n"
                f"Note: {self.settings.disclaimer}"
            )
            return TemporalFreshnessResult(
                freshness_score=round(score_val, 4),
                published_date=None,
                reference_date=ref_d,
                age_days=None,
                half_life_days=self.settings.half_life_days,
                explanation=explanation,
                is_missing_date=True,
                is_future_date=False,
                model_version=self.settings.model_version,
            )

        # Validate published_date
        if not isinstance(published_date, date):
            raise TemporalFreshnessInputError(
                f"published_date must be an instance of datetime.date or None, got {type(published_date).__name__}."
            )
        pub_d: date = published_date.date() if isinstance(published_date, datetime) else published_date

        # Compute calendar days elapsed
        diff_days = (ref_d - pub_d).days

        # Handle Case C: Future publication date
        if diff_days < 0:
            explanation = (
                f"Publication date ({pub_d.isoformat()}) is later than the supplied reference date "
                f"({ref_d.isoformat()}); age was capped at 0 days and freshness was assigned 1.00.\n\n"
                f"Note: {self.settings.disclaimer}"
            )
            return TemporalFreshnessResult(
                freshness_score=1.0,
                published_date=pub_d,
                reference_date=ref_d,
                age_days=0,
                half_life_days=self.settings.half_life_days,
                explanation=explanation,
                is_missing_date=False,
                is_future_date=True,
                model_version=self.settings.model_version,
            )

        # Handle Case A & B: Same-day or historical publication date
        age_days = diff_days
        raw_score = 2.0 ** (-age_days / self.settings.half_life_days)
        clamped_score = max(0.0, min(1.0, float(raw_score)))

        explanation = (
            f"Evidence published on {pub_d.isoformat()} is {age_days} days old relative to reference date "
            f"{ref_d.isoformat()}; temporal freshness score is {clamped_score:.2f} under the configured "
            f"decay half-life ({self.settings.half_life_days:.1f} days).\n\n"
            f"Note: {self.settings.disclaimer}"
        )

        return TemporalFreshnessResult(
            freshness_score=round(clamped_score, 4),
            published_date=pub_d,
            reference_date=ref_d,
            age_days=age_days,
            half_life_days=self.settings.half_life_days,
            explanation=explanation,
            is_missing_date=False,
            is_future_date=False,
            model_version=self.settings.model_version,
        )

    def score_document(
        self,
        document: Any,
        reference_date: date,
    ) -> TemporalFreshnessResult:
        """Convenience method to score an evidence document or mapping object."""
        if document is None or not (isinstance(document, Mapping) or hasattr(document, "__dict__") or hasattr(document, "__dataclass_fields__")):
            raise TemporalFreshnessInputError(f"Unsupported document input type: {type(document).__name__}.")

        if isinstance(document, Mapping):
            pub_date = document.get("published_date")
        else:
            pub_date = getattr(document, "published_date", None)

        return self.score(published_date=pub_date, reference_date=reference_date)

    def score_batch(
        self,
        items: Sequence[tuple[date | None, date]],
    ) -> list[TemporalFreshnessResult]:
        """Score a collection of (published_date, reference_date) pairs preserving input order."""
        if not items:
            return []
        return [self.score(pub_d, ref_d) for pub_d, ref_d in items]


# ---------------------------------------------------------------------------
# Singleton Accessor & Reset
# ---------------------------------------------------------------------------

_temporal_scorer_instance: TemporalFreshnessScorer | None = None


def get_temporal_freshness_scorer(
    settings: TemporalFreshnessSettings | None = None,
) -> TemporalFreshnessScorer:
    """Get or create singleton TemporalFreshnessScorer instance."""
    global _temporal_scorer_instance
    if _temporal_scorer_instance is None or settings is not None:
        _temporal_scorer_instance = TemporalFreshnessScorer(settings=settings)
    return _temporal_scorer_instance


def reset_temporal_freshness_scorer() -> None:
    """Reset singleton instance for test isolation."""
    global _temporal_scorer_instance
    _temporal_scorer_instance = None
