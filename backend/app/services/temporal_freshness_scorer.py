"""Temporal Freshness and Decay Scorer.

Computes a deterministic evidence freshness score based on exponential decay
relative to an explicitly provided reference date, with anchor-aware historical lookback
and a continuous upper-envelope transition to avoid boundary discontinuities.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import logging
import re
from typing import Any

from app.services.temporal_freshness_config import (
    TemporalFreshnessInputError,
    TemporalFreshnessSettings,
)
from app.services.temporal_freshness_models import TemporalFreshnessResult

logger = logging.getLogger("hallucination_verifier.temporal_freshness")


class TemporalFreshnessScorer:
    """Stateless, deterministic temporal freshness scorer using exponential decay and anchor awareness.

    Formulas:
        Contemporary decay:
            F_contemporary = 2 ** (-age_days / half_life_days)

        Historical anchor-aware evaluation (when temporal anchor is present):
            1. Contemporaneous documentation (delta_t_event <= contemporaneous_window_days):
               F_hist = 1.0 - (delta_t_event / contemporaneous_window_days) * contemporaneous_slope
               (With window=730d, slope=0.05: F >= 0.95 across 2 years)
            2. Retrospective scholarship (delta_t_event > contemporaneous_window_days):
               F_hist = 2 ** (-age_days / historical_half_life_days)
               (Default 10-year / 3650-day half life per Garfield 1972)

        Continuous Upper Envelope:
            F_final = max(F_contemporary, F_hist)

    Note:
        Temporal freshness is an evidence-quality indicator, not a factual truth signal.
        Fresh evidence is not automatically true; older evidence is not automatically false.
    """

    def __init__(self, settings: TemporalFreshnessSettings | None = None) -> None:
        """Initialize scorer with optional custom configuration settings."""
        self.settings = settings if settings is not None else TemporalFreshnessSettings()

    def _parse_temporal_anchor(self, anchor: Any) -> tuple[date | None, int | None]:
        """Parse a temporal anchor into an (event_date, event_year) tuple.

        Returns:
            (event_date, event_year):
            - If full date is parseable: (date, date.year)
            - If only a 4-digit year is found: (date(year, 1, 1), year)
            - If unparseable or None: (None, None)
        """
        if anchor is None:
            return None, None

        if isinstance(anchor, datetime):
            return anchor.date(), anchor.year
        if isinstance(anchor, date):
            return anchor, anchor.year

        if not isinstance(anchor, str):
            anchor_str = str(anchor).strip()
        else:
            anchor_str = anchor.strip()

        if not anchor_str:
            return None, None

        # 1. Try ISO date format: YYYY-MM-DD
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", anchor_str)
        if iso_match:
            try:
                d = date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
                return d, d.year
            except ValueError:
                pass

        # 2. Try Month DD, YYYY inside string
        date_pattern = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b",
            anchor_str,
            re.IGNORECASE,
        )
        if date_pattern:
            month_str, day_str, year_str = date_pattern.groups()
            for m_fmt in ("%B %d %Y", "%b %d %Y"):
                try:
                    dt = datetime.strptime(f"{month_str.capitalize()} {day_str} {year_str}", m_fmt)
                    return dt.date(), dt.year
                except ValueError:
                    continue

        # 3. Try standard formats
        for fmt in (
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ):
            try:
                dt = datetime.strptime(anchor_str, fmt)
                return dt.date(), dt.year
            except ValueError:
                pass

        # 4. Extract 4-digit year (1000 - 2099)
        year_match = re.search(r"\b(1\d{3}|20\d{2})\b", anchor_str)
        if year_match:
            year = int(year_match.group(1))
            return date(year, 1, 1), year

        return None, None

    def score(
        self,
        published_date: date | None,
        reference_date: date,
        temporal_anchor: str | None = None,
    ) -> TemporalFreshnessResult:
        """Compute temporal freshness score for a publication date against a reference date.

        Args:
            published_date: Document publication date, or None if unavailable in metadata.
            reference_date: Explicit baseline date (e.g. claim extraction date).
            temporal_anchor: Optional anchor string (year, date, or event description).

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
                temporal_anchor=str(temporal_anchor) if temporal_anchor else None,
                is_historical_anchor=False,
                historical_half_life_days=None,
                model_version=self.settings.model_version,
            )

        # Validate published_date
        if not isinstance(published_date, date):
            raise TemporalFreshnessInputError(
                f"published_date must be an instance of datetime.date or None, got {type(published_date).__name__}."
            )
        pub_d: date = published_date.date() if isinstance(published_date, datetime) else published_date

        # Compute calendar days elapsed relative to reference date
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
                temporal_anchor=str(temporal_anchor) if temporal_anchor else None,
                is_historical_anchor=False,
                historical_half_life_days=None,
                model_version=self.settings.model_version,
            )

        # Handle Case A & B: Baseline Contemporary Decay
        age_days = diff_days
        raw_contemporary = 2.0 ** (-age_days / self.settings.half_life_days)
        f_contemporary = max(0.0, min(1.0, float(raw_contemporary)))

        # Anchor-Aware Evaluation
        anchor_date, anchor_year = self._parse_temporal_anchor(temporal_anchor)
        if anchor_date is None:
            # No anchor: standard contemporary decay
            clamped_score = f_contemporary
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
                temporal_anchor=None,
                is_historical_anchor=False,
                historical_half_life_days=None,
                model_version=self.settings.model_version,
            )

        # Anchor is present: evaluate historical classification and upper envelope
        anchor_age_days = (ref_d - anchor_date).days
        anchor_age_years = anchor_age_days / 365.25
        is_historical = anchor_age_years >= self.settings.historical_lookback_years

        # Delta t to event: if pub_d in same year as parsed year, event delta is 0
        if anchor_year is not None and pub_d.year == anchor_year:
            delta_t_event = 0
        else:
            delta_t_event = abs((pub_d - anchor_date).days)

        if delta_t_event <= self.settings.contemporaneous_window_days:
            # Contemporaneous documentation: linear slope F >= 0.95 at window boundary
            ratio = float(delta_t_event) / float(self.settings.contemporaneous_window_days)
            f_hist = 1.0 - ratio * self.settings.contemporaneous_slope
            f_hist = max(0.0, min(1.0, f_hist))
            regime_note = (
                f"contemporaneous documentation (within {self.settings.contemporaneous_window_days} days of event; "
                f"event delta: {delta_t_event} days, score: {f_hist:.4f})"
            )
        else:
            # Retrospective scholarship: 10-year decay
            raw_retro = 2.0 ** (-age_days / self.settings.historical_half_life_days)
            f_hist = max(0.0, min(1.0, float(raw_retro)))
            regime_note = (
                f"retrospective scholarship (decay half-life: {self.settings.historical_half_life_days:.1f} days; "
                f"score: {f_hist:.4f})"
            )

        # Smooth Continuous Upper Envelope
        final_score = max(f_contemporary, f_hist)
        final_score = max(0.0, min(1.0, round(final_score, 4)))

        explanation = (
            f"Evidence published on {pub_d.isoformat()} is {age_days} days old relative to reference date {ref_d.isoformat()}.\n"
            f"Temporal anchor '{temporal_anchor}' parsed as {anchor_date.isoformat()} ({regime_note}).\n"
            f"Contemporary decay: {f_contemporary:.4f}, Historical evaluation: {f_hist:.4f}.\n"
            f"Final freshness score (continuous upper envelope): {final_score:.2f}.\n\n"
            f"Note: {self.settings.disclaimer}"
        )

        return TemporalFreshnessResult(
            freshness_score=final_score,
            published_date=pub_d,
            reference_date=ref_d,
            age_days=age_days,
            half_life_days=self.settings.half_life_days,
            explanation=explanation,
            is_missing_date=False,
            is_future_date=False,
            temporal_anchor=str(temporal_anchor),
            is_historical_anchor=is_historical,
            historical_half_life_days=self.settings.historical_half_life_days,
            model_version=self.settings.model_version,
        )

    def score_document(
        self,
        document: Any,
        reference_date: date,
        temporal_anchor: str | None = None,
    ) -> TemporalFreshnessResult:
        """Convenience method to score an evidence document or mapping object."""
        if document is None or not (
            isinstance(document, Mapping)
            or hasattr(document, "__dict__")
            or hasattr(document, "__dataclass_fields__")
        ):
            raise TemporalFreshnessInputError(f"Unsupported document input type: {type(document).__name__}.")

        if isinstance(document, Mapping):
            pub_date = document.get("published_date")
            doc_anchor = temporal_anchor or document.get("temporal_anchor")
            if doc_anchor is None:
                anchors_list = document.get("temporal_anchors")
                if anchors_list and isinstance(anchors_list, (list, tuple)):
                    doc_anchor = anchors_list[0]
        else:
            pub_date = getattr(document, "published_date", None)
            doc_anchor = temporal_anchor or getattr(document, "temporal_anchor", None)
            if doc_anchor is None:
                anchors_list = getattr(document, "temporal_anchors", None)
                if anchors_list and isinstance(anchors_list, (list, tuple)):
                    doc_anchor = anchors_list[0]

        return self.score(
            published_date=pub_date,
            reference_date=reference_date,
            temporal_anchor=doc_anchor,
        )

    def score_batch(
        self,
        items: Sequence[Any],
        reference_date: date | None = None,
        temporal_anchor: str | None = None,
        batch_size: int = 10,
    ) -> list[TemporalFreshnessResult]:
        """Score a collection of items preserving input order with memory-safe chunked batching.

        Supports:
        - Sequence of (published_date, reference_date) tuples [legacy 2-tuple]
        - Sequence of (published_date, reference_date, temporal_anchor) tuples [3-tuple]
        - Sequence of document mappings/objects with a shared reference_date
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise TemporalFreshnessInputError(
                f"batch_size must be an integer >= 1, got {batch_size!r}"
            )
        if not items:
            return []

        item_list = list(items)
        results: list[TemporalFreshnessResult] = []

        for i in range(0, len(item_list), batch_size):
            chunk = item_list[i : i + batch_size]
            for item in chunk:
                if isinstance(item, (tuple, list)):
                    if len(item) == 2:
                        p_date, r_date = item
                        t_anc = temporal_anchor
                    elif len(item) >= 3:
                        p_date, r_date, t_anc = item[0], item[1], item[2]
                    else:
                        raise TemporalFreshnessInputError(
                            f"Tuple item must have 2 or 3 elements, got {len(item)}."
                        )
                    results.append(
                        self.score(
                            published_date=p_date,
                            reference_date=r_date,
                            temporal_anchor=t_anc,
                        )
                    )
                elif isinstance(item, Mapping) or hasattr(item, "__dict__") or hasattr(item, "__dataclass_fields__"):
                    if reference_date is None:
                        ref_d = (
                            item.get("reference_date")
                            if isinstance(item, Mapping)
                            else getattr(item, "reference_date", None)
                        )
                        if ref_d is None:
                            raise TemporalFreshnessInputError(
                                "reference_date must be provided either as an argument to score_batch or on each document item."
                            )
                    else:
                        ref_d = reference_date
                    results.append(
                        self.score_document(
                            document=item,
                            reference_date=ref_d,
                            temporal_anchor=temporal_anchor,
                        )
                    )
                else:
                    raise TemporalFreshnessInputError(
                        f"Unsupported item type in score_batch: {type(item).__name__}."
                    )

        return results


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
