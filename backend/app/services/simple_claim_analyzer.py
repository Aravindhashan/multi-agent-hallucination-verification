"""Heuristic implementation of claim analysis and verification planning."""

import re

from app.config import settings
from app.schemas import ClaimAnalysisResult, PlanningAblationLog
from app.services.claim_analyzer import BaseClaimAnalyzer

_YEAR_OR_MONTH_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)
_CAPITALIZED_WORD_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")
_NUMBER_PATTERN = re.compile(r"\d")
_CHALLENGE_TERMS = {
    "only",
    "never",
    "always",
    "best",
    "worst",
    "first",
    "last",
    "most",
    "least",
    "largest",
    "smallest",
}


class SimpleClaimAnalyzer(BaseClaimAnalyzer):
    """Estimate verification effort using transparent lexical heuristics.

    These heuristics are intentionally lightweight and do not understand claim
    semantics, domain risk, source availability, or contextual ambiguity. A
    later planner can replace this class behind ``BaseClaimAnalyzer``.
    """

    def analyze(self, claim_text: str, atomic_claim: object = None) -> ClaimAnalysisResult:
        """Analyze claim features and return a bounded score and action plan.

        When an atomic_claim (or dictionary with anchors) is provided, uses
        precise linguistic anchors (temporal, numerical, entities, qualifiers);
        otherwise falls back to lexical heuristics.
        """

        words = claim_text.split()
        word_count = len(words)

        # Extract anchors from atomic_claim if available
        temporal_anchors: list[str] = []
        numerical_values: list[str] = []
        spatial_anchors: list[str] = []
        temporal_anchor = None
        numerical_value = None
        spatial_anchor = None
        subject = None
        predicate = None
        obj = None
        qualifiers = []

        if atomic_claim is not None:
            if isinstance(atomic_claim, dict):
                temporal_anchors = list(atomic_claim.get("temporal_anchors") or [])
                numerical_values = list(atomic_claim.get("numerical_values") or [])
                spatial_anchors = list(atomic_claim.get("spatial_anchors") or [])
                temporal_anchor = atomic_claim.get("temporal_anchor") or (temporal_anchors[0] if temporal_anchors else None)
                numerical_value = atomic_claim.get("numerical_value") or (numerical_values[0] if numerical_values else None)
                spatial_anchor = atomic_claim.get("spatial_anchor") or (spatial_anchors[0] if spatial_anchors else None)
                if temporal_anchor and not temporal_anchors:
                    temporal_anchors = [temporal_anchor]
                if numerical_value and not numerical_values:
                    numerical_values = [numerical_value]
                if spatial_anchor and not spatial_anchors:
                    spatial_anchors = [spatial_anchor]
                subject = atomic_claim.get("subject")
                predicate = atomic_claim.get("predicate")
                obj = atomic_claim.get("object")
                qualifiers = list(atomic_claim.get("qualifiers") or [])
            else:
                temporal_anchors = list(getattr(atomic_claim, "temporal_anchors", []) or [])
                numerical_values = list(getattr(atomic_claim, "numerical_values", []) or [])
                spatial_anchors = list(getattr(atomic_claim, "spatial_anchors", []) or [])
                temporal_anchor = getattr(atomic_claim, "temporal_anchor", None) or (temporal_anchors[0] if temporal_anchors else None)
                numerical_value = getattr(atomic_claim, "numerical_value", None) or (numerical_values[0] if numerical_values else None)
                spatial_anchor = getattr(atomic_claim, "spatial_anchor", None) or (spatial_anchors[0] if spatial_anchors else None)
                if temporal_anchor and not temporal_anchors:
                    temporal_anchors = [temporal_anchor]
                if numerical_value and not numerical_values:
                    numerical_values = [numerical_value]
                if spatial_anchor and not spatial_anchors:
                    spatial_anchors = [spatial_anchor]
                subject = getattr(atomic_claim, "subject", None)
                predicate = getattr(atomic_claim, "predicate", None)
                obj = getattr(atomic_claim, "object", None)
                qualifiers = list(getattr(atomic_claim, "qualifiers", []) or [])

        # Determine features with fallback
        contains_date = (
            bool(temporal_anchor or temporal_anchors)
            if (temporal_anchor is not None or temporal_anchors)
            else bool(_YEAR_OR_MONTH_PATTERN.search(claim_text))
        )
        contains_number = (
            bool(numerical_value or numerical_values)
            if (numerical_value is not None or numerical_values)
            else bool(_NUMBER_PATTERN.search(claim_text))
        )
        contains_spatial = bool(spatial_anchor or spatial_anchors)
        contains_qualifiers = bool(qualifiers)
        contains_entity = bool(subject) if subject is not None else bool(_CAPITALIZED_WORD_PATTERN.search(claim_text))

        lower_words = set(re.findall(r"\b\w+\b", claim_text.lower()))
        contains_challenge_term = bool(lower_words & _CHALLENGE_TERMS)

        # Start with a small baseline
        complexity_score = 0.20
        if word_count > 10:
            complexity_score += 0.15
        if contains_number or contains_date:
            complexity_score += 0.25
        if contains_entity:
            complexity_score += 0.20
        if contains_spatial or contains_qualifiers:
            complexity_score += 0.15
        if contains_challenge_term:
            complexity_score += 0.20

        complexity_score = round(min(max(complexity_score, 0.0), 1.0), 2)

        # Deep verification triggers
        requires_deep_verification = (
            complexity_score > settings.CLAIM_COMPLEXITY_THRESHOLD
            or contains_challenge_term
            or len(qualifiers) > 1
        )

        verification_plan = [
            "Query trusted sources for supporting evidence",
            "Search for contradictions or alternate facts",
            "Check evidence independence and possible duplication",
        ]
        if contains_number or contains_date:
            verification_plan.append("Validate numerical or date-specific details")
        if contains_entity:
            verification_plan.append("Verify named entities against authoritative references")
        if contains_spatial:
            verification_plan.append("Verify geographical and spatial constraints against authoritative references")
        if contains_qualifiers:
            verification_plan.append("Test restrictive qualifiers and scope boundaries for exceptions")
        if contains_challenge_term:
            verification_plan.append("Test absolute or comparative wording for exceptions")
        if requires_deep_verification:
            verification_plan.extend(
                [
                    "Perform additional retrieval rounds",
                    "Engage prosecutor agent for deeper challenge",
                ]
            )

        ablation_log = self._generate_ablation_log(
            claim_text=claim_text,
            subject=subject,
            predicate=predicate,
            obj=obj,
            temporal_anchor=temporal_anchor,
            numerical_value=numerical_value,
            spatial_anchor=spatial_anchor,
            qualifiers=qualifiers,
            requires_deep_verification=requires_deep_verification,
            contains_challenge_term=contains_challenge_term,
            temporal_anchors=temporal_anchors,
            numerical_values=numerical_values,
            spatial_anchors=spatial_anchors,
        )

        return ClaimAnalysisResult(
            complexity_score=complexity_score,
            verification_plan=verification_plan,
            requires_deep_verification=requires_deep_verification,
            ablation_log=ablation_log,
        )

    def _generate_ablation_log(
        self,
        claim_text: str,
        subject: str | None,
        predicate: str | None,
        obj: str | None,
        temporal_anchor: str | None,
        numerical_value: str | None,
        spatial_anchor: str | None,
        qualifiers: list[str],
        requires_deep_verification: bool,
        contains_challenge_term: bool,
        temporal_anchors: list[str] | None = None,
        numerical_values: list[str] | None = None,
        spatial_anchors: list[str] | None = None,
    ) -> PlanningAblationLog:
        """Submodule 4.5: Generate structured ablation logging comparing dynamic vs fixed-baseline retrieval."""
        fixed_baseline_strategy = (
            "Naive single-turn retrieval using verbatim claim text with fixed top-k=5 and no anchor specialization."
        )
        fixed_baseline_queries = [claim_text.strip()]

        dynamic_queries: list[str] = []

        # 1. Base anchor-focused query
        core_elements = [x for x in (subject, predicate, obj) if x]
        if core_elements:
            dynamic_queries.append(" ".join(core_elements).strip())
        else:
            dynamic_queries.append(claim_text.strip())

        entity_or_core = subject or " ".join(core_elements) or claim_text

        # 2. Temporal anchor query
        t_anchors = list(temporal_anchors or ([temporal_anchor] if temporal_anchor else []))
        for t in t_anchors:
            if t:
                q = f"{entity_or_core} {t}".strip()
                if q not in dynamic_queries:
                    dynamic_queries.append(q)

        # 3. Numerical anchor query
        n_anchors = list(numerical_values or ([numerical_value] if numerical_value else []))
        for n in n_anchors:
            if n:
                q = f"{entity_or_core} {n}".strip()
                if q not in dynamic_queries:
                    dynamic_queries.append(q)

        # 4. Spatial anchor query
        s_anchors = list(spatial_anchors or ([spatial_anchor] if spatial_anchor else []))
        for s in s_anchors:
            if s:
                q = f"{entity_or_core} {s}".strip()
                if q not in dynamic_queries:
                    dynamic_queries.append(q)

        # 5. Qualifier-scoped query
        if qualifiers:
            entity_or_core = subject or " ".join(core_elements) or claim_text
            dynamic_queries.append(f"{entity_or_core} {' '.join(qualifiers)}".strip())

        # 6. Deep verification adversarial counter-query
        if requires_deep_verification or contains_challenge_term:
            target = subject or claim_text
            dynamic_queries.append(f"{target} counter-evidence contradiction controversy criticism".strip())

        # Deduplicate while preserving order
        seen = set()
        deduped_dynamic_queries = []
        for q in dynamic_queries:
            if q and q not in seen:
                seen.add(q)
                deduped_dynamic_queries.append(q)

        # Dynamic strategy summary
        strategy_parts = ["Adaptive multi-anchor retrieval"]
        if temporal_anchor or numerical_value:
            strategy_parts.append("targeting specific temporal/numerical boundaries")
        if spatial_anchor:
            strategy_parts.append("verifying spatial constraints")
        if qualifiers:
            strategy_parts.append("testing qualifier boundaries")
        if requires_deep_verification or contains_challenge_term:
            strategy_parts.append("and scheduling adversarial prosecutor challenge queries")
        dynamic_strategy = " ".join(strategy_parts) + "."

        # Difference rationale
        rationale_items = []
        if temporal_anchor or numerical_value:
            rationale_items.append("isolates numeric and temporal constraints into targeted queries instead of broad verbatim matching")
        if spatial_anchor:
            rationale_items.append("targets geographic/spatial entity references")
        if qualifiers:
            rationale_items.append(f"explicitly tests restrictive scope ({', '.join(qualifiers)})")
        if requires_deep_verification or contains_challenge_term:
            rationale_items.append("dispatches adversarial counter-queries to detect refutations or controversies")

        if not rationale_items:
            difference_rationale = (
                "Dynamic planning verified simple claim structure; queries align closely with baseline but use normalized entity-predicate anchors."
            )
        else:
            difference_rationale = (
                "Dynamic planning altered retrieval strategy by: " + "; ".join(rationale_items) + "."
            )

        # Empirical lift expected
        lift_items = []
        if temporal_anchor or numerical_value:
            lift_items.append("higher retrieval precision on date and quantity boundaries")
        if requires_deep_verification or contains_challenge_term:
            lift_items.append("reduced confirmation bias and higher recall of refuting evidence through targeted adversarial querying")
        if qualifiers:
            lift_items.append("early detection of over-generalization and qualifier violations")

        if not lift_items:
            empirical_lift_expected = "Minimal difference expected for simple atomic claims without quantitative or scoping qualifiers."
        else:
            empirical_lift_expected = "; ".join(lift_items) + "."

        return PlanningAblationLog(
            dynamic_strategy=dynamic_strategy,
            dynamic_queries=deduped_dynamic_queries,
            fixed_baseline_strategy=fixed_baseline_strategy,
            fixed_baseline_queries=fixed_baseline_queries,
            difference_rationale=difference_rationale,
            empirical_lift_expected=empirical_lift_expected,
        )
