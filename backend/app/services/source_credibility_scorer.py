"""Source Credibility and Domain Authority Scorer.

Evaluates evidence document trustworthiness using deterministic metadata heuristics
(domain authority tiering, provenance completeness, and source categorization).
"""

from collections.abc import Mapping, Sequence
import ipaddress
import logging
from typing import Any
from urllib.parse import urlsplit

from app.services.source_credibility_config import (
    SourceCredibilityInputError,
    SourceCredibilityScoringError,
    SourceCredibilitySettings,
    SourceCredibilityWeights,
)
from app.services.source_credibility_models import (
    CredibilityFactor,
    SourceCredibilityResult,
)

logger = logging.getLogger("hallucination_verifier.source_credibility")


class SourceCredibilityScorer:
    """Stateless, deterministic source credibility and domain authority scoring engine."""

    def __init__(self, settings: SourceCredibilitySettings | None = None) -> None:
        """Initialize scorer with optional customized settings."""
        self.settings = settings if settings is not None else SourceCredibilitySettings()

    @property
    def weights(self) -> SourceCredibilityWeights:
        """Return configured scoring weights."""
        return self.settings.weights

    def normalize_domain(self, raw_value: Any) -> str:
        """Normalize URL or domain string into a clean lowercase hostname without path/port.

        Implements deterministic parsing rules:
        - Strips whitespace.
        - Identifies valid IPv4 and IPv6 addresses.
        - Strips scheme, path, query, fragment, and port via urllib.parse.urlsplit.
        - Strips a leading 'www.' once.
        - Returns 'unknown' on empty, malformed, or unparseable inputs without raising.
        """
        if raw_value is None:
            return "unknown"

        s = str(raw_value).strip()
        if not s:
            return "unknown"

        # Check for bare IP address (v4 or v6)
        try:
            ip = ipaddress.ip_address(s)
            return str(ip)
        except ValueError:
            pass

        # If missing protocol scheme, prepend for parsing unless containing spaces
        if not s.lower().startswith(("http://", "https://", "ftp://")):
            if " " in s:
                return "unknown"
            candidate = f"http://{s}"
        else:
            candidate = s

        try:
            parsed = urlsplit(candidate)
            host = parsed.hostname
            if not host:
                return "unknown"

            host = host.lower()
            if host.startswith("www."):
                host = host[4:]

            if not host or " " in host:
                return "unknown"

            # Check valid host structure: must contain a dot or be localhost or an IP
            if "." not in host and host != "localhost" and ":" not in host:
                return "unknown"

            return host
        except Exception:
            return "unknown"

    def classify_tier(self, normalized_domain: str) -> int:
        """Classify a normalized hostname into an authority tier (1 to 4)."""
        if not normalized_domain or normalized_domain == "unknown":
            return 4

        # Check reserved placeholder suffixes and localhost
        if normalized_domain == "localhost":
            return 4

        for placeholder in self.settings.placeholder_suffixes:
            if normalized_domain == placeholder.lstrip(".") or normalized_domain.endswith(placeholder):
                return 4

        # Bare IP addresses fall to Tier 4
        try:
            ipaddress.ip_address(normalized_domain)
            return 4
        except ValueError:
            pass

        # Check Tier 1 official / educational suffixes
        for suffix in self.settings.tier1_suffixes:
            if normalized_domain.endswith(suffix):
                return 1

        # Check Tier 2 curated established domains
        for tier2_dom in self.settings.tier2_domains:
            if normalized_domain == tier2_dom or normalized_domain.endswith(f".{tier2_dom}"):
                return 2

        # Check Tier 3 curated general domains
        for tier3_dom in self.settings.tier3_domains:
            if normalized_domain == tier3_dom or normalized_domain.endswith(f".{tier3_dom}"):
                return 3

        # Default fallback is Tier 4
        return 4

    def _is_field_present(self, value: Any) -> bool:
        """Determine if a metadata field counts as present and non-empty per Section 9.2."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True

    def _extract_field(self, doc: Any, field_name: str) -> Any:
        """Extract a field from an object (attribute) or mapping (key)."""
        if isinstance(doc, Mapping):
            return doc.get(field_name)
        return getattr(doc, field_name, None)

    def score(self, document: Any) -> SourceCredibilityResult:
        """Compute source credibility score and structured explanation for an evidence document.

        Args:
            document: CorpusDocument, RetrievalResult, or plain Mapping[str, Any].

        Returns:
            SourceCredibilityResult containing normalized metrics and explanation.
        """
        if document is None or not (isinstance(document, Mapping) or hasattr(document, "__dict__") or hasattr(document, "__dataclass_fields__")):
            raise SourceCredibilityInputError(f"Unsupported document input type: {type(document).__name__}.")

        # 1. Extract Document Identifier
        doc_id = (
            self._extract_field(document, "document_id")
            or self._extract_field(document, "id")
            or "unknown_doc"
        )
        doc_id_str = str(doc_id).strip()
        if not doc_id_str:
            doc_id_str = "unknown_doc"

        # 2. Extract and Normalize Domain
        raw_url = self._extract_field(document, "source_url") or self._extract_field(document, "url")
        raw_domain = self._extract_field(document, "domain")

        candidate_for_domain = raw_url if self._is_field_present(raw_url) else raw_domain
        normalized_domain = self.normalize_domain(candidate_for_domain)

        # 3. Domain Authority Tiering
        domain_tier = self.classify_tier(normalized_domain)
        domain_score = self.settings.tier_scores.get(domain_tier, 0.25)
        domain_tier_label = self.settings.tier_labels.get(domain_tier, "Unknown / unverified")

        # 4. Provenance Completeness
        considered_fields = self.settings.provenance_fields
        present_count = 0
        for f_name in considered_fields:
            if f_name == "document_id":
                val = self._extract_field(document, "document_id") or self._extract_field(document, "id")
            else:
                val = self._extract_field(document, f_name)

            if self._is_field_present(val):
                present_count += 1

        total_considered = len(considered_fields)
        provenance_score = float(present_count) / float(total_considered) if total_considered > 0 else 0.0

        # 5. Source Type Component & Dynamic Weight Normalization
        raw_source_type = self._extract_field(document, "source_type")
        has_source_type = self._is_field_present(raw_source_type)

        if has_source_type:
            st_key = str(raw_source_type).strip().lower()
            source_type_score = self.settings.source_type_map.get(st_key, 0.25)
            w_dom = self.settings.weights.w_domain
            w_typ = self.settings.weights.w_type
            w_prv = self.settings.weights.w_provenance
            effective_weights = {
                "w_domain": round(w_dom, 4),
                "w_type": round(w_typ, 4),
                "w_provenance": round(w_prv, 4),
            }
            composite = (w_dom * domain_score) + (w_typ * source_type_score) + (w_prv * provenance_score)
        else:
            source_type_score = None
            denom = self.settings.weights.w_domain + self.settings.weights.w_provenance
            if denom <= 0.0:
                raise SourceCredibilityScoringError("Sum of domain and provenance weights must be positive.")
            w_dom = self.settings.weights.w_domain / denom
            w_typ = 0.0
            w_prv = self.settings.weights.w_provenance / denom
            effective_weights = {
                "w_domain": round(w_dom, 4),
                "w_provenance": round(w_prv, 4),
            }
            composite = (w_dom * domain_score) + (w_prv * provenance_score)

        credibility_score = max(0.0, min(1.0, float(composite)))
        credibility_score = round(credibility_score, 4)

        # 6. Build Structured Explanation Factors
        factors: list[CredibilityFactor] = [
            CredibilityFactor(
                name="domain",
                raw_value=round(domain_score, 4),
                weight=round(w_dom, 4),
                contribution=round(w_dom * domain_score, 4),
                note=f"Tier {domain_tier} ({domain_tier_label}): {normalized_domain}",
            ),
            CredibilityFactor(
                name="provenance",
                raw_value=round(provenance_score, 4),
                weight=round(w_prv, 4),
                contribution=round(w_prv * provenance_score, 4),
                note=f"{present_count}/{total_considered} metadata fields present",
            ),
        ]

        if source_type_score is not None:
            factors.append(
                CredibilityFactor(
                    name="source_type",
                    raw_value=round(source_type_score, 4),
                    weight=round(w_typ, 4),
                    contribution=round(w_typ * source_type_score, 4),
                    note=f"Source type '{raw_source_type}'",
                )
            )

        # 7. Build Deterministic Explanation
        if has_source_type:
            st_label_str = str(raw_source_type).strip()
            st_score_str = f"{source_type_score:.2f}"
        else:
            st_label_str = "None (not available in schema)"
            st_score_str = "N/A"

        if domain_tier == 1:
            reason_sentence = "High-authority public sector or educational institution with verified domain credentials."
        elif domain_tier == 2:
            reason_sentence = "Established publication or recognized scholarly repository with strong institutional credibility."
        elif domain_tier == 3:
            reason_sentence = "General commercial, news, or informational domain subject to standard editorial standards."
        else:
            reason_sentence = "Unknown, unverified, or placeholder domain; assigned conservative baseline authority."

        explanation = (
            f"Domain: {normalized_domain}\n"
            f"Domain tier: {domain_tier_label}\n"
            f"Domain score: {domain_score:.2f}\n\n"
            f"Provenance completeness: {provenance_score:.2f}\n\n"
            f"Source type: {st_label_str}\n"
            f"Source type score: {st_score_str}\n\n"
            f"Final credibility: {credibility_score:.2f}\n\n"
            f"Reason:\n{reason_sentence}\n\n"
            f"Note: {self.settings.disclaimer}"
        )

        return SourceCredibilityResult(
            document_id=doc_id_str,
            normalized_domain=normalized_domain,
            domain_tier=domain_tier,
            domain_tier_label=domain_tier_label,
            domain_score=round(domain_score, 4),
            provenance_score=round(provenance_score, 4),
            source_type_score=round(source_type_score, 4) if source_type_score is not None else None,
            credibility_score=credibility_score,
            weights=effective_weights,
            factors=factors,
            explanation=explanation,
            model_version=self.settings.model_version,
        )

    def score_batch(self, documents: Sequence[Any]) -> list[SourceCredibilityResult]:
        """Score a collection of documents preserving input order."""
        if not documents:
            return []
        return [self.score(doc) for doc in documents]


# ---------------------------------------------------------------------------
# Singleton Accessor & Reset
# ---------------------------------------------------------------------------

_scorer_instance: SourceCredibilityScorer | None = None


def get_source_credibility_scorer(
    settings: SourceCredibilitySettings | None = None,
) -> SourceCredibilityScorer:
    """Get or create singleton SourceCredibilityScorer instance."""
    global _scorer_instance
    if _scorer_instance is None or settings is not None:
        _scorer_instance = SourceCredibilityScorer(settings=settings)
    return _scorer_instance


def reset_source_credibility_scorer() -> None:
    """Reset singleton instance for test isolation."""
    global _scorer_instance
    _scorer_instance = None
