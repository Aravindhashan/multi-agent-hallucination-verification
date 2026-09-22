"""Pydantic request and response models used by the API."""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    """Represent and validate a query submitted by a user."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question or statement to enter into the pipeline.",
        examples=["What is the capital of France?"],
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional caller-supplied query identifier to correlate across pipeline stages.",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Trim surrounding whitespace and reject whitespace-only queries."""

        normalized_query = value.strip()
        if not normalized_query:
            raise ValueError(
                "Query must contain at least one non-whitespace character."
            )
        return normalized_query


class QueryResponse(BaseModel):
    """Represent the accepted query and its request-tracking metadata."""

    query: str = Field(description="The normalized query accepted by the API.")
    query_id: UUID = Field(description="A unique identifier for this query.")
    timestamp: datetime = Field(
        description="The UTC date and time at which the query was accepted."
    )


class LLMGeneratedResponse(BaseModel):
    """Represent a base LLM response and its generation metadata.

    The input query is intentionally not duplicated in this model. ``query_id``
    provides the correlation point, while the service returns only the text and
    metadata needed by downstream claim-extraction modules.
    """

    query_id: UUID = Field(description="Identifier assigned to this query run.")
    response_text: str = Field(
        min_length=1,
        description="Non-empty text produced by the configured LLM client.",
    )
    model_name: str = Field(
        min_length=1,
        description="Identifier of the model that generated the response.",
    )
    generated_at: AwareDatetime = Field(
        description="Timezone-aware UTC generation timestamp."
    )


def _sync_anchors_dict(data: Any) -> Any:
    """Enforce scalar == plural[0] synchronization invariant on dictionary inputs."""
    if isinstance(data, dict):
        # Sync temporal
        t_list = data.get("temporal_anchors")
        t_scalar = data.get("temporal_anchor")
        if t_list and not t_scalar:
            data["temporal_anchor"] = t_list[0]
        elif t_scalar and not t_list:
            data["temporal_anchors"] = [t_scalar]
        elif t_list and t_scalar and t_scalar != t_list[0]:
            data["temporal_anchor"] = t_list[0]

        # Sync numerical
        n_list = data.get("numerical_values")
        n_scalar = data.get("numerical_value")
        if n_list and not n_scalar:
            data["numerical_value"] = n_list[0]
        elif n_scalar and not n_list:
            data["numerical_values"] = [n_scalar]
        elif n_list and n_scalar and n_scalar != n_list[0]:
            data["numerical_value"] = n_list[0]

        # Sync spatial
        s_list = data.get("spatial_anchors")
        s_scalar = data.get("spatial_anchor")
        if s_list and not s_scalar:
            data["spatial_anchor"] = s_list[0]
        elif s_scalar and not s_list:
            data["spatial_anchors"] = [s_scalar]
        elif s_list and s_scalar and s_scalar != s_list[0]:
            data["spatial_anchor"] = s_list[0]

    return data


class Claim(BaseModel):
    """Represent one ordered candidate claim extracted from an LLM response."""

    claim_id: UUID | str = Field(description="Unique identifier for this claim.")
    text: str = Field(
        min_length=1,
        description="Non-empty candidate claim text.",
    )
    order: int = Field(
        ge=0,
        description="Zero-based position among the extracted claims.",
    )
    subject: str | None = Field(
        default=None,
        description="Extracted subject entity of the claim.",
    )
    predicate: str | None = Field(
        default=None,
        description="Extracted action, relation, or state predicate.",
    )
    object: str | None = Field(
        default=None,
        description="Extracted object entity, complement, or target value.",
    )
    temporal_anchor: str | None = Field(
        default=None,
        description="Deprecated: use temporal_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    numerical_value: str | None = Field(
        default=None,
        description="Deprecated: use numerical_values as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    spatial_anchor: str | None = Field(
        default=None,
        description="Deprecated: use spatial_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    temporal_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted dates, years, or temporal constraints.",
    )
    numerical_values: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted numerical quantities, measurements, or metrics.",
    )
    spatial_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted spatial locations or geographical/astronomical entities.",
    )
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Extracted factual qualifiers, restrictive conditions, or modifiers.",
    )
    original_clause: str | None = Field(
        default=None,
        description="Original un-revised clause text prior to self-containment revision.",
    )
    is_self_contained: bool = Field(
        default=True,
        description="Whether this claim has been resolved to be verifiable in isolation.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_anchor_invariants(cls, data: Any) -> Any:
        """Enforce scalar == plural[0] synchronization invariant."""
        return _sync_anchors_dict(data)


class ClaimExtractionRequest(QueryRequest):
    """Represent the query and generated response submitted for extraction.

    Inheriting from ``QueryRequest`` preserves Module 1's query normalization
    and length constraints without duplicating validation logic.
    """

    response_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Generated LLM response to decompose into claims.",
    )
    method: str = Field(
        default="nlp",
        description="Extraction method to use ('nlp' or 'llm').",
    )

    @field_validator("response_text")
    @classmethod
    def normalize_response_text(cls, value: str) -> str:
        """Trim response whitespace and reject whitespace-only text."""

        normalized_response = value.strip()
        if not normalized_response:
            raise ValueError(
                "Response text must contain at least one non-whitespace character."
            )
        return normalized_response


class DecompositionMetrics(BaseModel):
    """Quality metrics evaluating the factual decomposition of text into atomic claims."""

    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall of key content tokens and named entities from the source text (did extraction drop info?).",
    )
    atomicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion of extracted claims that are truly indivisible single propositions.",
    )
    entailment_preservation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Degree to which the extracted claims faithfully preserve source facts without distortion.",
    )
    dropped_entities: list[str] = Field(
        default_factory=list,
        description="Salient named entities or quantities present in source text but omitted from claims.",
    )
    is_high_quality: bool = Field(
        description="True if coverage >= 0.70 and atomicity >= 0.80.",
    )


class ClaimExtractionResponse(BaseModel):
    """Represent the ordered claims produced for one extraction operation."""

    query_id: UUID = Field(
        description="Identifier assigned to this standalone extraction run."
    )
    claims: list[Claim] = Field(
        description="Candidate claims in their original relative order."
    )
    decomposition_metrics: DecompositionMetrics | None = Field(
        default=None,
        description="Automated quality evaluation metrics for the decomposition.",
    )
    extracted_at: AwareDatetime = Field(
        description="Timezone-aware UTC extraction timestamp."
    )


class PlanningAblationLog(BaseModel):
    """Submodule 4.5: Planning ablation comparison between dynamic strategy and fixed baseline."""

    dynamic_strategy: str = Field(
        description="The dynamically generated query/retrieval strategy based on claim complexity and anchors.",
    )
    dynamic_queries: list[str] = Field(
        default_factory=list,
        description="Targeted search queries generated dynamically from extracted entities, anchors, and qualifiers.",
    )
    fixed_baseline_strategy: str = Field(
        default="Naive single-turn retrieval using verbatim claim text with fixed top-k=5 and no anchor specialization.",
        description="The fixed non-adaptive baseline query strategy.",
    )
    fixed_baseline_queries: list[str] = Field(
        default_factory=list,
        description="Fixed baseline queries (verbatim claim text).",
    )
    difference_rationale: str = Field(
        description="Explanation of why dynamic planning altered the retrieval strategy compared to the naive baseline.",
    )
    empirical_lift_expected: str = Field(
        description="Hypothesized empirical benefit of the dynamic strategy (e.g. higher precision on dates/entities, deeper challenge).",
    )


class ClaimAnalysisResult(BaseModel):
    """Represent provider-independent claim analysis output."""

    complexity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized estimate of the claim's verification effort.",
    )
    verification_plan: list[str] = Field(
        min_length=1,
        description="Ordered verification actions recommended for the claim.",
    )
    requires_deep_verification: bool = Field(
        description="Whether additional retrieval and challenge work is needed."
    )
    ablation_log: PlanningAblationLog | None = Field(
        default=None,
        description="Submodule 4.5 ablation comparison logging dynamic vs fixed baseline query strategies.",
    )


class ClaimAnalysisRequest(BaseModel):
    """Represent one extracted claim submitted for complexity analysis."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The candidate factual claim to analyze.",
    )
    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional Module 3 claim identifier to preserve.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier to correlate in session store.",
    )
    subject: str | None = Field(default=None, description="Optional pre-extracted subject entity.")
    predicate: str | None = Field(default=None, description="Optional pre-extracted predicate.")
    object: str | None = Field(default=None, description="Optional pre-extracted object entity.")
    temporal_anchor: str | None = Field(
        default=None,
        description="Deprecated: use temporal_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    numerical_value: str | None = Field(
        default=None,
        description="Deprecated: use numerical_values as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    spatial_anchor: str | None = Field(
        default=None,
        description="Deprecated: use spatial_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    temporal_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all pre-extracted dates, years, or temporal constraints.",
    )
    numerical_values: list[str] = Field(
        default_factory=list,
        description="Canonical list of all pre-extracted numerical quantities, measurements, or metrics.",
    )
    spatial_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all pre-extracted spatial locations or geographical/astronomical entities.",
    )
    qualifiers: list[str] = Field(default_factory=list, description="Optional pre-extracted qualifiers.")

    @model_validator(mode="before")
    @classmethod
    def resolve_claim_text_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "claim_text" not in data and "text" in data:
                data["claim_text"] = data["text"]
        return data

    @model_validator(mode="before")
    @classmethod
    def sync_anchor_invariants(cls, data: Any) -> Any:
        """Enforce scalar == plural[0] synchronization invariant."""
        return _sync_anchors_dict(data)

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        """Trim surrounding whitespace and reject whitespace-only claims."""

        normalized_claim = value.strip()
        if not normalized_claim:
            raise ValueError(
                "Claim text must contain at least one non-whitespace character."
            )
        return normalized_claim


class ClaimAnalysisResponse(BaseModel):
    """Represent a claim's complexity assessment and verification plan."""

    claim_id: UUID | str = Field(description="Identifier for the analyzed claim.")
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier correlated in session store.",
    )
    claim_text: str = Field(
        min_length=1,
        description="Normalized claim text used by the analyzer.",
    )
    complexity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized estimate of the claim's verification effort.",
    )
    verification_plan: list[str] = Field(
        min_length=1,
        description="Ordered verification actions recommended for the claim.",
    )
    requires_deep_verification: bool = Field(
        description="Whether additional retrieval and challenge work is needed."
    )
    ablation_log: PlanningAblationLog | None = Field(
        default=None,
        description="Ablation log comparing dynamic planning against fixed baseline for empirical evaluation.",
    )
    analyzed_at: AwareDatetime = Field(
        description="Timezone-aware UTC analysis timestamp."
    )


class EvidenceItem(BaseModel):
    """Represent one retrieved evidence passage and its source metadata."""

    evidence_id: str | UUID = Field(description="Unique identifier for this evidence.")
    text: str = Field(
        min_length=1,
        description="Evidence passage returned by the retriever.",
    )
    source_url: str | None = Field(
        default=None,
        description="Optional source URL for the evidence passage.",
    )
    domain: str | None = Field(
        default=None,
        description="Optional source domain used in independence analysis.",
    )
    source_title: str | None = Field(
        default=None,
        description="Optional human-readable source title.",
    )
    retrieval_method: str = Field(
        default="hybrid",
        min_length=1,
        description="Method or provider that retrieved the evidence.",
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Normalized estimated relevance to the claim.",
    )


class DefenderResult(BaseModel):
    """Represent provider-independent output from a Defender agent."""

    claim_text: str = Field(
        min_length=1,
        description="Normalized claim for which support was gathered.",
    )
    supporting_evidence: list[EvidenceItem] = Field(
        description="Evidence passages intended to support the claim."
    )
    query_used: str = Field(
        min_length=1,
        description="Retrieval query issued by the Defender.",
    )
    defended_at: AwareDatetime = Field(
        description="Timezone-aware UTC timestamp for Defender completion."
    )


class DefenderRequest(BaseModel):
    """Represent one claim submitted to the supporting-evidence agent."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The candidate factual claim to defend.",
    )
    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional Module 3 claim identifier to preserve.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier to correlate in session store.",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        """Trim surrounding whitespace and reject whitespace-only claims."""

        normalized_claim = value.strip()
        if not normalized_claim:
            raise ValueError(
                "Claim text must contain at least one non-whitespace character."
            )
        return normalized_claim


class DefenderResponse(BaseModel):
    """Represent supporting evidence gathered for one claim."""

    claim_id: UUID | str = Field(description="Identifier for the defended claim.")
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier correlated in session store.",
    )
    claim_text: str = Field(
        min_length=1,
        description="Normalized claim for which support was gathered.",
    )
    supporting_evidence: list[EvidenceItem] = Field(
        description="Evidence passages intended to support the claim."
    )
    query_used: str = Field(
        min_length=1,
        description="Retrieval query issued by the Defender.",
    )
    defended_at: AwareDatetime = Field(
        description="Timezone-aware UTC timestamp for Defender completion."
    )


class ProsecutorResult(BaseModel):
    """Represent provider-independent output from a Prosecutor agent."""

    claim_text: str = Field(
        min_length=1,
        description="Normalized claim for which challenges were gathered.",
    )
    contradicting_evidence: list[EvidenceItem] = Field(
        description="Evidence passages intended to challenge the claim."
    )
    query_used: str = Field(
        min_length=1,
        description="Retrieval query issued by the Prosecutor.",
    )
    challenged_at: AwareDatetime = Field(
        description="Timezone-aware UTC timestamp for Prosecutor completion."
    )


class ProsecutorRequest(BaseModel):
    """Represent one claim submitted to the contradicting-evidence agent."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The candidate factual claim to challenge.",
    )
    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional Module 3 claim identifier to preserve.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier to correlate in session store.",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        """Trim surrounding whitespace and reject whitespace-only claims."""

        normalized_claim = value.strip()
        if not normalized_claim:
            raise ValueError(
                "Claim text must contain at least one non-whitespace character."
            )
        return normalized_claim


class ProsecutorResponse(BaseModel):
    """Represent contradicting evidence gathered for one claim."""

    claim_id: UUID | str = Field(description="Identifier for the challenged claim.")
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier correlated in session store.",
    )
    claim_text: str = Field(
        min_length=1,
        description="Normalized claim for which challenges were gathered.",
    )
    contradicting_evidence: list[EvidenceItem] = Field(
        description="Evidence passages intended to challenge the claim."
    )
    query_used: str = Field(
        min_length=1,
        description="Retrieval query issued by the Prosecutor.",
    )
    challenged_at: AwareDatetime = Field(
        description="Timezone-aware UTC timestamp for Prosecutor completion."
    )


class AtomicClaim(BaseModel):
    """Represent an independently verifiable atomic factual claim."""

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Deterministic unique identifier for this claim (e.g., 'c1', 'c2').",
    )
    claim_text: str = Field(
        ...,
        min_length=1,
        description="Normalized atomic factual claim text.",
    )
    subject: str | None = Field(
        default=None,
        description="Extracted subject entity of the claim.",
    )
    predicate: str | None = Field(
        default=None,
        description="Extracted action, relation, or state predicate.",
    )
    object: str | None = Field(
        default=None,
        description="Extracted object entity, complement, or target value.",
    )
    temporal_anchor: str | None = Field(
        default=None,
        description="Deprecated: use temporal_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    numerical_value: str | None = Field(
        default=None,
        description="Deprecated: use numerical_values as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    spatial_anchor: str | None = Field(
        default=None,
        description="Deprecated: use spatial_anchors as canonical source of truth.",
        json_schema_extra={"deprecated": True},
    )
    temporal_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted dates, years, or temporal constraints.",
    )
    numerical_values: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted numerical quantities, measurements, or metrics.",
    )
    spatial_anchors: list[str] = Field(
        default_factory=list,
        description="Canonical list of all extracted spatial locations or geographical/astronomical entities.",
    )
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Extracted factual qualifiers, restrictive conditions, or modifiers.",
    )
    original_clause: str | None = Field(
        default=None,
        description="Original un-revised clause text prior to self-containment revision.",
    )
    is_self_contained: bool = Field(
        default=True,
        description="Whether this claim has been resolved to be verifiable in isolation.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_anchor_invariants(cls, data: Any) -> Any:
        """Enforce scalar == plural[0] synchronization invariant."""
        return _sync_anchors_dict(data)

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        """Reject empty or whitespace-only claim IDs."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("claim_id must contain at least one non-whitespace character.")
        return trimmed

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, value: str) -> str:
        """Reject empty or whitespace-only claim text."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("claim_text must contain at least one non-whitespace character.")
        return trimmed


class ClaimExtractionResult(BaseModel):
    """Represent the result of decomposing text into atomic claims."""

    claims: list[AtomicClaim] = Field(
        default_factory=list,
        description="Ordered list of extracted atomic claims.",
    )
    extraction_method: str = Field(
        default="nlp",
        description="Method used for extraction (e.g., 'nlp', 'llm', 'hybrid').",
    )
    raw_response_text: str = Field(
        ...,
        description="The original unparsed response text.",
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware UTC timestamp of the extraction operation.",
    )


class EvidenceQualityRequest(BaseModel):
    """Represent candidate evidence items submitted for multi-dimensional quality assessment."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The factual claim against which evidence quality is evaluated.",
    )
    evidence_items: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        description="List of retrieved evidence passages to evaluate.",
    )
    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional claim identifier for correlation.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier for session correlation.",
    )
    reference_date: date | None = Field(
        default=None,
        description="Optional reference date for freshness decay (defaults to today).",
    )
    claim_origin_url: str | None = Field(
        default=None,
        description="Optional origin URL of the claim to detect self-referential evidence.",
    )


class EvidenceQualityResponse(BaseModel):
    """Represent composite quality assessment across all candidate evidence items."""

    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional claim identifier for correlation.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier for session correlation.",
    )
    claim_text: str = Field(description="The evaluated factual claim.")
    total_evidence_count: int = Field(description="Total passages evaluated.")
    admissible_count: int = Field(description="Passages passing quality and pollution gates.")
    polluted_count: int = Field(description="Passages rejected due to pollution or duplication.")
    cluster_count: int = Field(description="Redundancy clusters identified via complete linkage.")
    kish_effective_count: float = Field(description="Kish effective sample size discounting redundancy.")
    evaluated_items: list[dict[str, Any]] = Field(description="Per-item quality and defense reports.")
    disclaimer: str = Field(description="Research heuristic boundary notice.")
    evaluated_at: datetime = Field(description="Evaluation timestamp.")


