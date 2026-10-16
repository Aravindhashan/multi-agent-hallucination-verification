"""Pydantic request and response models used by the API."""

from datetime import date, datetime, timezone
from enum import Enum
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


class StructuredChallenge(BaseModel):
    """Represent structured factual challenge comparing claim anchors against contradicting evidence.

    Strictly structured data — contains zero free-text arguments or LLM persuasion prose.
    """

    conflict_type: str = Field(
        ...,
        description="Type of factual challenge: 'temporal', 'numerical', 'spatial', 'predicate', or 'refutation'.",
    )
    claim_anchor: str = Field(
        ...,
        description="Claim anchor or proposition component being challenged.",
    )
    evidence_snippet: str = Field(
        default="",
        description="Passage text snippet supporting the challenge.",
    )
    refutation_cues: list[str] = Field(
        default_factory=list,
        description="Dispute or controversy tokens detected in evidence passage.",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this structured challenge.",
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
    credibility_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional source credibility score.",
    )
    freshness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional temporal freshness score.",
    )
    composite_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Retrieval ranking heuristic score S_retrieval = 0.5*relevance + 0.3*credibility + 0.2*freshness "
            "used for candidate selection (Modules 5-7)."
        ),
    )
    cluster_reliability_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Bayesian epistemic channel reliability W(C_k) = S_intrinsic(e_hat_k) = cred * fresh * (1 - poll) "
            "for cluster representative pooling (Submodule 8.7)."
        ),
    )
    challenge: StructuredChallenge | None = Field(
        default=None,
        description="Optional structured challenge identified against the claim.",
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
    challenges: list[StructuredChallenge] = Field(
        default_factory=list,
        description="Structured factual challenges identified in retrieved evidence.",
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
    challenges: list[StructuredChallenge] = Field(
        default_factory=list,
        description="Structured factual challenges identified in retrieved evidence.",
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
    temporal_anchor: str | None = Field(
        default=None,
        description="Optional temporal anchor for historical freshness evaluation.",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        """Trim whitespace and reject empty or whitespace-only claims."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Claim text must contain at least one non-whitespace character.")
        return normalized



class EvidenceQualityBatchRequest(BaseModel):
    """Represent a batch of claims and evidence items submitted for quality assessment."""

    requests: list[EvidenceQualityRequest] = Field(
        default_factory=list,
        description="List of individual claim-evidence quality requests.",
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


class PooledEvidenceCluster(BaseModel):
    """Represent one redundancy cluster collapsed into its highest-quality representative (Submodule 8.7)."""

    cluster_id: int = Field(description="Unique cluster identifier from complete-linkage clustering.")
    representative_id: str = Field(description="Evidence ID of the selected highest-quality cluster member.")
    representative_text: str = Field(description="Text passage of the selected representative.")
    cluster_size: int = Field(ge=1, description="Total number of passages in this redundancy cluster.")
    member_ids: list[str] = Field(description="IDs of all candidate evidence passages in this cluster.")
    cluster_reliability_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Reliability weight W(C_k) = S_comp(e_hat_k) without corroboration bonus (Reviewer §4).",
    )
    is_admissible: bool = Field(description="True if cluster contains at least one admissible member.")
    admissibility_reason: str = Field(description="Explanation of cluster admissibility decision.")
    domain: str | None = Field(default=None, description="Domain of representative source.")
    source_url: str | None = Field(default=None, description="URL of representative source.")
    credibility_score: float = Field(ge=0.0, le=1.0, description="Representative source credibility score.")
    freshness_score: float = Field(ge=0.0, le=1.0, description="Representative temporal freshness score.")
    pollution_score: float = Field(ge=0.0, le=1.0, description="Representative pollution score.")
    pollution_class: str = Field(default="CLEAN", description="Representative pollution category.")
    directional_grounding_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Representative directional grounding score."
    )
    representative_item: EvidenceItem | dict[str, Any] | None = Field(
        default=None, description="Structured EvidenceItem representation of the winning representative."
    )


class PooledClusterReport(BaseModel):
    """Represent the set-level report of pooled cluster representatives for a claim (Submodule 8.7)."""

    claim_text: str = Field(description="The evaluated factual claim.")
    claim_id: UUID | str | None = Field(default=None, description="Optional claim identifier.")
    query_id: UUID | None = Field(default=None, description="Optional pipeline query identifier.")
    total_raw_evidence_count: int = Field(ge=0, description="Total raw evidence passages evaluated.")
    total_clusters: int = Field(ge=0, description="Total clusters identified via complete linkage.")
    admissible_clusters: int = Field(ge=0, description="Number of clusters with admissible evidence.")
    polluted_clusters: int = Field(ge=0, description="Number of clusters rejected due to pollution/low credibility.")
    kish_effective_count: float = Field(
        ge=0.0, description="Corrected Kish effective sample size n_eff discounting redundancy."
    )
    nli_call_reduction_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage reduction in downstream NLI inference calls achieved by cluster pooling.",
    )
    clusters: list[PooledEvidenceCluster] = Field(
        default_factory=list, description="Per-cluster representative pooling details."
    )
    pooled_representatives: list[EvidenceItem] = Field(
        default_factory=list,
        description="Flat list of admissible representative EvidenceItems ready for downstream Module 9 NLI.",
    )
    disclaimer: str = Field(description="Standard research heuristic boundary notice.")
    pooled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of pooling completion."
    )


class PooledClusterBatchRequest(BaseModel):
    """Batch request for pooling cluster representatives across multiple claims (Submodule 8.7)."""

    requests: list[EvidenceQualityRequest] = Field(
        default_factory=list, description="List of individual claim-evidence requests."
    )


class PooledClusterBatchResponse(BaseModel):
    """Batch response containing pooled cluster reports for each submitted claim (Submodule 8.7)."""

    reports: list[PooledClusterReport] = Field(
        default_factory=list, description="Ordered list of pooled cluster reports matching input sequence."
    )
    total_claims_processed: int = Field(ge=0, description="Total claims processed in this batch.")
    total_raw_evidence_evaluated: int = Field(ge=0, description="Cumulative raw evidence items evaluated.")
    total_pooled_representatives: int = Field(ge=0, description="Cumulative pooled representatives produced.")
    mean_nli_call_reduction_percent: float = Field(
        ge=0.0, le=100.0, description="Average downstream NLI call reduction across the batch."
    )


class PerClaimRecallMetrics(BaseModel):
    """Recall and perspective metrics for a single claim across retrieval configurations."""

    claim_id: str = Field(description="Identifier of the evaluated claim.")
    claim_text: str = Field(description="Claim text evaluated.")
    baseline_evidence_count: int = Field(description="Number of evidence items retrieved by baseline.")
    defender_evidence_count: int = Field(description="Number of evidence items retrieved by defender.")
    prosecutor_evidence_count: int = Field(description="Number of evidence items retrieved by prosecutor.")
    dual_perspective_evidence_count: int = Field(description="Deduplicated union count of defender and prosecutor.")
    jaccard_complementarity: float = Field(
        description="Jaccard similarity between defender and prosecutor evidence sets (target < 0.50)."
    )
    baseline_recall: float = Field(default=0.0, description="Recall against gold/pooled evidence for baseline.")
    defender_recall: float = Field(default=0.0, description="Recall against gold/pooled evidence for defender.")
    prosecutor_recall: float = Field(default=0.0, description="Recall against gold/pooled evidence for prosecutor.")
    dual_recall: float = Field(default=0.0, description="Recall against gold/pooled evidence for dual perspective.")
    contradiction_recall_baseline: float = Field(default=0.0, description="Contradiction evidence recall for baseline.")
    contradiction_recall_prosecutor: float = Field(default=0.0, description="Contradiction evidence recall for prosecutor.")
    contradiction_recall_dual: float = Field(default=0.0, description="Contradiction evidence recall for dual perspective.")
    recall_lift_percent: float = Field(default=0.0, description="Percentage lift in total recall of dual perspective over baseline.")
    unique_evidence_lift_percent: float = Field(default=0.0, description="Percentage lift in unique evidence count over baseline.")
    baseline_recall_at_1: float = Field(default=0.0, description="Recall@1 for baseline retrieval.")
    baseline_recall_at_3: float = Field(default=0.0, description="Recall@3 for baseline retrieval.")
    baseline_recall_at_5: float = Field(default=0.0, description="Recall@5 for baseline retrieval.")
    dual_recall_at_1: float = Field(default=0.0, description="Recall@1 for dual-perspective retrieval.")
    dual_recall_at_3: float = Field(default=0.0, description="Recall@3 for dual-perspective retrieval.")
    dual_recall_at_5: float = Field(default=0.0, description="Recall@5 for dual-perspective retrieval.")


class RecallAblationSummary(BaseModel):
    """Aggregate summary metrics across a batch of evaluated claims."""

    total_claims_evaluated: int = Field(description="Total number of claims evaluated.")
    mean_baseline_recall: float = Field(description="Mean recall for baseline retrieval.")
    mean_defender_recall: float = Field(description="Mean recall for defender retrieval.")
    mean_prosecutor_recall: float = Field(description="Mean recall for prosecutor retrieval.")
    mean_dual_recall: float = Field(description="Mean recall for dual perspective retrieval.")
    mean_jaccard_complementarity: float = Field(
        description="Mean Jaccard similarity between defender and prosecutor evidence sets (target < 0.50)."
    )
    mean_contradiction_recall_baseline: float = Field(default=0.0, description="Mean contradiction recall for baseline.")
    mean_contradiction_recall_prosecutor: float = Field(default=0.0, description="Mean contradiction recall for prosecutor.")
    mean_contradiction_recall_dual: float = Field(default=0.0, description="Mean contradiction recall for dual perspective.")
    mean_recall_lift_percent: float = Field(default=0.0, description="Mean recall lift of dual perspective over baseline.")
    mean_unique_evidence_lift_percent: float = Field(default=0.0, description="Mean unique evidence lift over baseline.")
    mean_baseline_recall_at_1: float = Field(default=0.0, description="Mean Recall@1 for baseline.")
    mean_baseline_recall_at_3: float = Field(default=0.0, description="Mean Recall@3 for baseline.")
    mean_baseline_recall_at_5: float = Field(default=0.0, description="Mean Recall@5 for baseline.")
    mean_dual_recall_at_1: float = Field(default=0.0, description="Mean Recall@1 for dual perspective.")
    mean_dual_recall_at_3: float = Field(default=0.0, description="Mean Recall@3 for dual perspective.")
    mean_dual_recall_at_5: float = Field(default=0.0, description="Mean Recall@5 for dual perspective.")
    per_claim_results: list[PerClaimRecallMetrics] = Field(
        default_factory=list,
        description="Detailed per-claim recall and complementarity metrics.",
    )



class RecallAblationRequest(BaseModel):
    """Request payload for running the perspective recall ablation harness."""

    claims: list[dict[str, Any] | str] = Field(
        ...,
        min_length=1,
        description="List of claims (raw strings or claim objects) to evaluate.",
    )
    gold_evidence_by_claim: dict[str, list[str]] | None = Field(
        default=None,
        description="Optional mapping from claim ID or text to list of gold evidence IDs.",
    )
    gold_contradictions_by_claim: dict[str, list[str]] | None = Field(
        default=None,
        description="Optional mapping from claim ID or text to list of gold contradiction evidence IDs.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of evidence items to retrieve per perspective.",
    )


class DebateRoundRecord(BaseModel):
    """Record of a single round of evidentiary debate between Defender and Prosecutor.

    Contains strictly structured retrieval queries, retrieved evidence, and structured challenges.
    Zero free-text argumentation or conversational monologues are generated.
    """

    round_number: int = Field(ge=1, le=2, description="The debate round index (1 or 2).")
    defender_query: str = Field(description="Query formulated by the Defender for this round.")
    prosecutor_query: str = Field(description="Counter-query formulated by the Prosecutor for this round.")
    supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items retrieved with supporting stance.",
    )
    contradicting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items retrieved with contradicting stance.",
    )
    challenges: list[StructuredChallenge] = Field(
        default_factory=list,
        description="Structured factual challenges identified in this round.",
    )
    jaccard_complementarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity J(D, P) between supporting and contradicting evidence in this round.",
    )


class DebateClaimRequest(BaseModel):
    """Input payload for debating a single claim across structured rounds."""

    claim_text: str = Field(
        ...,
        min_length=1,
        description="The candidate factual claim to verify through multi-round debate.",
    )
    claim_id: UUID | str | None = Field(
        default=None,
        description="Optional claim identifier.",
    )
    query_id: UUID | None = Field(
        default=None,
        description="Optional pipeline query identifier for session correlation.",
    )
    max_rounds: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Maximum number of debate rounds (bounded at 2).",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Claim text must contain at least one non-whitespace character.")
        return normalized


class DebateClaimResponse(BaseModel):
    """Full multi-round evidentiary debate response.

    Note: final_supporting_evidence, final_contradicting_evidence, and incremental_evidence_lift_percent
    reflect raw exact-deduplicated candidate items (exact ID / canonical URL / text hash). These counts
    are pre-Module-8 raw counts. Semantic redundancy clustering, near-duplicate collapse, and Kish's
    effective sample size (n_eff) are calculated in Module 8 (Evidence Redundancy & Independence Clustering).

    Strictly enforces zero conversational argumentation, speech, or free-text rhetoric fields.
    """

    claim_id: UUID | str = Field(description="Unique claim identifier.")
    query_id: UUID | None = Field(default=None, description="Pipeline query identifier.")
    claim_text: str = Field(description="The normalized claim text debated.")
    rounds: list[DebateRoundRecord] = Field(
        default_factory=list,
        description="Chronological record of debate rounds (exactly 2 rounds upon full execution).",
    )
    final_supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Consolidated, exact-deduplicated supporting evidence ranked by composite score.",
    )
    final_contradicting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Consolidated, exact-deduplicated contradicting evidence ranked by composite score.",
    )
    final_challenges: list[StructuredChallenge] = Field(
        default_factory=list,
        description="Consolidated structured challenges identified across rounds.",
    )
    total_rounds: int = Field(default=2, description="Total rounds executed (bounded at 2).")
    early_stopped: bool = False
    jaccard_complementarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final Jaccard complementarity across consolidated evidence pools.",
    )
    incremental_evidence_lift_percent: float = Field(
        default=0.0,
        description="Percentage lift in unique evidence count of Round 2 over Round 1.",
    )
    debated_at: datetime = Field(description="UTC timestamp of debate completion.")


class DebateQualityRequest(BaseModel):
    """Input payload for dual-perspective quality and redundancy evaluation (Submodule 8.8)."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The factual claim debated by Defender and Prosecutor.",
    )
    claim_id: UUID | str | None = Field(default=None, description="Optional claim identifier.")
    query_id: UUID | None = Field(default=None, description="Optional pipeline query identifier.")
    supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items gathered by Defender agent.",
    )
    contradicting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items gathered by Prosecutor agent.",
    )
    reference_date: date | None = Field(
        default=None,
        description="Optional reference date for freshness decay (defaults to today).",
    )
    claim_origin_url: str | None = Field(
        default=None,
        description="Optional origin URL of the claim to detect self-referential evidence.",
    )
    temporal_anchor: str | None = Field(
        default=None,
        description="Optional temporal anchor for historical freshness evaluation.",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Claim text must contain at least one non-whitespace character.")
        return normalized


class DebateQualityResponse(BaseModel):
    """Dual-perspective quality assessment discounting within-stance and cross-stance redundancy (Submodule 8.8)."""

    claim_id: UUID | str | None = Field(default=None, description="Claim identifier.")
    query_id: UUID | None = Field(default=None, description="Pipeline query identifier.")
    claim_text: str = Field(description="The evaluated factual claim.")
    supporting_report: PooledClusterReport = Field(
        description="Quality and cluster pooling report for supporting evidence."
    )
    contradicting_report: PooledClusterReport = Field(
        description="Quality and cluster pooling report for contradicting evidence."
    )
    supporting_n_eff: float = Field(
        ge=0.0,
        description="Kish effective sample size (n_eff,sup) for supporting evidence.",
    )
    contradicting_n_eff: float = Field(
        ge=0.0,
        description="Kish effective sample size (n_eff,con) for contradicting evidence.",
    )
    joint_n_eff: float = Field(
        ge=0.0,
        description="Joint Kish effective sample size (n_eff,joint) across all admissible evidence.",
    )
    cross_perspective_overlap_discount: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportionate reduction in effective sample size caused by cross-perspective redundancy.",
    )
    total_raw_evidence_count: int = Field(
        ge=0,
        description="Total raw candidate evidence items across both perspectives (M_sup + M_con).",
    )
    total_admissible_count: int = Field(
        ge=0,
        description="Total admissible evidence items passing quality and pollution gates.",
    )
    total_pooled_representatives_count: int = Field(
        ge=0,
        description="Total winning cluster representatives ready for downstream NLI (K_sup,adm + K_con,adm).",
    )
    overall_nli_call_reduction_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage reduction in downstream NLI calls achieved by dual cluster pooling.",
    )
    pooled_supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Flat list of admissible supporting cluster representatives ready for Module 9.",
    )
    pooled_contradicting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Flat list of admissible contradicting cluster representatives ready for Module 9.",
    )
    disclaimer: str = Field(description="Standard research heuristic boundary notice.")
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of dual-perspective evaluation completion.",
    )


class DebateQualityBatchRequest(BaseModel):
    """Batch request for evaluating debate evidence quality across multiple claims (Submodule 8.8)."""

    requests: list[DebateQualityRequest] = Field(
        default_factory=list,
        description="List of individual debate quality requests.",
    )


class DebateQualityBatchResponse(BaseModel):
    """Batch response for dual-perspective debate quality evaluations (Submodule 8.8)."""

    responses: list[DebateQualityResponse] = Field(
        default_factory=list,
        description="Ordered list of debate quality responses matching input sequence.",
    )
    total_claims_processed: int = Field(ge=0, description="Total claims processed in this batch.")
    total_raw_evidence_evaluated: int = Field(ge=0, description="Cumulative raw evidence items evaluated.")
    total_pooled_representatives: int = Field(ge=0, description="Cumulative pooled representatives produced.")
    mean_nli_call_reduction_percent: float = Field(
        ge=0.0, le=100.0, description="Average downstream NLI call reduction across the batch."
    )


# ============================================================================
# Module 9: Natural Language Inference (NLI) Adjudication Schemas
# ============================================================================

class AdjudicationVerdict(str, Enum):
    """Calibrated verdict class for a verified claim (Module 9/10)."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class AdjudicationItemResult(BaseModel):
    """Per-evidence item NLI cross-attention prediction and reliability weight."""

    evidence_id: str = Field(description="Identifier of the evaluated evidence item.")
    text: str = Field(description="Evidence passage text.")
    premise_text: str | None = Field(default=None, description="Exact premise text sent to DeBERTa.")
    token_length: int | None = Field(default=None, description="Truncated token length evaluated by DeBERTa.")
    raw_token_length: int | None = Field(default=None, description="Pre-truncation token length.")
    truncated: bool = Field(default=False, description="Whether input was truncated to max_length.")
    domain: str | None = Field(default=None, description="Domain of evidence source.")
    source_url: str | None = Field(default=None, description="Source URL of evidence.")
    source_title: str | None = Field(default=None, description="Title of evidence document.")
    stance: str = Field(default="supporting", description="Evidence stance: supporting or contradicting.")
    cluster_reliability_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Cluster intrinsic reliability weight W(C_k) = C * F * (1 - P).",
    )
    p_entailment: float = Field(ge=0.0, le=1.0, description="Softmax entailment probability.")
    p_contradiction: float = Field(ge=0.0, le=1.0, description="Softmax contradiction probability.")
    p_neutral: float = Field(ge=0.0, le=1.0, description="Softmax neutral probability.")
    label: str = Field(description="Top predicted NLI label (entailment, contradiction, neutral).")
    confidence: float = Field(ge=0.0, le=1.0, description="NLI model confidence.")
    inference_time_ms: float = Field(default=0.0, ge=0.0, description="Inference latency in ms.")


class AdjudicationRequest(BaseModel):
    """Input payload for Module 9 NLI cross-encoder claim adjudication."""

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The candidate factual claim to verify.",
    )
    claim_id: UUID | str | None = Field(default=None, description="Optional claim identifier.")
    query_id: UUID | None = Field(default=None, description="Optional pipeline query identifier.")
    supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Optional pre-retrieved supporting evidence. If empty, runs federated debate.",
    )
    contradicting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Optional pre-retrieved contradicting evidence. If empty, runs federated debate.",
    )
    run_debate: bool = Field(
        default=True,
        description="Whether to run federated multi-turn debate if evidence is not provided.",
    )

    @field_validator("claim_text")
    @classmethod
    def normalize_claim_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Claim text must contain at least one non-whitespace character.")
        return normalized


class AdjudicationResponse(BaseModel):
    """Full cross-encoder NLI adjudication response with Bayesian belief aggregation."""

    claim_id: UUID | str | None = Field(default=None, description="Claim identifier.")
    query_id: UUID | None = Field(default=None, description="Pipeline query identifier.")
    claim_text: str = Field(description="The verified factual claim.")
    verdict: AdjudicationVerdict = Field(description="Adjudicated verdict: SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Calibrated confidence in the verdict.")
    support_probability: float = Field(ge=0.0, le=1.0, description="Bayesian posterior support probability.")
    contradiction_probability: float = Field(ge=0.0, le=1.0, description="Bayesian posterior contradiction probability.")
    neutral_probability: float = Field(ge=0.0, le=1.0, description="Estimated neutral / inconclusive probability.")
    supporting_n_eff: float = Field(ge=0.0, description="Effective sample size of supporting evidence.")
    contradicting_n_eff: float = Field(ge=0.0, description="Effective sample size of contradicting evidence.")
    joint_n_eff: float = Field(ge=0.0, description="Joint effective sample size across all evidence.")
    evaluated_evidence: list[AdjudicationItemResult] = Field(
        default_factory=list,
        description="Per-item NLI evaluations for cluster representatives.",
    )
    total_evidence_evaluated: int = Field(ge=0, description="Total evidence items evaluated by DeBERTa.")
    retrieved_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All distinct internet sources retrieved during debate/search.",
    )
    adjudicated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of adjudication completion.",
    )
    disclaimer: str = Field(
        default="Adjudication based on federated open-source evidence and DeBERTa cross-encoder inference.",
        description="Standard research heuristic boundary notice.",
    )


class AdjudicationBatchRequest(BaseModel):
    """Batch request for adjudicating multiple claims."""

    requests: list[AdjudicationRequest] = Field(
        default_factory=list,
        description="List of individual adjudication requests.",
    )


class AdjudicationBatchResponse(BaseModel):
    """Batch response containing adjudication responses for each submitted claim."""

    responses: list[AdjudicationResponse] = Field(
        default_factory=list,
        description="Ordered list of adjudication responses matching input sequence.",
    )
    total_claims_processed: int = Field(ge=0, description="Total claims processed in this batch.")
    mean_confidence: float = Field(ge=0.0, le=1.0, description="Mean confidence score across batch.")


# ==============================================================================
# Module 10 & 11: Master End-to-End Verification & Calibrated Rollup Schemas
# ==============================================================================


class ResponseVerdict(str, Enum):
    """Calibrated response-level rollup verdict for multi-claim statements (Module 10/11)."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class EvidenceSnippetSummary(BaseModel):
    """Structured representation of supportive or opposing evidence with internet provenance."""

    text: str = Field(description="Evidence passage text.")
    premise_text: str | None = Field(default=None, description="Exact premise text sent to DeBERTa.")
    token_length: int | None = Field(default=None, description="Truncated token length evaluated by DeBERTa.")
    p_entailment: float | None = Field(default=None, description="Softmax entailment probability.")
    p_neutral: float | None = Field(default=None, description="Softmax neutral probability.")
    p_contradiction: float | None = Field(default=None, description="Softmax contradiction probability.")
    source_title: str | None = Field(default=None, description="Title of the source.")
    source_url: str | None = Field(default=None, description="Direct web URL of the evidence.")
    domain: str | None = Field(default=None, description="Domain hostname (e.g. wikipedia.org, nih.gov).")
    stance: str = Field(default="supporting", description="Evidence stance: supporting or opposing.")
    confidence: float | None = Field(default=None, description="Confidence score.")


class ClaimVerificationSummary(BaseModel):
    """Summary of individual claim verification within an end-to-end verification response."""

    claim_id: UUID | str = Field(description="Unique claim identifier.")
    claim_text: str = Field(description="The extracted atomic claim text.")
    order: int = Field(ge=0, description="Relative position of claim in the source text.")
    verdict: AdjudicationVerdict = Field(description="Adjudicated verdict: SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Calibrated confidence score.")
    support_probability: float = Field(ge=0.0, le=1.0, description="Bayesian posterior support probability.")
    contradiction_probability: float = Field(ge=0.0, le=1.0, description="Bayesian posterior contradiction probability.")
    neutral_probability: float = Field(ge=0.0, le=1.0, description="Estimated neutral probability.")
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Semantic complexity score.")
    temporal_anchors: list[str] = Field(default_factory=list, description="Extracted temporal qualifiers.")
    numerical_values: list[str] = Field(default_factory=list, description="Extracted numerical qualifiers.")
    spatial_anchors: list[str] = Field(default_factory=list, description="Extracted spatial qualifiers.")
    supporting_n_eff: float = Field(default=0.0, ge=0.0, description="Kish effective sample size for supporting evidence.")
    contradicting_n_eff: float = Field(default=0.0, ge=0.0, description="Kish effective sample size for contradicting evidence.")
    joint_n_eff: float = Field(default=0.0, ge=0.0, description="Joint Kish effective sample size across all evidence.")
    winning_evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Key representative evidence passages evaluated by DeBERTa.",
    )
    supportive_claims: list[EvidenceSnippetSummary] = Field(
        default_factory=list,
        description="The supportive claims and evidence passages found from search.",
    )
    opposing_claims: list[EvidenceSnippetSummary] = Field(
        default_factory=list,
        description="The opposing / contradicting claims and evidence passages found from search.",
    )
    internet_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The distinct sources and web URLs found after searching the internet.",
    )
    raw_nli_inputs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw premise text, token lengths, and 3-class probabilities sent to DeBERTa.",
    )
    consensus_confirmed: bool = Field(
        default=False,
        description="True if adversarial prosecutor search returned 0 credible contradictions, confirming scientific consensus.",
    )


class VerificationRequest(BaseModel):
    """Input payload for end-to-end text or claim verification (Module 11)."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text passage or factual statement to verify.",
    )
    query_id: UUID | None = Field(default=None, description="Optional pipeline query identifier.")
    evidence_mode: str | None = Field(
        default=None,
        description="Optional evidence retrieval mode override: 'live_web', 'benchmark_scifact', 'benchmark_fever', 'hybrid', or 'toy'.",
    )
    extraction_method: str = Field(
        default="nlp",
        description="Claim extraction strategy: 'nlp' (deterministic rule-based) or 'llm'.",
    )
    max_debate_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum debate rounds between Defender and Prosecutor (default 2).",
    )

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text must contain at least one non-whitespace character.")
        return normalized


class VerificationResponse(BaseModel):
    """End-to-end verification response aggregating all pipeline modules (Module 11)."""

    query_id: UUID = Field(description="Pipeline query identifier for session tracking.")
    text: str = Field(description="The source text or statement verified.")
    overall_verdict: ResponseVerdict = Field(
        description="Calibrated response-level rollup verdict: SUPPORTED, PARTIALLY_SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.",
    )
    verdict_category: str | None = Field(
        default=None,
        description="Fine-grained epistemic verdict category: 'SUPPORTED', 'PARTIALLY_VERIFIED_COVERAGE_GAP', 'PARTIALLY_SUPPORTED_CONTRADICTED', 'REFUTED', or 'NOT_ENOUGH_INFO'.",
    )
    hallucination_index: float = Field(
        ge=0.0,
        le=1.0,
        description="Global factual hallucination risk index H (H_refuted) in [0.0, 1.0]. 0.0 means 0% refuted/hallucinated, 1.0 means fully refuted fabrication.",
    )
    unverified_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Epistemic uncertainty / coverage gap index U (U_unverified) in [0.0, 1.0]. 0.0 means complete decisive evidence, 1.0 means completely ungrounded/inconclusive evidence.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate calibrated confidence score across verified claims.",
    )
    claim_count: int = Field(ge=0, description="Total atomic claims extracted and evaluated.")
    supported_count: int = Field(ge=0, description="Number of atomic claims verified as SUPPORTED.")
    refuted_count: int = Field(ge=0, description="Number of atomic claims verified as REFUTED.")
    unverified_count: int = Field(ge=0, description="Number of atomic claims deemed NOT_ENOUGH_INFO.")
    claims: list[ClaimVerificationSummary] = Field(
        default_factory=list,
        description="Detailed verification summaries for each atomic claim.",
    )
    decomposition_metrics: DecompositionMetrics | None = Field(
        default=None,
        description="Decomposition quality evaluation metrics (coverage, atomicity, entailment).",
    )
    total_evidence_evaluated: int = Field(ge=0, description="Total evidence items evaluated across all claims.")
    aggregate_joint_n_eff: float = Field(
        ge=0.0,
        description="Aggregate joint Kish effective sample size across all claims.",
    )
    evidence_mode_used: str = Field(description="Evidence mode used for retrieval.")
    execution_time_ms: float = Field(ge=0.0, description="End-to-end execution time in milliseconds.")
    internet_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Aggregated distinct internet sources retrieved across all claims.",
    )
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of verification completion.",
    )
    disclaimer: str = Field(
        default="Verification based on 100% free open-source federated evidence engine and DeBERTa cross-encoder inference.",
        description="Standard research heuristic boundary notice.",
    )


class VerificationBatchRequest(BaseModel):
    """Batch payload for verifying multiple text passages."""

    requests: list[VerificationRequest] = Field(
        default_factory=list,
        description="List of individual verification requests.",
    )


class VerificationBatchResponse(BaseModel):
    """Batch response for verifying multiple text passages."""

    responses: list[VerificationResponse] = Field(
        default_factory=list,
        description="Ordered list of verification responses matching input sequence.",
    )
    total_processed: int = Field(ge=0, description="Total text items processed in this batch.")
    mean_hallucination_index: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean factual hallucination index (H_refuted) across batch.",
    )
    mean_unverified_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean epistemic uncertainty / coverage gap index (U_unverified) across batch.",
    )
    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean confidence score across batch.",
    )






