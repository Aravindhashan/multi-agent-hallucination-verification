"""Service-layer components for the hallucination verification pipeline."""

from app.services.claim_extractor import BaseClaimExtractor
from app.services.claim_models import AtomicClaim, ClaimExtractionResult
from app.services.claim_extraction_service import (
    extract_atomic_claims,
    extract_claims,
    get_claim_extractor,
)
from app.services.llm_claim_extractor import LLMClaimExtractor
from app.services.nlp_claim_extractor import NLPClaimExtractor
from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine, get_nli_engine, reset_nli_engine
from app.services.nli_models import (
    NLIConfigurationError,
    NLIError,
    NLIInferenceError,
    NLIInputError,
    NLILabel,
    NLIModelLoadError,
    NLIResult,
)
from app.services.simple_claim_extractor import SimpleClaimExtractor
from app.services.source_credibility_config import (
    SourceCredibilityConfigError,
    SourceCredibilityError,
    SourceCredibilityInputError,
    SourceCredibilityScoringError,
    SourceCredibilitySettings,
    SourceCredibilityWeights,
)
from app.services.source_credibility_models import (
    CredibilityFactor,
    SourceCredibilityResult,
)
from app.services.source_credibility_scorer import (
    SourceCredibilityScorer,
    get_source_credibility_scorer,
    reset_source_credibility_scorer,
)
from app.services.temporal_freshness_config import (
    TemporalFreshnessConfigError,
    TemporalFreshnessError,
    TemporalFreshnessInputError,
    TemporalFreshnessSettings,
)
from app.services.temporal_freshness_models import (
    TemporalFreshnessResult,
)
from app.services.temporal_freshness_scorer import (
    TemporalFreshnessScorer,
    get_temporal_freshness_scorer,
    reset_temporal_freshness_scorer,
)
from app.services.source_independence_config import (
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DEFAULT_WEAK_RELATION_THRESHOLD,
    DISCLAIMER_PHRASE as SOURCE_INDEPENDENCE_DISCLAIMER_PHRASE,
    SourceIndependenceConfigError,
    SourceIndependenceError,
    SourceIndependenceInputError,
    SourceIndependenceSettings,
)
from app.services.source_independence_models import (
    IndependenceCluster,
    IndependencePairResult,
    IndependenceSetResult,
    RedundancyClass,
)
from app.services.source_independence_scorer import (
    SourceIndependenceDetector,
    compute_char_ngram_jaccard,
    compute_jaccard_similarity,
    extract_token_set,
    get_source_independence_detector,
    normalize_domain,
    normalize_source,
    normalize_text,
    normalize_url,
    reset_source_independence_detector,
    tokenize as tokenize_independence,
)
from app.services.evidence_pollution_config import (
    DEFAULT_MINIMUM_MEANINGFUL_TOKENS as POLLUTION_DEFAULT_MINIMUM_MEANINGFUL_TOKENS,
    DEFAULT_MINIMUM_TOKEN_LENGTH,
    DEFAULT_STOPWORDS,
    DEFAULT_UNGROUNDED_THRESHOLD,
    DEFAULT_WEAK_GROUNDING_THRESHOLD,
    DISCLAIMER_PHRASE as EVIDENCE_POLLUTION_DISCLAIMER_PHRASE,
    EvidencePollutionConfigError,
    EvidencePollutionError,
    EvidencePollutionInputError,
    EvidencePollutionSettings,
)
from app.services.evidence_pollution_models import (
    BatchPollutionResult,
    EvidencePollutionResult,
    PollutionClass,
)
from app.services.evidence_pollution_detector import (
    EvidencePollutionDetector,
    compute_directional_overlap,
    compute_grounding_score,
    extract_meaningful_tokens,
    get_evidence_pollution_detector,
    reset_evidence_pollution_detector,
)
from app.services.session_store import (
    PipelineSession,
    PipelineSessionStore,
    get_session_store,
    reset_session_store,
)
from app.services.evidence_quality_service import (
    EvidenceQualityItemReport,
    EvidenceQualityReport,
    EvidenceQualityService,
    get_evidence_quality_service,
)

__all__ = [
    "EvidenceQualityItemReport",
    "EvidenceQualityReport",
    "EvidenceQualityService",
    "get_evidence_quality_service",
    "PipelineSession",
    "PipelineSessionStore",
    "get_session_store",
    "reset_session_store",
    "AtomicClaim",
    "ClaimExtractionResult",
    "BaseClaimExtractor",
    "NLPClaimExtractor",
    "LLMClaimExtractor",
    "SimpleClaimExtractor",
    "extract_atomic_claims",
    "extract_claims",
    "get_claim_extractor",
    "NLISettings",
    "NLIEngine",
    "get_nli_engine",
    "reset_nli_engine",
    "NLILabel",
    "NLIResult",
    "NLIError",
    "NLIConfigurationError",
    "NLIModelLoadError",
    "NLIInferenceError",
    "NLIInputError",
    "SourceCredibilityError",
    "SourceCredibilityConfigError",
    "SourceCredibilityInputError",
    "SourceCredibilityScoringError",
    "SourceCredibilityWeights",
    "SourceCredibilitySettings",
    "CredibilityFactor",
    "SourceCredibilityResult",
    "SourceCredibilityScorer",
    "get_source_credibility_scorer",
    "reset_source_credibility_scorer",
    "TemporalFreshnessError",
    "TemporalFreshnessConfigError",
    "TemporalFreshnessInputError",
    "TemporalFreshnessSettings",
    "TemporalFreshnessResult",
    "TemporalFreshnessScorer",
    "get_temporal_freshness_scorer",
    "reset_temporal_freshness_scorer",
    "SourceIndependenceError",
    "SourceIndependenceConfigError",
    "SourceIndependenceInputError",
    "SourceIndependenceSettings",
    "DEFAULT_NEAR_DUPLICATE_THRESHOLD",
    "DEFAULT_WEAK_RELATION_THRESHOLD",
    "DEFAULT_MINIMUM_MEANINGFUL_TOKENS",
    "SOURCE_INDEPENDENCE_DISCLAIMER_PHRASE",
    "RedundancyClass",
    "IndependencePairResult",
    "IndependenceCluster",
    "IndependenceSetResult",
    "SourceIndependenceDetector",
    "get_source_independence_detector",
    "reset_source_independence_detector",
    "normalize_text",
    "tokenize_independence",
    "extract_token_set",
    "compute_jaccard_similarity",
    "compute_char_ngram_jaccard",
    "normalize_url",
    "normalize_domain",
    "normalize_source",
    "EvidencePollutionError",
    "EvidencePollutionConfigError",
    "EvidencePollutionInputError",
    "EvidencePollutionSettings",
    "POLLUTION_DEFAULT_MINIMUM_MEANINGFUL_TOKENS",
    "DEFAULT_MINIMUM_TOKEN_LENGTH",
    "DEFAULT_STOPWORDS",
    "DEFAULT_UNGROUNDED_THRESHOLD",
    "DEFAULT_WEAK_GROUNDING_THRESHOLD",
    "EVIDENCE_POLLUTION_DISCLAIMER_PHRASE",
    "PollutionClass",
    "EvidencePollutionResult",
    "BatchPollutionResult",
    "EvidencePollutionDetector",
    "get_evidence_pollution_detector",
    "reset_evidence_pollution_detector",
    "extract_meaningful_tokens",
    "compute_directional_overlap",
    "compute_grounding_score",
]
