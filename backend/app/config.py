"""Central configuration for the Hallucination Verification API."""

from dataclasses import dataclass, field
import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BACKEND_DIR / "data"


def _resolve_data_path(env_var: str, filename: str) -> str:
    """Resolve database or corpus path dynamically, prioritizing env vars while guaranteeing existing local files are found."""
    val = os.getenv(env_var)
    if val and val.strip():
        return val.strip()
    target = _DATA_DIR / filename
    if target.exists():
        return str(target)
    return str(target)


@dataclass(frozen=True)
class Settings:
    """Store application metadata and runtime configuration.

    This immutable object keeps configuration imports consistent across the
    application. It dynamically reads environment variables while providing
    safe development defaults.
    """

    APP_NAME: str = "Hallucination Verification API"
    VERSION: str = "0.6.0"
    DEBUG: bool = False

    # Module 2 & Step 1.4 LLM Configuration
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").strip().lower()
    )
    LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    )
    LLM_API_KEY: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    )
    LLM_BASE_URL: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )
    LLM_TEMPERATURE: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    LLM_MAX_OUTPUT_TOKENS: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
    )
    LLM_TIMEOUT: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "30.0"))
    )
    MOCK_MODEL_NAME: str = "mock-llm-v0"

    # Session & Pipeline persistence
    SESSION_DB_PATH: str = field(
        default_factory=lambda: _resolve_data_path("SESSION_DB_PATH", "pipeline_sessions.db")
    )

    # Module 4 / Phase 1 Retrieval settings
    CORPUS_PATH: str = str(_DATA_DIR / "corpus.json")
    INDEX_DIR: str = str(_DATA_DIR / "indexes")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TOP_K: int = 5
    RETRIEVER_TYPE: str = field(
        default_factory=lambda: os.getenv("RETRIEVER_TYPE", "hybrid").strip().lower()
    )

    # Free Web Retrieval & Local SQLite Cache settings (Component 1)
    WEB_CACHE_DB_PATH: str = field(
        default_factory=lambda: _resolve_data_path("WEB_CACHE_DB_PATH", "web_cache.db")
    )
    WEB_CACHE_TTL_SECONDS: float = field(
        default_factory=lambda: float(os.getenv("WEB_CACHE_TTL_SECONDS", "172800.0"))
    )
    WEB_USER_AGENT: str = field(
        default_factory=lambda: os.getenv(
            "WEB_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ).strip()
    )
    WIKIPEDIA_USER_AGENT: str = field(
        default_factory=lambda: os.getenv(
            "WIKIPEDIA_USER_AGENT",
            "HallucinationVerifierBot/1.0 (https://github.com/hallucination-detection; contact@academic-research.org)",
        ).strip()
    )
    WEB_SEARCH_TIMEOUT: float = field(
        default_factory=lambda: float(os.getenv("WEB_SEARCH_TIMEOUT", "5.0"))
    )
    WEB_SEARCH_MAX_RESULTS: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    )

    # Curated Benchmark Corpora settings (Component 2: SciFact & FEVER)
    SCIFACT_CORPUS_PATH: str = field(
        default_factory=lambda: _resolve_data_path("SCIFACT_CORPUS_PATH", "scifact_corpus.jsonl")
    )
    SCIFACT_CLAIMS_PATH: str = field(
        default_factory=lambda: _resolve_data_path("SCIFACT_CLAIMS_PATH", "scifact_claims.jsonl")
    )
    FEVER_CORPUS_PATH: str = field(
        default_factory=lambda: _resolve_data_path("FEVER_CORPUS_PATH", "fever_corpus.jsonl")
    )
    FEVER_CLAIMS_PATH: str = field(
        default_factory=lambda: _resolve_data_path("FEVER_CLAIMS_PATH", "fever_claims.jsonl")
    )

    # Federated Evidence Engine settings (Component 3)
    # Supported modes: 'live_web', 'benchmark_scifact', 'benchmark_fever', 'hybrid', 'toy'
    EVIDENCE_MODE: str = field(
        default_factory=lambda: os.getenv("EVIDENCE_MODE", "hybrid").strip().lower()
    )
    FEDERATED_ESCALATION_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("FEDERATED_ESCALATION_THRESHOLD", "0.50"))
    )

    # Phase 2 NLI Engine settings
    NLI_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("NLI_MODEL_NAME", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli").strip()
    )
    NLI_DEVICE: str = field(
        default_factory=lambda: os.getenv("NLI_DEVICE", "auto").strip().lower()
    )
    NLI_MAX_LENGTH: int = field(
        default_factory=lambda: int(os.getenv("NLI_MAX_LENGTH", "512"))
    )
    NLI_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("NLI_BATCH_SIZE", "8"))
    )
    NLI_MAX_CLAIM_CHARS: int = field(
        default_factory=lambda: int(os.getenv("NLI_MAX_CLAIM_CHARS", "2000"))
    )
    NLI_MAX_EVIDENCE_CHARS: int = field(
        default_factory=lambda: int(os.getenv("NLI_MAX_EVIDENCE_CHARS", "8000"))
    )
    NLI_TRUNCATION_POLICY: str = field(
        default_factory=lambda: os.getenv("NLI_TRUNCATION_POLICY", "truncate").strip().lower()
    )
    NLI_CACHE_DIR: str | None = field(
        default_factory=lambda: os.getenv("NLI_CACHE_DIR", None)
    )

    # Submodule 3.7 Decomposition Metrics Settings
    DECOMPOSITION_WORD_LIMIT: int = 35
    DECOMPOSITION_MIN_COVERAGE: float = 0.70
    DECOMPOSITION_MIN_ATOMICITY: float = 0.80
    DECOMPOSITION_MIN_ENTAILMENT: float = 0.70

    # Submodule 4.1 Claim Complexity Analysis
    CLAIM_COMPLEXITY_THRESHOLD: float = 0.60

    # Submodule 5.1 Query Formulation Diversity Check
    # Token Jaccard threshold: >= 0.85 indicates near-verbatim overlap with baseline claim text
    # (typically differing by only 1-2 trivial tokens or punctuation). At or above this threshold,
    # dynamic reformulation offers no substantial retrieval differentiation and is rejected
    # in favor of the clean baseline query to avoid generating redundant or brittle search queries.
    DEFENDER_QUERY_DIVERSITY_THRESHOLD: float = 0.85

    # Submodule 5.3 Defender Evidence Filtering & Ranking
    DEFENDER_MIN_RELEVANCE: float = 0.2
    DEFENDER_RELEVANCE_WEIGHT: float = 0.5
    DEFENDER_CREDIBILITY_WEIGHT: float = 0.3
    DEFENDER_FRESHNESS_WEIGHT: float = 0.2

    # Submodule 6.1 Prosecutor Query Formulation Diversity Check
    # Token Jaccard threshold: >= 0.85 indicates near-verbatim overlap with baseline query.
    # At or above this threshold, dynamic counter-query offers no substantial differentiation
    # and is rejected to avoid issuing redundant search queries.
    PROSECUTOR_QUERY_DIVERSITY_THRESHOLD: float = 0.85

    # Submodule 6.3 Prosecutor Evidence Filtering & Ranking
    PROSECUTOR_MIN_RELEVANCE: float = 0.2
    PROSECUTOR_RELEVANCE_WEIGHT: float = 0.5
    PROSECUTOR_CREDIBILITY_WEIGHT: float = 0.3
    PROSECUTOR_FRESHNESS_WEIGHT: float = 0.2

    # Submodule 7 Dynamic Multi-Turn Debate Orchestrator
    DEBATE_MAX_ROUNDS: int = 2
    DEBATE_EARLY_STOPPING_ENABLED: bool = False
    DEBATE_MAX_BATCH_SIZE: int = 5

    # Submodule 7.3 Incremental Lift Threshold (Provisional, pending FEVER/SciFact calibration in 8.6)
    # Rationale: Multi-turn debate introduces a 2x retrieval cost multiplier (4 calls per claim across 2 agents).
    # On typical candidate pools (k=3 to 5 items per round, ~6 to 10 total items), a 15% relative expansion
    # corresponds to retrieving at least 1-2 genuinely new, high-scoring corroborating or disputing items
    # absent in Round 1. Incremental lift below 15% without new structured challenges indicates marginal or
    # redundant retrieval that does not justify the doubled retrieval latency/API cost, triggering an
    # evidentiary stagnation diagnostic log. Labeled PROVISIONAL pending empirical calibration in Module 8.6.
    DEBATE_INCREMENTAL_LIFT_THRESHOLD_PERCENT: float = 15.0

    # Submodule 8.4 Evidence Pollution Defense Heuristic Thresholds
    # [PROVISIONAL] Derived from empirical lexical overlap distributions on news scrapers where
    # boilerplate headers exhibit < 15% content overlap while substantive quotations exceed 50%.
    # Clean retention target (>95%) enforces non-destructive retrieval pre-filtering.
    # Routed to Submodule 8.6 FEVER/SciFact Calibration Checkpoint.
    POLLUTION_CIRCULARITY_THRESHOLD: float = 0.90  # P >= 0.90: Circularity / Self-referential / Exact duplicate
    POLLUTION_LOW_GROUNDING_THRESHOLD: float = 0.80  # P >= 0.80: Ungrounded / Severely ungrounded evidence
    POLLUTION_BOILERPLATE_THRESHOLD: float = 0.60  # P >= 0.60: Boilerplate / Scraped navigation template
    POLLUTION_CLEAN_THRESHOLD: float = 0.20  # P < 0.20: Clean admissible evidence
    POLLUTION_CLEAN_RETENTION_TARGET: float = 0.95  # Target specificity (>95%) for non-destructive pre-filtering


# Use one shared settings instance throughout the application.
settings = Settings()

