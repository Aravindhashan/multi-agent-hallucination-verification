"""Central configuration for the Hallucination Verification API."""

from dataclasses import dataclass, field
import os


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
        default_factory=lambda: os.getenv("SESSION_DB_PATH", "data/pipeline_sessions.db").strip()
    )

    # Module 4 / Phase 1 Retrieval settings
    CORPUS_PATH: str = "data/corpus.json"
    INDEX_DIR: str = "data/indexes"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TOP_K: int = 5
    RETRIEVER_TYPE: str = field(
        default_factory=lambda: os.getenv("RETRIEVER_TYPE", "hybrid").strip().lower()
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
        default_factory=lambda: os.getenv("NLI_TRUNCATION_POLICY", "strict").strip().lower()
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


# Use one shared settings instance throughout the application.
settings = Settings()
