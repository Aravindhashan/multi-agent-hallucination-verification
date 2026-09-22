"""Configuration dataclass for the DeBERTa NLI Engine."""

from dataclasses import dataclass
from typing import Any

from app.services.nli_models import NLIConfigurationError


@dataclass
class NLISettings:
    """Runtime configuration settings for DeBERTa cross-encoder NLI inference."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 8
    max_claim_chars: int = 2000
    max_evidence_chars: int = 8000
    truncation_policy: str = "strict"  # "strict" or "truncate"
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate all NLI configuration parameters."""
        if not self.model_name or not str(self.model_name).strip():
            raise NLIConfigurationError("NLI model_name cannot be empty.")
        self.model_name = str(self.model_name).strip()

        dev = str(self.device).strip().lower()
        if dev not in {"auto", "cpu", "cuda", "mps"} and not dev.startswith("cuda:"):
            raise NLIConfigurationError(
                f"Invalid NLI device: '{self.device}'. Must be 'auto', 'cpu', 'cuda', 'mps', or 'cuda:N'."
            )
        self.device = dev

        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise NLIConfigurationError(f"NLI max_length must be a positive integer, got {self.max_length}.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise NLIConfigurationError(f"NLI batch_size must be a positive integer, got {self.batch_size}.")

        if not isinstance(self.max_claim_chars, int) or self.max_claim_chars <= 0:
            raise NLIConfigurationError(f"NLI max_claim_chars must be a positive integer, got {self.max_claim_chars}.")

        if not isinstance(self.max_evidence_chars, int) or self.max_evidence_chars <= 0:
            raise NLIConfigurationError(f"NLI max_evidence_chars must be a positive integer, got {self.max_evidence_chars}.")

        trunc = str(self.truncation_policy).strip().lower()
        if trunc not in {"strict", "truncate"}:
            raise NLIConfigurationError(
                f"Invalid truncation_policy: '{self.truncation_policy}'. Must be 'strict' or 'truncate'."
            )
        self.truncation_policy = trunc

    @classmethod
    def from_settings(cls, base_settings: Any = None) -> "NLISettings":
        """Build NLISettings from application Settings object or environment defaults."""
        if base_settings is None:
            from app.config import settings as app_settings
            base_settings = app_settings

        return cls(
            model_name=getattr(base_settings, "NLI_MODEL_NAME", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
            device=getattr(base_settings, "NLI_DEVICE", "auto"),
            max_length=getattr(base_settings, "NLI_MAX_LENGTH", 512),
            batch_size=getattr(base_settings, "NLI_BATCH_SIZE", 8),
            max_claim_chars=getattr(base_settings, "NLI_MAX_CLAIM_CHARS", 2000),
            max_evidence_chars=getattr(base_settings, "NLI_MAX_EVIDENCE_CHARS", 8000),
            truncation_policy=getattr(base_settings, "NLI_TRUNCATION_POLICY", "strict"),
            cache_dir=getattr(base_settings, "NLI_CACHE_DIR", None),
        )
