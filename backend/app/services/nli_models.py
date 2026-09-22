"""Data models, label enum, and exception hierarchy for the DeBERTa NLI Engine."""

from enum import Enum
import math
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class NLIError(Exception):
    """Base exception for all NLI service failures."""

    pass


class NLIConfigurationError(NLIError):
    """Raised when NLI settings or runtime parameters are invalid."""

    pass


class NLIModelLoadError(NLIError):
    """Raised when the NLI cross-encoder weights or tokenizer cannot be loaded."""

    pass


class NLIInferenceError(NLIError):
    """Raised when forward pass execution or probability contract verification fails."""

    pass


class NLIInputError(NLIError, ValueError):
    """Raised when input claim or evidence fails validation."""

    pass


# ---------------------------------------------------------------------------
# Label Enum
# ---------------------------------------------------------------------------

class NLILabel(str, Enum):
    """Standard 3-class NLI classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# NLI Result Model
# ---------------------------------------------------------------------------

class NLIResult(BaseModel):
    """Result of cross-encoder Natural Language Inference for a (claim, evidence) pair."""

    claim: str = Field(..., description="The factual claim / hypothesis being verified.")
    evidence: str = Field(..., description="The reference evidence / premise passage.")
    label: NLILabel = Field(..., description="Predicted label: entailment, contradiction, or neutral.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (softmax probability of predicted label).")
    p_entailment: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for entailment computed from raw NLI logits (not assumed to be calibrated).")
    p_neutral: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for neutral computed from raw NLI logits (not assumed to be calibrated).")
    p_contradiction: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for contradiction computed from raw NLI logits (not assumed to be calibrated).")
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary mapping label names to their uncalibrated softmax probabilities.",
    )
    truncated: bool = Field(
        default=False,
        description="Whether the input sequence was truncated during tokenization.",
    )
    model_name: str = Field(..., description="Name or path of the NLI model checkpoint used.")
    device: str = Field(..., description="Device used for inference ('cpu', 'cuda', 'mps', etc.).")
    inference_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Inference execution latency in milliseconds.",
    )

    @field_validator("claim")
    @classmethod
    def validate_claim_non_empty(cls, v: str) -> str:
        """Ensure claim is non-empty and not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Claim must contain non-whitespace text.")
        return v

    @field_validator("evidence")
    @classmethod
    def validate_evidence_non_empty(cls, v: str) -> str:
        """Ensure evidence is non-empty and not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Evidence must contain non-whitespace text.")
        return v

    @model_validator(mode="after")
    def validate_probability_contract(self) -> "NLIResult":
        """Enforce strict probability contract: sum to 1.0, confidence matches max probability, label matches argmax."""
        # Ensure scores dict is populated consistently
        if not self.scores:
            self.scores = {
                NLILabel.ENTAILMENT.value: self.p_entailment,
                NLILabel.NEUTRAL.value: self.p_neutral,
                NLILabel.CONTRADICTION.value: self.p_contradiction,
            }

        # Check probability bounds
        for name, p in [
            ("p_entailment", self.p_entailment),
            ("p_neutral", self.p_neutral),
            ("p_contradiction", self.p_contradiction),
        ]:
            if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
                raise ValueError(f"Probability {name}={p} is not within [0.0, 1.0].")

        # Check partition function (probabilities sum to 1.0 within numerical tolerance)
        prob_sum = self.p_entailment + self.p_neutral + self.p_contradiction
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"NLI probabilities must sum to 1.0 (got {prob_sum:.6f}).")

        # Check label matches argmax
        prob_map = {
            NLILabel.ENTAILMENT: self.p_entailment,
            NLILabel.NEUTRAL: self.p_neutral,
            NLILabel.CONTRADICTION: self.p_contradiction,
        }
        max_label = max(prob_map, key=prob_map.get)
        max_prob = prob_map[max_label]

        if self.label != max_label:
            raise ValueError(
                f"Label '{self.label.value}' does not match argmax label '{max_label.value}' (probs: {prob_map})."
            )

        if abs(self.confidence - max_prob) > 1e-3:
            raise ValueError(
                f"Confidence ({self.confidence:.6f}) does not match maximum class probability ({max_prob:.6f})."
            )

        return self
