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
    premise_text: str | None = Field(
        default=None,
        description="Exact premise text sent to DeBERTa cross-encoder.",
    )
    token_length: int | None = Field(
        default=None,
        description="Truncated token length of the input sequence evaluated by DeBERTa.",
    )
    raw_token_length: int | None = Field(
        default=None,
        description="Pre-truncation token length of the input sequence.",
    )
    selected_sentence_indices: list[int] | None = Field(
        default=None,
        description="0-indexed sentence indices extracted by sentence windowing from evidence passage.",
    )
    sentence_window_score: float | None = Field(
        default=None,
        description="Relevance matching score of the extracted sentence window against the claim.",
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


# ---------------------------------------------------------------------------
# Sentence-Level NLI & Max-Pooling Models
# ---------------------------------------------------------------------------

class SentencePrediction(BaseModel):
    """NLI prediction and confidence for a single sentence within a document."""

    doc_id: str | None = Field(default=None, description="Document ID the sentence belongs to.")
    sentence_idx: int = Field(..., ge=0, description="0-indexed position of sentence in document.")
    sentence_text: str = Field(..., description="Text of the evaluated sentence.")
    p_entailment: float = Field(..., ge=0.0, le=1.0, description="Probability of entailment.")
    p_neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral.")
    p_contradiction: float = Field(..., ge=0.0, le=1.0, description="Probability of contradiction.")
    label: NLILabel = Field(..., description="Argmax predicted label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of predicted label.")


class DocumentNLIPoolResult(BaseModel):
    """Result of sentence-level NLI scoring and max-pooling across one or more documents."""

    claim: str = Field(..., description="The factual claim / hypothesis evaluated.")
    num_documents: int = Field(default=1, ge=0, description="Number of candidate documents evaluated.")
    num_sentences_evaluated: int = Field(default=0, ge=0, description="Total sentences evaluated.")
    p_entailment_max: float = Field(..., ge=0.0, le=1.0, description="Max-pooled entailment probability.")
    p_contradiction_max: float = Field(..., ge=0.0, le=1.0, description="Max-pooled contradiction probability.")
    winning_label: NLILabel = Field(..., description="Final adjudicated label from max-pooling.")
    winning_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the winning label.")
    best_entailment_sentence: SentencePrediction | None = Field(default=None, description="Sentence achieving max entailment score.")
    best_contradiction_sentence: SentencePrediction | None = Field(default=None, description="Sentence achieving max contradiction score.")
    winning_sentence: SentencePrediction | None = Field(default=None, description="Sentence that determined the winning label.")
    sentence_predictions: list[SentencePrediction] = Field(default_factory=list, description="All sentence-level predictions.")

