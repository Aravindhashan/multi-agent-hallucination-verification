"""DeBERTa Cross-Encoder NLI Engine for factual claim verification."""

import logging
import time
from typing import Any

import torch

from app.services.nli_config import NLISettings
from app.services.nli_models import (
    NLIConfigurationError,
    NLIInferenceError,
    NLIInputError,
    NLILabel,
    NLIModelLoadError,
    NLIResult,
)

logger = logging.getLogger("hallucination_verifier.nli_engine")


class NLIEngine:
    """DeBERTa Cross-Encoder NLI Engine predicting Entailment, Contradiction, and Neutral.

    Performs full cross-attention over concatenated premise (evidence) and hypothesis
    (claim) pairs to compute softmax class probabilities from raw logits.
    Note: These scores are raw softmax probabilities and are not assumed to be calibrated.
    """

    def __init__(
        self,
        config: NLISettings | None = None,
        model: Any = None,
        tokenizer: Any = None,
        lazy_load: bool = True,
    ) -> None:
        """Initialize NLIEngine with configuration and optional preloaded model/tokenizer."""
        self.config = config if config is not None else NLISettings.from_settings()
        self.model = model
        self.tokenizer = tokenizer
        self.is_loaded = False
        self.label_mapping: dict[int, NLILabel] = {}
        self.entailment_idx: int = 0
        self.neutral_idx: int = 1
        self.contradiction_idx: int = 2

        self.device = self._resolve_device(self.config.device)

        if self.model is not None and self.tokenizer is not None:
            self._resolve_label_mapping()
            self.is_loaded = True
        elif not lazy_load:
            self._load_model()

    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve device string to torch.device object."""
        device_lower = str(device_str).strip().lower()
        if device_lower == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        try:
            return torch.device(device_lower)
        except Exception as exc:
            raise NLIConfigurationError(f"Failed to resolve torch device '{device_str}': {exc}") from exc

    def _load_model(self) -> None:
        """Load cross-encoder tokenizer and model weights from Hugging Face Hub / local cache."""
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return

        logger.info("Loading NLI model checkpoint: %s on device: %s", self.config.model_name, self.device)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=False,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self._resolve_label_mapping()
            self.is_loaded = True
            logger.info("NLI model successfully loaded and configured.")
        except Exception as exc:
            raise NLIModelLoadError(
                f"Failed to load NLI model '{self.config.model_name}': {exc}"
            ) from exc

    def _resolve_label_mapping(self) -> None:
        """Inspect model configuration to resolve exact index-to-label mappings."""
        resolved: dict[int, NLILabel] = {}
        config_obj = getattr(self.model, "config", None)
        id2label = getattr(config_obj, "id2label", None) if config_obj else None

        if id2label and isinstance(id2label, dict):
            for raw_idx, raw_lbl in id2label.items():
                idx = int(raw_idx)
                lbl_lower = str(raw_lbl).strip().lower()
                if "entail" in lbl_lower:
                    resolved[idx] = NLILabel.ENTAILMENT
                elif "contradict" in lbl_lower:
                    resolved[idx] = NLILabel.CONTRADICTION
                elif "neutral" in lbl_lower:
                    resolved[idx] = NLILabel.NEUTRAL

        if (
            len(resolved) == 3
            and set(resolved.values()) == {NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION}
        ):
            self.label_mapping = resolved
        else:
            # Standard DeBERTa MNLI / FEVER / ANLI fallback order: 0: entailment, 1: neutral, 2: contradiction
            self.label_mapping = {
                0: NLILabel.ENTAILMENT,
                1: NLILabel.NEUTRAL,
                2: NLILabel.CONTRADICTION,
            }

        self.entailment_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.ENTAILMENT][0]
        self.neutral_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.NEUTRAL][0]
        self.contradiction_idx = [k for k, v in self.label_mapping.items() if v == NLILabel.CONTRADICTION][0]

    def _validate_pair(self, claim: Any, evidence: Any) -> tuple[str, str]:
        """Validate single (claim, evidence) input pair."""
        if not isinstance(claim, str):
            raise NLIInputError(f"Claim must be a string, got {type(claim).__name__}.")
        if not isinstance(evidence, str):
            raise NLIInputError(f"Evidence must be a string, got {type(evidence).__name__}.")

        claim_trimmed = claim.strip()
        evidence_trimmed = evidence.strip()

        if not claim_trimmed:
            raise NLIInputError("Claim cannot be empty or whitespace-only.")
        if not evidence_trimmed:
            raise NLIInputError("Evidence cannot be empty or whitespace-only.")

        if len(claim) > self.config.max_claim_chars:
            raise NLIInputError(
                f"Claim exceeds maximum allowed length of {self.config.max_claim_chars} characters (got {len(claim)})."
            )
        if len(evidence) > self.config.max_evidence_chars:
            raise NLIInputError(
                f"Evidence exceeds maximum allowed length of {self.config.max_evidence_chars} characters (got {len(evidence)})."
            )

        return claim_trimmed, evidence_trimmed

    def predict(self, claim: str, evidence: str) -> NLIResult:
        """Run cross-encoder NLI inference on a single (claim, evidence) pair.

        Args:
            claim: Factual claim / hypothesis text to verify.
            evidence: Reference premise text to verify against.

        Returns:
            NLIResult containing softmax probabilities, predicted label, and confidence.
        """
        results = self.predict_batch([(claim, evidence)])
        return results[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Run batch cross-encoder NLI inference on a list of (claim, evidence) pairs.

        Args:
            pairs: List of tuples in (claim, evidence) format.

        Returns:
            List of NLIResult objects preserving the original input order.
        """
        if not pairs:
            return []

        # 1. Pre-validate all pairs before executing any inference
        validated_pairs = [self._validate_pair(c, e) for c, e in pairs]

        # 2. Ensure model is loaded
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self._load_model()

        results: list[NLIResult] = []

        # 3. Process in batches
        for batch_start in range(0, len(validated_pairs), self.config.batch_size):
            batch_slice = validated_pairs[batch_start : batch_start + self.config.batch_size]

            # In standard NLI cross-encoders: Premise = Evidence, Hypothesis = Claim
            premises = [e for c, e in batch_slice]
            hypotheses = [c for c, e in batch_slice]

            # 4. Tokenization and truncation policy enforcement
            if self.config.truncation_policy == "strict":
                raw_encodings = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=False,
                    truncation=False,
                )
                for ids in raw_encodings["input_ids"]:
                    if len(ids) > self.config.max_length:
                        raise NLIInputError(
                            f"Input sequence length ({len(ids)} tokens) exceeds max_length "
                            f"({self.config.max_length}) under strict truncation policy."
                        )
                tokens = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                truncated_flags = [False] * len(batch_slice)
            else:  # "truncate"
                raw_encodings = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=False,
                    truncation=False,
                )
                truncated_flags = [len(ids) > self.config.max_length for ids in raw_encodings["input_ids"]]
                tokens = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

            # Move tokens to target device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # 5. Forward pass with error handling
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    logits = outputs.logits
            except Exception as exc:
                raise NLIInferenceError(f"NLI model forward pass failed: {exc}") from exc

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise NLIInferenceError("NLI model produced NaN or Inf logits during inference.")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(batch_slice)

            # 6. Compute calibrated softmax probabilities
            probs_tensor = torch.nn.functional.softmax(logits.float(), dim=-1)
            probs_matrix = probs_tensor.cpu().tolist()

            # 7. Build NLIResult objects with contract validation
            for i, (claim_text, evidence_text) in enumerate(batch_slice):
                probs = probs_matrix[i]
                p_e = float(probs[self.entailment_idx])
                p_n = float(probs[self.neutral_idx])
                p_c = float(probs[self.contradiction_idx])

                scores = {
                    NLILabel.ENTAILMENT.value: p_e,
                    NLILabel.NEUTRAL.value: p_n,
                    NLILabel.CONTRADICTION.value: p_c,
                }
                max_label_str = max(scores, key=scores.get)
                best_label = NLILabel(max_label_str)
                best_conf = scores[max_label_str]

                try:
                    result = NLIResult(
                        claim=claim_text,
                        evidence=evidence_text,
                        label=best_label,
                        confidence=best_conf,
                        p_entailment=p_e,
                        p_neutral=p_n,
                        p_contradiction=p_c,
                        scores=scores,
                        truncated=truncated_flags[i],
                        model_name=self.config.model_name,
                        device=str(self.device),
                        inference_time_ms=round(elapsed_ms, 2),
                    )
                except ValueError as val_err:
                    raise NLIInferenceError(f"Probability contract violated: {val_err}") from val_err

                results.append(result)

        return results


# ---------------------------------------------------------------------------
# Singleton Factory and Reset
# ---------------------------------------------------------------------------

_nli_engine_instance: NLIEngine | None = None


def get_nli_engine(config: NLISettings | None = None) -> NLIEngine:
    """Get or create singleton NLIEngine instance."""
    global _nli_engine_instance
    if _nli_engine_instance is None or config is not None:
        _nli_engine_instance = NLIEngine(config=config)
    return _nli_engine_instance


def reset_nli_engine() -> None:
    """Reset singleton NLIEngine instance for test isolation."""
    global _nli_engine_instance
    _nli_engine_instance = None
