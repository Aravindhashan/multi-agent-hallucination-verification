"""DeBERTa Cross-Encoder NLI Engine for factual claim verification."""

import logging
import threading
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os

try:
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from app.services.nli_config import NLISettings
from app.services.nli_models import (
    DocumentNLIPoolResult,
    NLIConfigurationError,
    NLIInferenceError,
    NLIInputError,
    NLILabel,
    NLIModelLoadError,
    NLIResult,
    SentencePrediction,
)
from app.services.sentence_window import (
    SentenceWindow,
    select_best_sentence_window,
    split_into_sentences,
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
        self._lock = threading.RLock()
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
        with self._lock:
            if self.is_loaded and self.model is not None and self.tokenizer is not None:
                return

            logger.info("Loading NLI model checkpoint: %s on device: %s", self.config.model_name, self.device)
            try:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        cache_dir=self.config.cache_dir,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.model_name,
                        cache_dir=self.config.cache_dir,
                        local_files_only=True,
                        trust_remote_code=False,
                        low_cpu_mem_usage=False,
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        cache_dir=self.config.cache_dir,
                        trust_remote_code=False,
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.model_name,
                        cache_dir=self.config.cache_dir,
                        trust_remote_code=False,
                        low_cpu_mem_usage=False,
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
                if "entail" in lbl_lower or "support" in lbl_lower:
                    resolved[idx] = NLILabel.ENTAILMENT
                elif "contradict" in lbl_lower or "refute" in lbl_lower:
                    resolved[idx] = NLILabel.CONTRADICTION
                elif "neutral" in lbl_lower or "not_enough_info" in lbl_lower:
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
            if self.config.truncation_policy == "truncate":
                claim = claim[:self.config.max_claim_chars]
                claim_trimmed = claim.strip()
            else:
                raise NLIInputError(
                    f"Claim exceeds maximum allowed length of {self.config.max_claim_chars} characters (got {len(claim)})."
                )
        if len(evidence) > self.config.max_evidence_chars:
            if self.config.truncation_policy == "truncate":
                evidence = evidence[:self.config.max_evidence_chars]
                evidence_trimmed = evidence.strip()
            else:
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

    def predict_multi_document_sentences(
        self,
        claim: str,
        documents: list[tuple[str, str]],
        threshold: float = 0.5,
    ) -> DocumentNLIPoolResult:
        """Score every sentence of candidate documents against the claim and max-pool.

        Standard SciFact evidence verification pipeline:
        1. Splits each candidate abstract into discrete sentences using abbreviation-safe splitting.
        2. Scores every (claim, sentence) pair directly with cross-encoder NLI (no heuristic windowing).
        3. Max-pools entailment and contradiction scores across all sentences.
        4. Selects winning label and assigns precise sentence attribution.

        Args:
            claim: Factual claim / hypothesis to verify.
            documents: List of (doc_id, doc_text) tuples.
            threshold: Probability threshold for adopting entailment or contradiction stance (default: 0.5).

        Returns:
            DocumentNLIPoolResult with max-pooled probabilities and winning sentence attribution.
        """
        if not claim or not str(claim).strip():
            raise NLIInputError("Claim text cannot be empty.")

        claim_clean = str(claim).strip()
        all_sentence_entries: list[tuple[str, int, str]] = []  # (doc_id, sent_idx, sent_text)

        for doc_id, doc_text in documents:
            if not doc_text or not str(doc_text).strip():
                continue
            sents = split_into_sentences(doc_text)
            for s_idx, s_text in enumerate(sents):
                clean_s = s_text.strip()
                if clean_s:
                    all_sentence_entries.append((doc_id, s_idx, clean_s))

        if not all_sentence_entries:
            return DocumentNLIPoolResult(
                claim=claim_clean,
                num_documents=len(documents),
                num_sentences_evaluated=0,
                p_entailment_max=0.0,
                p_contradiction_max=0.0,
                winning_label=NLILabel.NEUTRAL,
                winning_confidence=1.0,
                sentence_predictions=[],
            )

        # Build raw sentence pairs. Note: bypass heuristic windowing by running
        # directly on raw sentence text with enable_sentence_windows temporarily bypassed.
        pairs = [(claim_clean, s_text) for _, _, s_text in all_sentence_entries]

        original_window_setting = self.config.enable_sentence_windows
        try:
            self.config.enable_sentence_windows = False
            raw_results = self.predict_batch(pairs)
        finally:
            self.config.enable_sentence_windows = original_window_setting

        sentence_preds: list[SentencePrediction] = []
        max_ent = -1.0
        max_con = -1.0
        best_ent_pred: SentencePrediction | None = None
        best_con_pred: SentencePrediction | None = None

        for (doc_id, s_idx, s_text), res in zip(all_sentence_entries, raw_results):
            sp = SentencePrediction(
                doc_id=doc_id,
                sentence_idx=s_idx,
                sentence_text=s_text,
                p_entailment=res.p_entailment,
                p_neutral=res.p_neutral,
                p_contradiction=res.p_contradiction,
                label=res.label,
                confidence=res.confidence,
            )
            sentence_preds.append(sp)

            if res.p_entailment > max_ent:
                max_ent = res.p_entailment
                best_ent_pred = sp

            if res.p_contradiction > max_con:
                max_con = res.p_contradiction
                best_con_pred = sp

        # Adjudication via max-pooling
        max_ent_val = max(max_ent, 0.0)
        max_con_val = max(max_con, 0.0)

        if max_ent_val > max_con_val and max_ent_val >= threshold:
            winning_label = NLILabel.ENTAILMENT
            winning_confidence = max_ent_val
            winning_sent = best_ent_pred
        elif max_con_val > max_ent_val and max_con_val >= threshold:
            winning_label = NLILabel.CONTRADICTION
            winning_confidence = max_con_val
            winning_sent = best_con_pred
        else:
            winning_label = NLILabel.NEUTRAL
            winning_confidence = max(1.0 - max(max_ent_val, max_con_val), threshold)
            winning_sent = None

        return DocumentNLIPoolResult(
            claim=claim_clean,
            num_documents=len(documents),
            num_sentences_evaluated=len(sentence_preds),
            p_entailment_max=max_ent_val,
            p_contradiction_max=max_con_val,
            winning_label=winning_label,
            winning_confidence=winning_confidence,
            best_entailment_sentence=best_ent_pred,
            best_contradiction_sentence=best_con_pred,
            winning_sentence=winning_sent,
            sentence_predictions=sentence_preds,
        )

    def predict_document_sentences(
        self,
        claim: str,
        document_text: str,
        document_id: str | None = None,
        threshold: float = 0.5,
    ) -> DocumentNLIPoolResult:
        """Score every sentence of a single abstract against the claim and max-pool."""
        return self.predict_multi_document_sentences(
            claim=claim,
            documents=[(document_id or "doc_0", document_text)],
            threshold=threshold,
        )

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
            hypotheses = [c for c, e in batch_slice]
            raw_evidence_list = [e for c, e in batch_slice]

            # If sentence windowing is enabled, extract the best 1-3 sentences:
            premises = []
            extracted_windows: list[SentenceWindow | None] = []
            for c, e in batch_slice:
                if self.config.enable_sentence_windows:
                    window = select_best_sentence_window(
                        claim=c,
                        evidence_passage=e,
                        min_sentences=self.config.min_window_sentences,
                        max_sentences=self.config.max_window_sentences,
                        strategy=self.config.sentence_window_strategy,
                    )
                    premises.append(window.text)
                    extracted_windows.append(window)
                else:
                    premises.append(e)
                    extracted_windows.append(None)

            # Standard NLI cross-encoder convention: Premise (evidence) is sequence 1, Hypothesis (claim) is sequence 2.
            first_seq = premises
            second_seq = hypotheses
            raw_first = raw_evidence_list
            raw_second = hypotheses

            # Raw un-truncated encodings of original input pairs for diagnostic tracking
            raw_full_encodings = self.tokenizer(
                raw_first,
                raw_second,
                padding=False,
                truncation=False,
            )

            # 4. Tokenization and truncation policy enforcement on model inputs
            if self.config.truncation_policy == "strict":
                raw_window_encodings = self.tokenizer(
                    first_seq,
                    second_seq,
                    padding=False,
                    truncation=False,
                )
                for ids in raw_window_encodings["input_ids"]:
                    if len(ids) > self.config.max_length:
                        raise NLIInputError(
                            f"Input sequence length ({len(ids)} tokens) exceeds max_length "
                            f"({self.config.max_length}) under strict truncation policy."
                        )
                tokens = self.tokenizer(
                    first_seq,
                    second_seq,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                truncated_flags = [False] * len(batch_slice)
            else:  # "truncate"
                raw_window_encodings = self.tokenizer(
                    first_seq,
                    second_seq,
                    padding=False,
                    truncation=False,
                )
                truncated_flags = [len(ids) > self.config.max_length for ids in raw_window_encodings["input_ids"]]
                tokens = self.tokenizer(
                    first_seq,
                    second_seq,
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
                with torch.inference_mode():
                    outputs = self.model(**tokens)
                    logits = outputs.logits
            except Exception as exc:
                raise NLIInferenceError(f"NLI model forward pass failed: {exc}") from exc

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise NLIInferenceError("NLI model produced NaN or Inf logits during inference.")

            batch_sec = time.perf_counter() - t0
            elapsed_ms = batch_sec * 1000.0 / len(batch_slice)

            # 6. Compute calibrated softmax probabilities
            probs_tensor = torch.nn.functional.softmax(logits.float(), dim=-1)
            probs_matrix = probs_tensor.cpu().tolist()

            total_batches = (len(validated_pairs) + self.config.batch_size - 1) // self.config.batch_size
            curr_batch = (batch_start // self.config.batch_size) + 1
            logger.info("NLI Batch [%d/%d] processed %d pairs in %.2fs (%.2f ms/pair)", curr_batch, total_batches, len(batch_slice), batch_sec, elapsed_ms)

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

                # Compute token lengths
                raw_full_ids = raw_full_encodings["input_ids"][i]
                raw_token_len = len(raw_full_ids)
                if "attention_mask" in tokens and hasattr(tokens["attention_mask"], "shape"):
                    actual_token_len = int(tokens["attention_mask"][i].sum().item())
                else:
                    actual_token_len = min(len(raw_window_encodings["input_ids"][i]), self.config.max_length)
                is_truncated = bool(truncated_flags[i])

                premise_text = premises[i]
                win_obj = extracted_windows[i]
                selected_indices = win_obj.sentence_indices if win_obj else None
                window_score = win_obj.score if win_obj else None

                # Log raw input and model outputs
                logger.info(
                    "DeBERTa NLI Raw Input & Prediction:\n"
                    "  [Claim (Hypothesis)]: %s\n"
                    "  [Premise (Evidence)]: %s\n"
                    "  [Sentence Window]: sents=%s, score=%s\n"
                    "  [Token Length]: %d (raw: %d, truncated: %s, max_cap: %d)\n"
                    "  [Probabilities]: Entailment=%.4f, Neutral=%.4f, Contradiction=%.4f -> Predicted: %s (conf: %.4f)",
                    claim_text,
                    premise_text,
                    selected_indices,
                    f"{window_score:.4f}" if window_score is not None else "N/A",
                    actual_token_len,
                    raw_token_len,
                    is_truncated,
                    self.config.max_length,
                    p_e,
                    p_n,
                    p_c,
                    best_label.value,
                    best_conf,
                )

                try:
                    result = NLIResult(
                        claim=claim_text,
                        evidence=evidence_text,
                        premise_text=premise_text,
                        token_length=actual_token_len,
                        raw_token_length=raw_token_len,
                        selected_sentence_indices=selected_indices,
                        sentence_window_score=window_score,
                        label=best_label,
                        confidence=best_conf,
                        p_entailment=p_e,
                        p_neutral=p_n,
                        p_contradiction=p_c,
                        scores=scores,
                        truncated=is_truncated,
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
