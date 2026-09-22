"""LLM-based Atomic Claim Extractor using structured prompt templates."""

from datetime import datetime, timezone
import json
import re
from typing import Any

from app.prompts.verification_prompts import CLAIM_EXTRACTION_PROMPT
from app.services.claim_extractor import BaseClaimExtractor
from app.services.claim_models import AtomicClaim, ClaimExtractionResult
from app.services.decomposition_evaluator import DecompositionQualityEvaluator
from app.services.llm_client import BaseLLMClient, LLMResponseError
from app.services.llm_service import get_llm_client
from app.services.nlp_claim_extractor import NLPClaimExtractor


class LLMClaimExtractor(BaseClaimExtractor):
    """Extract atomic claims using an LLM guided by CLAIM_EXTRACTION_PROMPT.

    Parses structured JSON outputs, normalizes claims, assigns deterministic IDs,
    and enriches claims with anchor and qualifier metadata.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        nlp_enricher: NLPClaimExtractor | None = None,
    ) -> None:
        """Initialize LLM extractor with an LLM client and optional NLP enricher."""
        self._llm_client = llm_client
        self._nlp_enricher = nlp_enricher or NLPClaimExtractor()

    @property
    def llm_client(self) -> BaseLLMClient:
        """Return the active LLM client or dynamically resolve from factory."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def extract_claims(self, response_text: str) -> list[str]:
        """Extract plain string claims from response text."""
        result = self.extract_atomic_claims(response_text)
        return [c.claim_text for c in result.claims]

    def extract_atomic_claims(
        self, response_text: str, query: str = ""
    ) -> ClaimExtractionResult:
        """Decompose text into atomic claims via LLM prompting and JSON parsing."""
        self._validate_input(response_text)

        formatted_query = query.strip() if query and query.strip() else "Extract factual claims from the text."
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            query=formatted_query,
            response_text=response_text.strip(),
        )

        raw_output = self.llm_client.generate(prompt)
        parsed_claims_data = self._parse_json_claims(raw_output)

        # Build AtomicClaim models with NLP metadata enrichment and deduplication
        seen_keys: set[str] = set()
        atomic_claims: list[AtomicClaim] = []

        for item in parsed_claims_data:
            if isinstance(item, str):
                claim_text = item.strip()
                # Run NLP enrichment on claim text
                analysis = self._nlp_enricher._analyze_clause(claim_text)
                subject = analysis.get("subject")
                predicate = analysis.get("predicate")
                obj = analysis.get("object")
                temporal_anchor = analysis.get("temporal_anchor")
                numerical_value = analysis.get("numerical_value")
                spatial_anchor = analysis.get("spatial_anchor")
                temporal_anchors = analysis.get("temporal_anchors", [])
                numerical_values = analysis.get("numerical_values", [])
                spatial_anchors = analysis.get("spatial_anchors", [])
                qualifiers = analysis.get("qualifiers", [])
            elif isinstance(item, dict):
                claim_text = str(item.get("claim_text", item.get("text", ""))).strip()
                subject = item.get("subject")
                predicate = item.get("predicate")
                obj = item.get("object")
                temporal_anchor = item.get("temporal_anchor")
                numerical_value = item.get("numerical_value")
                spatial_anchor = item.get("spatial_anchor")
                temporal_anchors = item.get("temporal_anchors", [])
                numerical_values = item.get("numerical_values", [])
                spatial_anchors = item.get("spatial_anchors", [])
                qualifiers = item.get("qualifiers", [])
            else:
                continue

            if not claim_text:
                continue

            normalized_key = re.sub(r"\s+", " ", claim_text.lower().rstrip(".!?"))
            if not normalized_key or normalized_key in seen_keys:
                continue

            seen_keys.add(normalized_key)
            claim_id = f"c{len(atomic_claims) + 1}"

            atomic_claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    temporal_anchor=temporal_anchor,
                    numerical_value=numerical_value,
                    spatial_anchor=spatial_anchor,
                    temporal_anchors=temporal_anchors,
                    numerical_values=numerical_values,
                    spatial_anchors=spatial_anchors,
                    qualifiers=qualifiers,
                )
            )

        evaluator = DecompositionQualityEvaluator()
        decomposition_metrics = evaluator.evaluate(response_text, atomic_claims)

        return ClaimExtractionResult(
            claims=atomic_claims,
            extraction_method="llm",
            raw_response_text=response_text,
            decomposition_metrics=decomposition_metrics,
            extracted_at=datetime.now(timezone.utc),
        )

    def _validate_input(self, text: Any) -> None:
        """Validate that input text is a non-empty string."""
        if not isinstance(text, str):
            raise ValueError(f"Expected response_text to be a string, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("Response text must contain at least one non-whitespace character.")

    def _parse_json_claims(self, raw_output: str) -> list[Any]:
        """Extract and parse JSON object containing the claims array."""
        if not raw_output or not raw_output.strip():
            raise LLMResponseError("LLM returned an empty response for claim extraction.")

        cleaned = raw_output.strip()

        # Remove markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Attempt direct JSON decode
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of JSON object
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError as exc:
                    raise LLMResponseError(
                        f"Failed to parse LLM JSON claims output: {exc}. Raw: {raw_output[:200]}"
                    ) from exc
            else:
                raise LLMResponseError(
                    f"No valid JSON object found in LLM claim extraction output: {raw_output[:200]}"
                )

        if isinstance(data, dict):
            claims = data.get("claims", [])
            if isinstance(claims, list):
                return claims
            raise LLMResponseError("JSON response 'claims' field must be a list.")
        elif isinstance(data, list):
            return data
        else:
            raise LLMResponseError(f"Unexpected JSON structure from LLM: {type(data).__name__}")
