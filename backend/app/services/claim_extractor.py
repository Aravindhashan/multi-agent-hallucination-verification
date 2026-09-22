"""Provider-independent interface for extracting claims from LLM text."""

from abc import ABC, abstractmethod
from app.services.claim_models import AtomicClaim, ClaimExtractionResult


class BaseClaimExtractor(ABC):
    """Define the contract for splitting an LLM response into claims.

    Implementations can be rule-based NLP pipelines, LLM prompt extractors,
    or hybrid multi-agent consensus extractors.
    """

    @abstractmethod
    def extract_claims(self, response_text: str) -> list[str]:
        """Split an LLM response into plain string candidate claims."""
        raise NotImplementedError

    def extract_atomic_claims(
        self, response_text: str, query: str = ""
    ) -> ClaimExtractionResult:
        """Extract structured AtomicClaim objects with qualifiers and anchors."""
        raw_claims = self.extract_claims(response_text)
        atomic_claims = [
            AtomicClaim(
                claim_id=f"c{idx + 1}",
                claim_text=text,
            )
            for idx, text in enumerate(raw_claims)
        ]
        return ClaimExtractionResult(
            claims=atomic_claims,
            extraction_method="base",
            raw_response_text=response_text,
        )
