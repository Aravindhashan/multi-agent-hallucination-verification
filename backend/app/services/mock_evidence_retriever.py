"""Deterministic placeholder implementation of evidence retrieval."""

from uuid import uuid4

from app.schemas import EvidenceItem
from app.services.evidence_retriever import BaseEvidenceRetriever


class MockEvidenceRetriever(BaseEvidenceRetriever):
    """Return fixed-shape mock evidence for development and testing.

    This implementation performs no search or network calls. It must be
    replaced by a real retrieval system before evidence can be treated as
    factual, independent, or suitable for production verification.
    """

    def retrieve(self, claim_text: str, stance: str) -> list[EvidenceItem]:
        """Return two mock evidence items for a supported retrieval stance."""

        normalized_claim = claim_text.strip()
        normalized_stance = stance.strip().lower()

        if normalized_stance == "supporting":
            marker = "MOCK SUPPORTING EVIDENCE"
            path_segment = "supporting"
            source_title = "Mock Supporting Source"
            relevance_score = 0.85
        elif normalized_stance == "contradicting":
            marker = "MOCK CONTRADICTING EVIDENCE"
            path_segment = "contradicting"
            source_title = "Mock Contradicting Source"
            relevance_score = 0.75
        else:
            # Explicit rejection prevents callers from silently receiving
            # evidence for the wrong adversarial stance.
            raise ValueError(
                "Evidence stance must be 'supporting' or 'contradicting'."
            )

        # UUIDs differ for tracking, while all other content is deterministic
        # for a given claim and stance. Real retrieval will replace this block.
        return [
            EvidenceItem(
                evidence_id=uuid4(),
                text=(
                    f"{marker} for claim: '{normalized_claim}'. "
                    f"Placeholder {path_segment} passage {index}."
                ),
                source_url=(
                    f"https://mock.example.com/{path_segment}/{index}"
                ),
                domain="example.com",
                source_title=source_title,
                retrieval_method="mock",
                relevance_score=relevance_score,
            )
            for index in range(1, 3)
        ]
