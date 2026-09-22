"""Basic sentence-based implementation of the claim extractor contract."""

import re

from app.services.claim_extractor import BaseClaimExtractor

# Split on whitespace following common sentence-ending punctuation.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_MOCK_RESPONSE_MARKER = "mock response for query"


class SimpleClaimExtractor(BaseClaimExtractor):
    """Extract candidate claims by treating each sentence as one claim.

    This initial heuristic does not split compound sentences, resolve pronouns,
    determine factuality, or understand abbreviations reliably. A later NLP
    implementation can address these limitations behind the same interface.
    """

    def extract_claims(self, response_text: str) -> list[str]:
        """Return non-empty sentence candidates in their original order."""

        sentences = _SENTENCE_BOUNDARY_PATTERN.split(response_text)
        claims: list[str] = []

        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue

            # Module 2 labels its development output with this marker. The
            # marker describes the mock mechanism rather than a factual claim,
            # so the containing sentence is excluded from downstream checks.
            if _MOCK_RESPONSE_MARKER in candidate.lower():
                continue

            claims.append(candidate)

        return claims
