"""Provider-independent interface for claim evidence retrieval."""

from abc import ABC, abstractmethod

from app.schemas import EvidenceItem


class BaseEvidenceRetriever(ABC):
    """Define the contract for retrieving stance-specific claim evidence.

    Future hybrid retrievers combining BM25, semantic search, reranking, and
    external sources should implement this interface. Defender and Prosecutor
    agents can then share retrieval without depending on a specific provider.
    """

    @abstractmethod
    def retrieve(self, claim_text: str, stance: str) -> list[EvidenceItem]:
        """Retrieve evidence for a claim with a specific stance.

        Supported stances are ``supporting`` and ``contradicting``.
        """
