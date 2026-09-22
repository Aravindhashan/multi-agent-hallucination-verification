"""Provider-independent interface for the supporting-evidence agent."""

from abc import ABC, abstractmethod

from app.schemas import EvidenceItem


class BaseDefender(ABC):
    """Define the Defender contract for gathering support for a claim.

    The Defender seeks evidence in favor of a claim. Future implementations may
    use Module 4's plan, generate multiple queries, rerank results, and apply
    source-quality filters without changing callers of this interface.
    """

    @abstractmethod
    def gather_supporting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Gather evidence supporting the claim and return the query used."""
