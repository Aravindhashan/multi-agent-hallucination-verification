"""Provider-independent interface for the adversarial Prosecutor agent."""

from abc import ABC, abstractmethod

from app.schemas import EvidenceItem


class BaseProsecutor(ABC):
    """Define the Prosecutor contract for challenging a claim.

    The Prosecutor actively seeks contradictions, counterexamples, outdated
    information, and missing qualifiers. Future implementations can add richer
    adversarial strategies without changing callers of this interface.
    """

    @abstractmethod
    def gather_contradicting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Gather evidence contradicting or challenging the claim.

        Return both the retrieved evidence and the query used.
        """
