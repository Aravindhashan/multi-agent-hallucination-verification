"""Provider-independent interface for claim analysis and planning."""

from abc import ABC, abstractmethod

from app.schemas import ClaimAnalysisResult


class BaseClaimAnalyzer(ABC):
    """Define the contract for claim-complexity analysis.

    Future statistical, transformer-based, or agentic planners should implement
    this interface. The service and API contracts can therefore remain stable
    when the heuristic implementation is replaced.
    """

    @abstractmethod
    def analyze(self, claim_text: str, atomic_claim: object = None) -> ClaimAnalysisResult:
        """Analyze a claim and produce a verification plan."""
