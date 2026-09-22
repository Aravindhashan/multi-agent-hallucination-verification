"""Initial retrieval-backed implementation of the Prosecutor agent."""

from app.schemas import EvidenceItem
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_agent import BaseProsecutor


class SimpleProsecutor(BaseProsecutor):
    """Gather contradicting evidence with one direct retrieval query.

    Injecting the retriever keeps adversarial strategy independent from search.
    Future Prosecutors may expand challenge queries and attack evidence quality,
    duplication, or pollution; evidence-base attacks are reserved for Module 9.
    """

    def __init__(self, retriever: BaseEvidenceRetriever) -> None:
        """Initialize the Prosecutor with a provider-independent retriever."""

        self.retriever = retriever

    def gather_contradicting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Retrieve contradicting evidence using the claim as the query."""

        query_used = claim_text.strip()
        evidence = self.retriever.retrieve(query_used, stance="contradicting")
        return evidence, query_used
