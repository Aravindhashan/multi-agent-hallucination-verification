"""Initial retrieval-backed implementation of the Defender agent."""

from app.schemas import EvidenceItem
from app.services.defender_agent import BaseDefender
from app.services.evidence_retriever import BaseEvidenceRetriever


class SimpleDefender(BaseDefender):
    """Gather supporting evidence with one direct retrieval query.

    The retriever is injected so retrieval can be replaced independently of the
    Defender strategy. Future Defenders may expand the claim into several
    queries, consume Module 4's plan, filter evidence, or rerank sources.
    """

    def __init__(self, retriever: BaseEvidenceRetriever) -> None:
        """Initialize the Defender with a provider-independent retriever."""

        self.retriever = retriever

    def gather_supporting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Retrieve supporting evidence using the normalized claim as query."""

        query_used = claim_text.strip()
        evidence = self.retriever.retrieve(query_used, stance="supporting")
        return evidence, query_used
