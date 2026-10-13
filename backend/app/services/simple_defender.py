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


class PlanGuidedDefender(BaseDefender):
    """Gather supporting evidence using plan-guided query formulation.

    Consumes extracted anchors (temporal, numerical, spatial) and verification
    plan steps to formulate high-precision queries.
    """

    def __init__(
        self,
        retriever: BaseEvidenceRetriever,
        anchors: dict[str, list[str]] | None = None,
        verification_plan: list[str] | None = None,
    ) -> None:
        """Initialize the Defender with retriever, anchors, and plan."""

        self.retriever = retriever
        self.anchors = anchors or {"temporal": [], "numerical": [], "spatial": []}
        self.verification_plan = verification_plan or []

    def build_query(self, claim_text: str) -> str:
        """Formulate an anchor- and plan-augmented query string."""

        base = claim_text.strip()
        parts: list[str] = []
        if self.anchors.get("temporal"):
            parts.append(f"{', '.join(self.anchors['temporal'])}")
        if self.anchors.get("numerical"):
            parts.append(f"{', '.join(self.anchors['numerical'])}")
        if self.anchors.get("spatial"):
            parts.append(f"{', '.join(self.anchors['spatial'])}")

        if parts:
            anchor_str = " ".join(parts)
            # Only append anchors if not already present in the base claim
            if anchor_str.lower() not in base.lower():
                return f"{base} {anchor_str}"
        return base

    def gather_supporting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Retrieve supporting evidence using the plan-guided query."""

        query_used = self.build_query(claim_text)
        evidence = self.retriever.retrieve(query_used, stance="supporting")
        return evidence, query_used
