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


class AdvCounterQueryProsecutor(BaseProsecutor):
    """Gather contradicting evidence using adversarial counter-query formulation.

    Moves beyond blunt keyword appending by formulating structured counter-queries
    targeting specific factual anchors (temporal disputes, numerical discrepancies,
    spatial conflicts) and predicate refutations.
    """

    def __init__(
        self,
        retriever: BaseEvidenceRetriever,
        anchors: dict[str, list[str]] | None = None,
        verification_plan: list[str] | None = None,
    ) -> None:
        """Initialize the Prosecutor with retriever, anchors, and optional plan."""
        self.retriever = retriever
        self.anchors = anchors or {"temporal": [], "numerical": [], "spatial": []}
        self.verification_plan = verification_plan or []

    def build_counter_query(self, claim_text: str) -> str:
        """Formulate an anchor-aware adversarial counter-query."""
        base = claim_text.strip()
        probes: list[str] = []

        # 1. Temporal discrepancy probe
        if self.anchors.get("temporal"):
            t_str = ", ".join(self.anchors["temporal"])
            probes.append(f"controversy {t_str}")

        # 2. Numerical discrepancy probe
        if self.anchors.get("numerical"):
            n_str = ", ".join(self.anchors["numerical"])
            probes.append(f"discrepancy {n_str}")

        # 3. Spatial discrepancy probe
        if self.anchors.get("spatial"):
            s_str = ", ".join(self.anchors["spatial"])
            probes.append(f"dispute {s_str}")

        if probes:
            probe_str = " ".join(probes)
            return f"{base} {probe_str}"

        # 4. Default predicate refutation probe if no anchors are present
        return f"{base} false refuted controversy"

    def gather_contradicting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Retrieve contradicting evidence using the adversarial counter-query."""
        query_used = self.build_counter_query(claim_text)
        evidence = self.retriever.retrieve(query_used, stance="contradicting")
        return evidence, query_used

