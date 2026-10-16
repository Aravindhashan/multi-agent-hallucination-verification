"""Universal adversarial retrieval implementation of the Prosecutor agent.

Pillar 2: Bidirectional Adversarial Retrieval (Prosecutor Engine)
Universal Algorithm:
- Dynamic Adversarial Query Transformation: For any claim C = <S, P, O>, dynamically derive:
  1. Primary Counter-Query: Anchor-contrast and predicate-refutation probes
  2. Negated Predicate: "not <P>" or "<P> false OR controversy OR refuted OR debunked"
  3. Inverted Outcome: Antonym search based on claim anchors
- Dual-Perspective Retrieval & Aggregation:
  - If refuting evidence found: retrieves opposing snippets -> DeBERTa scores p_contradiction > 0.55 -> REFUTED -> H_refuted -> 1.00.
  - If 0 refutations found: confirms consensus adherence without ambiguous warnings.
"""

import re
from typing import Sequence

from app.schemas import EvidenceItem
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.prosecutor_agent import BaseProsecutor

# Polar opposites / antonym pairs across biomedical, clinical, and general domains
_POLAR_OPPOSITES: dict[str, str] = {
    # Clinical & Biomedical Efficacy
    "cure": "ineffective",
    "cures": "ineffective OR fails to cure",
    "cured": "ineffective OR failed to cure",
    "treat": "ineffective for",
    "treats": "ineffective for",
    "treated": "ineffective for",
    "effective": "ineffective OR contraindicated",
    "beneficial": "harmful OR adverse",
    "safe": "toxic OR dangerous OR contraindicated",
    "prevents": "fails to prevent OR risk",
    "prevent": "fail to prevent",
    "reduces": "elevates OR increases",
    "lowers": "raises OR elevates",
    "increases": "reduces OR decreases",
    "elevates": "lowers OR reduces",
    "inhibits": "activates OR stimulates",
    "activates": "inhibits OR blocks",
    "stimulates": "suppresses OR inhibits",
    "suppresses": "stimulates OR induces",
    "causes": "does not cause OR unrelated",
    "induces": "does not induce OR prevents",
    "improves": "worsens OR deteriorates",
    "protects": "exposes OR harms",
    "accelerates": "slows OR decelerates",
    "promotes": "impairs OR restricts",
    "helps": "hinders OR worsens",
    # Open-Domain & Factual
    "won": "lost",
    "wins": "loses",
    "succeeded": "failed",
    "approved": "rejected OR banned OR recalled",
    "valid": "invalid OR debunked",
    "true": "false",
    "real": "fictional OR mythical OR hoax",
    "born": "died",
    "expanded": "contracted OR shrank",
}


class SimpleProsecutor(BaseProsecutor):
    """Gather contradicting evidence with one direct retrieval query.

    Injecting the retriever keeps adversarial strategy independent from search.
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
    """Gather contradicting evidence using dynamic adversarial counter-query formulation.

    Implements Pillar 2 Dynamic Adversarial Query Transformation for any claim C = <S, P, O>:
    - Negated Predicate: "not <P>" or "<P> false OR controversy OR refuted OR debunked"
    - Inverted Outcome: Antonym search based on claim anchors
    - Anchor-contrast probes (dates, numbers, locations, directives)
    """

    def __init__(
        self,
        retriever: BaseEvidenceRetriever,
        anchors: dict[str, list[str]] | None = None,
        spo: dict[str, str | None] | None = None,
        verification_plan: list[str] | None = None,
    ) -> None:
        """Initialize the Prosecutor with retriever, anchors, SPO triple, and optional plan."""
        self.retriever = retriever
        self.anchors = anchors or {"temporal": [], "numerical": [], "spatial": []}
        self.spo = spo or {"subject": None, "predicate": None, "object": None}
        self.verification_plan = verification_plan or []

    def derive_negated_predicate_query(self, claim_text: str) -> str:
        """Derive 'not <P>' or '<P> false OR controversy OR refuted OR debunked'."""
        s = self.spo.get("subject")
        p = self.spo.get("predicate")
        o = self.spo.get("object")

        if s and p:
            p_clean = p.strip()
            p_lower = p_clean.lower()
            if p_lower in {"is", "was", "are", "were"}:
                negated_p = f"{p_clean} not"
            elif p_lower in {"has", "had", "have"}:
                negated_p = f"does not {p_clean}"
            elif p_lower in {"can", "could", "will", "would", "should", "may", "might", "must"}:
                negated_p = f"{p_clean} not"
            elif p_lower.endswith("s"):
                negated_p = f"does not {p_clean.rstrip('s')}"
            elif p_lower.endswith("ed"):
                negated_p = f"did not {p_clean}"
            else:
                negated_p = f"not {p_clean}"

            if o:
                return f"{s} {negated_p} {o}".strip()
            return f"{s} {negated_p}".strip()

        return f"{claim_text.strip()} false OR controversy OR refuted OR debunked"

    def derive_inverted_outcome_query(self, claim_text: str) -> str | None:
        """Derive antonym/inverted outcome search based on claim predicate or anchors."""
        words = re.findall(r"\b[a-zA-Z]+\b", claim_text)
        for w in words:
            w_lower = w.lower()
            if w_lower in _POLAR_OPPOSITES:
                antonym = _POLAR_OPPOSITES[w_lower]
                inverted = re.sub(rf"\b{re.escape(w)}\b", antonym, claim_text, count=1, flags=re.IGNORECASE)
                return inverted.strip()

        return None

    def build_counter_query(self, claim_text: str) -> str:
        """Formulate an anchor-aware adversarial counter-query."""
        base = claim_text.strip()
        probes: list[str] = []

        # 1. Verification plan challenges
        if self.verification_plan:
            plan_str = "; ".join(p.strip() for p in self.verification_plan if p.strip())
            if plan_str:
                probes.append(f"challenge: {plan_str}")

        # 2. Temporal anchor probe
        if self.anchors.get("temporal"):
            t_str = ", ".join(self.anchors["temporal"])
            probes.append(f"dispute alternative dates timeline controversy: {t_str}")

        # 3. Numerical anchor probe
        if self.anchors.get("numerical"):
            n_str = ", ".join(self.anchors["numerical"])
            probes.append(f"discrepancy dispute alternative numbers count: {n_str}")

        # 4. Spatial anchor probe
        if self.anchors.get("spatial"):
            s_str = ", ".join(self.anchors["spatial"])
            probes.append(f"dispute alternative location: {s_str}")

        if probes:
            probe_str = " ".join(probes)
            return f"{base} {probe_str}"

        # 5. Default predicate refutation probe if no anchors or directives are present
        return f"{base} false refuted debunked controversy myth"

    def build_adversarial_queries(self, claim_text: str) -> list[str]:
        """Generate the complete set of dynamic counter-hypotheses for bidirectional retrieval."""
        base = claim_text.strip()
        queries: list[str] = []

        # 1. Primary anchor/refutation counter-query
        primary = self.build_counter_query(base)
        queries.append(primary)

        # 2. Negated predicate query if distinct
        negated = self.derive_negated_predicate_query(base)
        if negated and negated.lower() != primary.lower() and negated not in queries:
            queries.append(negated)

        # 3. Inverted outcome antonym query if available
        inverted = self.derive_inverted_outcome_query(base)
        if inverted and inverted not in queries:
            queries.append(inverted)

        return queries

    def gather_contradicting_evidence(
        self,
        claim_text: str,
    ) -> tuple[list[EvidenceItem], str]:
        """Retrieve contradicting evidence using dynamic adversarial counter-queries."""
        queries = self.build_adversarial_queries(claim_text)
        primary_query = queries[0]

        all_evidence: list[EvidenceItem] = []
        seen_ids: set[str] = set()

        for q in queries[:2]:
            evidence = self.retriever.retrieve(q, stance="contradicting")
            for item in evidence:
                doc_id = str(item.evidence_id or item.source_url or hash(item.text.strip()))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_evidence.append(item)

        return all_evidence, primary_query
