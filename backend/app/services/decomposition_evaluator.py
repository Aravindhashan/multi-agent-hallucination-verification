"""Submodule 3.7: Decomposition Quality Metrics Engine.

Evaluates the factual decomposition of source text into atomic claims:
1. Coverage: Verifies whether decomposition dropped vital information or named entities.
2. Atomicity: Verifies whether each extracted claim is an indivisible single proposition.
3. Entailment-Preservation: Verifies whether the claim set faithfully captures source facts.
"""

import re
from typing import Any
from app.config import settings
from app.services.claim_models import AtomicClaim, DecompositionMetrics

from app.utils.lexical import CANONICAL_STOPWORDS

# Common English stopwords to ignore when computing content token overlap
_STOPWORDS = CANONICAL_STOPWORDS

_COORDINATING_CONJUNCTION_PATTERNS = [
    re.compile(r",\s*(?:and|but|or|yet|while|whereas)\b", re.IGNORECASE),
    re.compile(r";\s*"),
    re.compile(r"\b(?:as well as|not only .* but also)\b", re.IGNORECASE),
]


class DecompositionQualityEvaluator:
    """Evaluate coverage, atomicity, and entailment-preservation of decomposed claims."""

    def __init__(
        self,
        coverage_threshold: float = settings.DECOMPOSITION_MIN_COVERAGE,
        atomicity_threshold: float = settings.DECOMPOSITION_MIN_ATOMICITY,
        entailment_threshold: float = settings.DECOMPOSITION_MIN_ENTAILMENT,
        word_limit: int = settings.DECOMPOSITION_WORD_LIMIT,
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.atomicity_threshold = atomicity_threshold
        self.entailment_threshold = entailment_threshold
        self.word_limit = word_limit

    def evaluate(
        self,
        source_text: str,
        claims: list[AtomicClaim] | list[dict[str, Any]],
    ) -> DecompositionMetrics:
        """Compute coverage, atomicity, and entailment preservation metrics."""
        if not claims:
            return DecompositionMetrics(
                coverage_score=0.0,
                atomicity_score=0.0,
                entailment_preservation_score=0.0,
                dropped_entities=self._extract_salient_entities(source_text),
                is_high_quality=False,
            )

        # Normalize claims into list of strings and objects
        claim_texts = [c.claim_text if hasattr(c, "claim_text") else str(c.get("claim_text", c.get("text", ""))) for c in claims]
        combined_claims_text = " ".join(claim_texts)

        # 1. Coverage Calculation
        source_entities = self._extract_salient_entities(source_text)
        claim_entities = self._extract_salient_entities(combined_claims_text)

        source_tokens = self._extract_content_tokens(source_text)
        claim_tokens = self._extract_content_tokens(combined_claims_text)

        # Detect dropped entities
        dropped_entities: list[str] = []
        for ent in source_entities:
            # Check if entity (or its salient words) appears in any claim
            ent_norm = ent.lower().rstrip(".,;:!?")
            if not any(ent_norm in ct.lower() for ct in claim_texts):
                dropped_entities.append(ent)

        if source_tokens:
            token_recall = len(source_tokens.intersection(claim_tokens)) / len(source_tokens)
        else:
            token_recall = 1.0

        if source_entities:
            entity_recall = (len(source_entities) - len(dropped_entities)) / len(source_entities)
            # Weighted blend: 60% entity recall, 40% content token recall
            coverage_score = round(0.60 * entity_recall + 0.40 * token_recall, 4)
        else:
            coverage_score = round(token_recall, 4)

        coverage_score = max(0.0, min(1.0, coverage_score))

        # 2. Atomicity Calculation
        atomicity_scores: list[float] = []
        for ct in claim_texts:
            c_score = 1.0
            # Penalize coordinating conjunctions joining clauses
            conjunction_matches = 0
            for pat in _COORDINATING_CONJUNCTION_PATTERNS:
                conjunction_matches += len(pat.findall(ct))

            if conjunction_matches == 1:
                c_score = 0.60
            elif conjunction_matches >= 2:
                c_score = 0.30

            # Penalize word count > self.word_limit words for a single atomic claim
            word_count = len(ct.split())
            if word_count > self.word_limit:
                c_score = min(c_score, 0.50)

            atomicity_scores.append(c_score)

        atomicity_score = round(sum(atomicity_scores) / len(atomicity_scores), 4)

        # 3. Entailment Preservation Calculation
        # Precision: proportion of claim content words found in source
        if claim_tokens:
            precision = len(claim_tokens.intersection(source_tokens)) / len(claim_tokens)
        else:
            precision = 1.0

        recall = token_recall

        if precision + recall > 0:
            entailment_preservation = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            entailment_preservation = 0.0

        entailment_preservation = max(0.0, min(1.0, entailment_preservation))

        # 4. Overall Quality Flag
        is_high_quality = bool(
            coverage_score >= self.coverage_threshold
            and atomicity_score >= self.atomicity_threshold
            and entailment_preservation >= self.entailment_threshold
        )

        return DecompositionMetrics(
            coverage_score=coverage_score,
            atomicity_score=atomicity_score,
            entailment_preservation_score=entailment_preservation,
            dropped_entities=dropped_entities,
            is_high_quality=is_high_quality,
        )

    def evaluate_batch(
        self,
        batch: list[tuple[str, list[AtomicClaim] | list[dict[str, Any]]]],
    ) -> dict[str, float]:
        """Evaluate a batch of (source_text, claims) tuples and return aggregate metrics."""
        if not batch:
            return {
                "mean_coverage": 0.0,
                "mean_atomicity": 0.0,
                "mean_entailment": 0.0,
                "high_quality_rate": 0.0,
                "total_items": 0,
            }

        coverages: list[float] = []
        atomicities: list[float] = []
        entailments: list[float] = []
        high_quality_count = 0

        for source_text, claims in batch:
            metrics = self.evaluate(source_text, claims)
            coverages.append(metrics.coverage_score)
            atomicities.append(metrics.atomicity_score)
            entailments.append(metrics.entailment_preservation_score)
            if metrics.is_high_quality:
                high_quality_count += 1

        n = len(batch)
        return {
            "mean_coverage": round(sum(coverages) / n, 4),
            "mean_atomicity": round(sum(atomicities) / n, 4),
            "mean_entailment": round(sum(entailments) / n, 4),
            "high_quality_rate": round(high_quality_count / n, 4),
            "total_items": n,
        }

    def evaluate_corpus(
        self,
        corpus: dict[str, list[str]],
        extractor: Any,
    ) -> dict[str, Any]:
        """Evaluate a dictionary of domain -> sentences against an extractor and report domain-level and aggregate metrics."""
        domain_results: dict[str, dict[str, float]] = {}
        all_coverages: list[float] = []
        all_atomicities: list[float] = []
        all_entailments: list[float] = []
        total_hq = 0
        total_sentences = 0
        total_claims = 0

        for domain, sentences in corpus.items():
            domain_batch: list[tuple[str, list[AtomicClaim]]] = []
            for sent in sentences:
                res = extractor.extract_atomic_claims(sent)
                domain_batch.append((sent, res.claims))
                total_claims += len(res.claims)

            summary = self.evaluate_batch(domain_batch)
            domain_results[domain] = summary
            total_sentences += len(sentences)
            total_hq += int(round(summary["high_quality_rate"] * len(sentences)))
            all_coverages.append(summary["mean_coverage"])
            all_atomicities.append(summary["mean_atomicity"])
            all_entailments.append(summary["mean_entailment"])

        num_domains = max(1, len(corpus))
        aggregate = {
            "total_sentences": total_sentences,
            "total_claims": total_claims,
            "mean_coverage": round(sum(all_coverages) / num_domains, 4),
            "mean_atomicity": round(sum(all_atomicities) / num_domains, 4),
            "mean_entailment": round(sum(all_entailments) / num_domains, 4),
            "high_quality_rate": round(total_hq / max(1, total_sentences), 4),
        }

        return {
            "domains": domain_results,
            "aggregate": aggregate,
        }

    def _extract_content_tokens(self, text: str) -> set[str]:
        """Extract set of lowercase content words (ignoring stopwords and punctuation)."""
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) >= 3}

    def _extract_salient_entities(self, text: str) -> list[str]:
        """Extract proper nouns, alphanumeric entities, years, dates, and numbers from text."""
        entities: list[str] = []

        def add_entity(e: str):
            clean = e.strip().rstrip(".,;:!?")
            if clean and clean.lower() not in _STOPWORDS and clean not in entities:
                entities.append(clean)

        # Multi-word named entities with Roman numerals / numbers (e.g., Saturn V, Apollo 11, World War II)
        named_patterns = re.findall(
            r"\b[A-Z][a-zA-Z0-9]*(?:\s+(?:[A-Z][a-zA-Z0-9]*|[0-9]+|[IVXLCDM]+))+\b",
            text,
        )
        for p in named_patterns:
            add_entity(p)

        # Capitalized words of length >= 3 not starting a sentence
        for match in re.finditer(r"(?<=[a-z0-9,;\s]\s)\b[A-Z][a-z]{2,}\b", text):
            add_entity(match.group(0))

        # Acronyms (e.g. NASA, ESA, FBI, MIT)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        for a in acronyms:
            if a not in {"THE", "AND", "FOR", "NOT", "BUT"}:
                add_entity(a)

        # Quantities and measurements (e.g. 7.5 million, 100 kg)
        quantities = re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|pounds|tons|kg|km|miles|percent|%)\b",
            text,
            re.IGNORECASE,
        )
        for q in quantities:
            add_entity(q)

        # Years (e.g. 1969, 2024)
        years = re.findall(r"\b(?:1[7-9][0-9]{2}|20[0-9]{2})\b", text)
        for y in years:
            add_entity(y)

        # Significant numbers with commas or decimals
        numbers = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b", text)
        for n in numbers:
            add_entity(n)

        return entities
