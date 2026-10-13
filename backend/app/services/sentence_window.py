"""Sentence window extraction service for DeBERTa cross-encoder NLI.

Replaces blunt token truncation with claim-aware sentence windowing.
Splits evidence passages into discrete sentences, scores them against the
target claim, and selects the most relevant 1–3 sentences as the NLI premise.

This accelerates DeBERTa inference on CPU by reducing sequence lengths from
256 tokens down to ~40–80 tokens (quadratic O(N^2) speedup), while dramatically
improving verification accuracy by preventing vital evidence from being truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Sequence

logger = logging.getLogger("hallucination_verifier.sentence_window")

# Common abbreviations to prevent false sentence boundary splitting
DEFAULT_ABBREVIATIONS: tuple[str, ...] = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "inc.",
    "ltd.",
    "corp.",
    "co.",
    "st.",
    "ave.",
    "rd.",
    "blvd.",
    "dept.",
    "univ.",
    "approx.",
    "est.",
    "no.",
    "vol.",
    "p.",
    "pp.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
    "al.",
    "gen.",
    "gov.",
    "sen.",
    "rep.",
    "capt.",
    "col.",
    "maj.",
    "lt.",
    "sgt.",
)

# Standard English stopwords to focus lexical overlap on informative content words
STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "don", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
})

DOT_PLACEHOLDER = "@@DOT@@"


@dataclass(frozen=True)
class SentenceWindow:
    """Represents a selected 1–3 sentence window extracted from an evidence passage."""

    text: str
    sentence_indices: list[int]
    score: float
    sentence_count: int
    total_passage_sentences: int

    @property
    def is_truncated_window(self) -> bool:
        """True if the passage had more sentences than the extracted window."""
        return self.total_passage_sentences > self.sentence_count


def split_into_sentences(text: str, abbreviations: Sequence[str] | None = None) -> list[str]:
    """Split an evidence passage into grammatically clean, individual sentences.

    Safely protects abbreviations (e.g. 'Dr.', 'U.S.', 'e.g.'), decimal numbers
    ('3.14'), and quotation punctuation without external NLP dependencies.

    Args:
        text: Raw evidence passage text.
        abbreviations: Optional collection of abbreviations to protect.

    Returns:
        List of trimmed sentence strings.
    """
    if not text or not str(text).strip():
        return []

    # Clean whitespace and linebreaks
    cleaned = re.sub(r"[\r\n]+", " ", str(text)).strip()
    if not cleaned:
        return []

    abbrev_list = abbreviations if abbreviations is not None else DEFAULT_ABBREVIATIONS

    # 1. Protect abbreviations by replacing periods with placeholder
    protected = cleaned
    for abb in abbrev_list:
        pattern = re.compile(r"\b" + re.escape(abb), re.IGNORECASE)
        protected = pattern.sub(lambda m: m.group(0).replace(".", DOT_PLACEHOLDER), protected)

    # 2. Protect periods between digits (e.g. 3.14, $10.50)
    protected = re.sub(r"(\d)\.(\d)", rf"\1{DOT_PLACEHOLDER}\2", protected)

    # 3. Split on sentence boundaries: [.!?] followed by optional quotes/brackets and whitespace
    splits = re.split(r"([.!?][\"'\)\]]?\s+)", protected)

    sentences: list[str] = []
    current_acc = ""
    for piece in splits:
        current_acc += piece
        if re.search(r"[.!?][\"'\)\]]?\s+$", current_acc):
            sent = current_acc.replace(DOT_PLACEHOLDER, ".").strip()
            if sent:
                sentences.append(sent)
            current_acc = ""

    if current_acc.strip():
        sent = current_acc.replace(DOT_PLACEHOLDER, ".").strip()
        if sent:
            sentences.append(sent)

    # Fallback: if no punctuation-based sentences were found, return the original text
    if not sentences and cleaned:
        return [cleaned]

    return sentences


def _tokenize(text: str) -> list[str]:
    """Extract alphanumeric token strings lowercased."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text)]


def score_sentence(claim: str, sentence: str) -> float:
    """Compute claim-relevance score for an individual sentence.

    Uses a weighted multi-feature lexical, entity, numerical, and n-gram model:
    - Content word overlap (stopword-filtered): 40%
    - Numbers, dates, and numerical values: 25%
    - Capitalized named entities / proper nouns: 25%
    - Bigram phrase matches: 10%

    Args:
        claim: Target factual claim text.
        sentence: Candidate sentence text.

    Returns:
        Relevance score in [0.0, 1.0].
    """
    if not claim or not sentence:
        return 0.0

    claim_tokens = _tokenize(claim)
    sent_tokens = _tokenize(sentence)

    if not claim_tokens or not sent_tokens:
        return 0.0

    # 1. Content word overlap
    claim_content = [t for t in claim_tokens if t not in STOP_WORDS]
    if not claim_content:
        claim_content = claim_tokens

    sent_content_set = set(t for t in sent_tokens if t not in STOP_WORDS)
    overlap_matches = [t for t in claim_content if t in sent_content_set]
    overlap_ratio = len(overlap_matches) / max(1, len(claim_content))

    # 2. Numerical / Date / Year matching bonus (vital for temporal & numerical claims)
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    sent_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))
    if claim_numbers:
        number_matches = claim_numbers.intersection(sent_numbers)
        number_bonus = len(number_matches) / len(claim_numbers)
    else:
        number_bonus = 0.0

    # 3. Capitalized Named Entity / Proper Noun matching bonus
    claim_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b", claim))
    sent_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b", sentence))
    if claim_entities:
        entity_matches = claim_entities.intersection(sent_entities)
        entity_bonus = len(entity_matches) / len(claim_entities)
    else:
        entity_bonus = 0.0

    # 4. Bigram phrase overlap bonus
    def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
        return set(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else set()

    claim_bigrams = _bigrams(claim_tokens)
    sent_bigrams = _bigrams(sent_tokens)
    if claim_bigrams:
        bigram_matches = claim_bigrams.intersection(sent_bigrams)
        bigram_bonus = len(bigram_matches) / len(claim_bigrams)
    else:
        bigram_bonus = 0.0

    total_score = (
        0.40 * overlap_ratio
        + 0.25 * number_bonus
        + 0.25 * entity_bonus
        + 0.10 * bigram_bonus
    )
    return round(min(1.0, max(0.0, total_score)), 4)


def select_best_sentence_window(
    claim: str,
    evidence_passage: str,
    min_sentences: int = 1,
    max_sentences: int = 3,
    strategy: str = "hybrid",
) -> SentenceWindow:
    """Select the best 1–3 sentences from an evidence passage relevant to the claim.

    Args:
        claim: Target factual claim to verify.
        evidence_passage: Evidence passage or document text.
        min_sentences: Minimum sentence count in window (default 1).
        max_sentences: Maximum sentence count in window (default 3).
        strategy: Window selection strategy ('hybrid', 'contiguous', 'top_k').

    Returns:
        SentenceWindow containing selected text, indices, relevance score, and counts.
    """
    sentences = split_into_sentences(evidence_passage)
    n = len(sentences)

    if n == 0:
        return SentenceWindow(
            text="",
            sentence_indices=[],
            score=0.0,
            sentence_count=0,
            total_passage_sentences=0,
        )

    # If passage already has <= max_sentences, preserve entire passage as-is
    if n <= max_sentences:
        all_text = " ".join(sentences)
        score = score_sentence(claim, all_text)
        return SentenceWindow(
            text=all_text,
            sentence_indices=list(range(n)),
            score=score,
            sentence_count=n,
            total_passage_sentences=n,
        )

    # Score each sentence individually
    sentence_scores = [score_sentence(claim, s) for s in sentences]

    # --- Strategy 1: Contiguous Window ---
    best_contig_window = (0, min(max_sentences, n))
    best_contig_score = -1.0

    for size in range(min_sentences, max_sentences + 1):
        for start in range(0, n - size + 1):
            end = start + size
            window_text = " ".join(sentences[start:end])
            direct_score = score_sentence(claim, window_text)
            sum_scores = sum(sentence_scores[start:end])
            # Penalize length slightly to prefer concise, high-density windows
            combined = direct_score * 0.70 + (sum_scores / (size**0.5)) * 0.30

            if combined > best_contig_score:
                best_contig_score = combined
                best_contig_window = (start, end)

    contig_indices = list(range(best_contig_window[0], best_contig_window[1]))
    contig_text = " ".join(sentences[i] for i in contig_indices)

    if strategy == "contiguous":
        return SentenceWindow(
            text=contig_text,
            sentence_indices=contig_indices,
            score=round(max(0.0, best_contig_score), 4),
            sentence_count=len(contig_indices),
            total_passage_sentences=n,
        )

    # --- Strategy 2: Top-K Individual Sentences (in chronological order) ---
    scored_indices = sorted(
        [(idx, score) for idx, score in enumerate(sentence_scores) if score > 0.0],
        key=lambda pair: pair[1],
        reverse=True,
    )

    if scored_indices:
        # Take top K up to max_sentences
        top_k_selected = sorted([idx for idx, _ in scored_indices[:max_sentences]])
    else:
        # Fallback to first max_sentences
        top_k_selected = list(range(min(max_sentences, n)))

    top_k_text = " ".join(sentences[i] for i in top_k_selected)
    top_k_direct_score = score_sentence(claim, top_k_text)
    top_k_sum_scores = sum(sentence_scores[i] for i in top_k_selected)
    top_k_score = top_k_direct_score * 0.70 + (top_k_sum_scores / (len(top_k_selected) ** 0.5)) * 0.30

    if strategy == "top_k":
        return SentenceWindow(
            text=top_k_text,
            sentence_indices=top_k_selected,
            score=round(max(0.0, top_k_score), 4),
            sentence_count=len(top_k_selected),
            total_passage_sentences=n,
        )

    # --- Strategy 3: Hybrid (Compare and pick higher scoring candidate) ---
    if top_k_score > best_contig_score:
        winner_text = top_k_text
        winner_indices = top_k_selected
        winner_score = top_k_score
    else:
        winner_text = contig_text
        winner_indices = contig_indices
        winner_score = best_contig_score

    logger.debug(
        "Extracted sentence window [%s / %d sents] (score: %.4f):\n  Claim: %s\n  Window: %s",
        strategy,
        len(winner_indices),
        winner_score,
        claim,
        winner_text,
    )

    return SentenceWindow(
        text=winner_text,
        sentence_indices=winner_indices,
        score=round(max(0.0, winner_score), 4),
        sentence_count=len(winner_indices),
        total_passage_sentences=n,
    )
