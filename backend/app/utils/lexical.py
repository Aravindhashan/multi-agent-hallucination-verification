"""Canonical lexical resources and stopwords shared across decomposition, pollution, and quality modules."""

import re

# Comprehensive canonical English stopwords combining evidence pollution and decomposition sets
CANONICAL_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "just", "may", "me", "might", "more",
    "most", "must", "mustn't", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "s", "same", "shall", "shan't", "she", "should",
    "shouldn't", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won't", "would", "wouldn't", "yet",
    "you", "your", "yours", "yourself", "yourselves",
})

# Standard abbreviations that should not trigger sentence boundary splits
COMMON_ABBREVIATIONS: frozenset[str] = frozenset({
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.",
    "u.s.", "u.k.", "u.n.", "e.g.", "i.e.", "vs.", "etc.",
    "approx.", "no.", "st.", "v.", "jan.", "feb.", "mar.",
    "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
    "nov.", "dec.", "fig.", "eq.", "inc.", "corp.", "co.", "ltd.", "llc.",
    "dept.", "univ.", "inst.",
})


def tokenize_content_words(
    text: str,
    min_length: int = 3,
    stopwords: frozenset[str] = CANONICAL_STOPWORDS,
) -> set[str]:
    """Extract set of lowercase content words, filtering stopwords and short tokens."""
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    return {w for w in words if w not in stopwords and len(w) >= min_length}


def is_stopword(word: str) -> bool:
    """Return whether a token is in the canonical stopword set."""
    return word.strip().lower() in CANONICAL_STOPWORDS
