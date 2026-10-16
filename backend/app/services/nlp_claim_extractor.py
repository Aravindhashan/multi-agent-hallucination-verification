"""Linguistic rule-based NLP Atomic Claim Extractor.

This module provides a deterministic, highly interpretable NLP pipeline to decompose
untrusted LLM responses into atomic, independently verifiable factual claims.
"""

from datetime import datetime, timezone
import re
from typing import Any

from app.services.claim_extractor import BaseClaimExtractor
from app.services.claim_models import AtomicClaim, ClaimExtractionResult, DecompositionMetrics
from app.services.decomposition_evaluator import DecompositionQualityEvaluator
from app.services.self_containment_reviser import AntecedentMemory, SelfContainmentReviser
from app.utils.lexical import COMMON_ABBREVIATIONS
from app.utils.verbs import (
    ALL_VERBS,
    VERB_PHRASE_PATTERNS,
    get_has_verb_pattern,
    get_starts_with_verb_pattern,
    get_subject_split_pattern,
)

# ---------------------------------------------------------------------------
# Linguistic Patterns & Dictionaries
# ---------------------------------------------------------------------------

# Abbreviations that should not trigger sentence splitting
_ABBREVIATIONS = COMMON_ABBREVIATIONS

_CONVERSATIONAL_PREFIXES = [
    re.compile(r"^(?:sure|certainly|of course|as an ai)[!.,;:]?\s*", re.IGNORECASE),
    re.compile(r"^(?:here (?:is|are)(?: what i found| the information| the answer)?)[!.,;:]?\s*", re.IGNORECASE),
    re.compile(r"^(?:based on (?:my|the) knowledge|according to (?:my|the) records)[!.,;:]?\s*", re.IGNORECASE),
    re.compile(r"^(?:in summary|to summarize|in conclusion|note that)[!.,;:]?\s*", re.IGNORECASE),
]

# Development / Mock response markers to ignore
_MOCK_RESPONSE_MARKER = "mock response for query"

# Common temporal regex patterns
_TEMPORAL_PATTERNS = [
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:in|during|around|circa|by|since|until)\s+(?:the\s+)?(?:1[0-9]{3}|20[0-9]{2})(?:s)?\b", re.IGNORECASE),
    re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b"),
]

# Numerical value patterns
_NUMERICAL_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),  # e.g., 299,792,458
    re.compile(r"\b\d+\.\d+\b"),                      # e.g., 3.14
    re.compile(r"\b\d+\b"),                           # e.g., 1969, 42
]

# Measurement units often attached to numbers
_UNIT_PATTERN = re.compile(
    r"\b(?:meters per second|m/s|km/s|km/h|miles per hour|mph|meters|kilometers|miles|kg|kilograms|grams|pounds|lbs|seconds|minutes|hours|days|years|percent|%|degrees|celsius|fahrenheit|kelvin|light years|au)\b",
    re.IGNORECASE,
)

# Role and title indicators for non-restrictive appositive identification
_ROLE_PATTERNS = re.compile(
    r"^(?:the\s+)?(?:former\s+|current\s+|co-)?(?:CEO|CTO|CFO|COO|founder|president|director|chairman|chairwoman|minister|prime\s+minister|secretary|governor|senator|judge|justice|professor|chancellor|author|creator|inventor|architect|lead\s+engineer|chief\s+scientist|head|leader|pioneer|physicist|chemist|biologist|astronomer|mathematician|economist|philosopher|historian)\b",
    re.IGNORECASE,
)

# Finite verb whitelist and inflection patterns for universal antecedent stitching (Pillar 1)
_FINITE_VERB_WHITELIST: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having", "does", "do", "did", "done",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must", "ought",
    "became", "become", "becomes", "remains", "remained", "seemed", "seems",
    "born", "built", "written", "wrote", "began", "begun", "made", "taken", "took", "given", "gave",
    "seen", "saw", "run", "ran", "held", "won", "lost", "sold", "bought", "brought", "thought", "taught",
    "caught", "felt", "left", "meant", "kept", "slept", "known", "knew", "grown", "grew", "shown", "flown",
    "flew", "drawn", "drew", "eaten", "ate", "fallen", "fell", "driven", "drove", "risen", "rose", "spoken",
    "spoke", "broken", "broke", "chosen", "chose", "frozen", "froze", "stolen", "stole", "forgotten",
    "forgot", "hidden", "hid", "got", "gotten", "found", "bound", "wound", "ground", "stood", "struck",
    "spent", "sent", "met", "led", "fed", "read", "shot", "shut", "spread", "split",
    "inhibits", "reduces", "stimulates", "causes", "binds", "prevents", "treats", "decreases", "increases",
    "functions", "works", "acts", "serves", "regulates", "secretes", "enhances", "leads", "contains",
    "consists", "features", "requires", "provides", "lowers", "targets", "expresses", "promotes",
    "induces", "mediates", "generates", "improves", "alters", "controls", "activates", "blocks",
})

_FINITE_VERB_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in sorted(_FINITE_VERB_WHITELIST, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_NON_VERB_ED_WORDS: frozenset[str] = frozenset({
    "red", "bed", "sled", "shed", "speed", "seed", "weed", "bleed", "feed",
    "need", "creed", "breed", "greed", "deed", "hundred", "naked", "sacred",
    "wicked", "crooked", "beloved", "rugged", "blessed", "learned", "ragged",
})

# Spatial / location indicators (generalized beyond hardcoded whitelist)
_SPATIAL_PREPOSITION_PATTERN = re.compile(
    r"\b(?i:at|in|on|near|above|across|orbiting)\s+(?:(?i:the)\s+)?"
    r"(?:"
    r"(?i:low\s+Earth\s+orbit|orbit|space|vacuum|Lagrange\s+Point\s+\d+|Moon|Earth|Mars|Jupiter|Venus|Sun-Earth\s+Lagrange\s+Point\s+\d+|CERN|UK|USA)"
    r"|"
    r"(?!(?i:January|February|March|April|May|June|July|August|September|October|November|December|\d{4})\b)[A-Z][a-zA-Z0-9\'-]+(?:\s+[A-Z][a-zA-Z0-9\'-]+)*"
    r")\b"
)

# Negative entity filters for spatial preposition matching (Rank 8 precision check)
_NON_SPATIAL_ENTITIES = {
    # Days of week & temporal adverbs
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "yesterday", "today", "tomorrow", "tonight", "weekend", "weekday",
    "morning", "afternoon", "evening", "night", "century", "decade", "millennium",
    # Months, seasons, quarters
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "spring", "summer", "autumn", "fall", "winter",
    "q1", "q2", "q3", "q4", "h1", "h2",
    # Document, filing, publication, communication entities
    "report", "study", "filing", "call", "earnings", "statement", "memo", "paper",
    "article", "document", "table", "figure", "chapter", "section", "appendix",
    "volume", "page", "chart", "graph", "transcript", "release", "index", "survey",
    "review", "audit", "brief", "book", "journal", "filings", "reports", "studies",
    "newsletter", "brochure", "whitepaper", "guidelines", "protocol", "docket",
    "patent", "treatise", "manual", "handbook", "press", "wire", "broadcast", "podcast",
    "footnote", "citation", "reference", "edition", "version", "clause", "subclause",
    # Well-known organizations / financial / tech
    "microsoft", "goldman sachs", "google", "apple", "amazon", "meta", "tesla",
    "netflix", "nvidia", "ibm", "intel", "openai", "pfizer", "moderna", "boeing",
    "lockheed", "jpmorgan", "citigroup", "morgan stanley", "barclays", "federal reserve",
    "bank of america", "treasury", "wall street", "harvard", "stanford", "mit",
    "biontech", "astrazeneca", "novartis", "roche", "johnson & johnson",
    # Events, meetings, conferences & historical conflicts
    "world war", "world war i", "world war ii", "wwi", "wwii", "covid", "covid-19",
    "olympics", "super bowl", "conference", "summit", "hearing", "meeting", "convention",
    "symposium", "forum", "assembly", "colloquium", "seminar", "workshop", "webinar",
    "expo", "exhibition", "festival", "tournament", "championship", "finals", "playoffs",
    "games", "ceremony", "gala", "election", "elections", "referendum", "session", "briefing",
    # Software & languages
    "python", "java", "linux", "windows", "unix", "english", "french", "spanish",
    "german", "chinese", "russian", "latin",
}

_NON_SPATIAL_SUFFIXES = {
    # Corporate & business suffixes
    "inc", "corp", "corporation", "llc", "ltd", "limited", "co", "company",
    "group", "holdings", "partners", "capital", "securities", "fund", "bank",
    "enterprises", "ventures", "technologies", "tech", "systems", "solutions",
    "software", "consulting", "services", "industries", "holdings", "network",
    # Institutional & educational suffixes
    "university", "college", "institute", "institution", "academy", "school",
    "foundation", "association", "organization", "agency", "ministry", "department",
    "committee", "commission", "council", "board", "court", "parliament", "congress", "senate",
    "hospital", "clinic", "laboratory", "laboratories", "labs", "pharmaceuticals", "pharma",
    "biotech", "therapeutics", "trust", "union", "federation",
    # Document, publication, and communication suffixes
    "report", "call", "filing", "study", "paper", "memo", "statement", "meeting",
    "release", "survey", "review", "brief", "audit", "guidelines", "protocol",
    "times", "post", "journal", "gazette", "chronicle", "tribune", "wire", "broadcast",
}


def _is_valid_spatial_entity(match_phrase: str) -> bool:
    """Validate that candidate preposition phrase references a spatial location (precision check)."""
    prep_match = re.match(
        r"^(?i:at|in|on|near|above|across|orbiting)\s+(?:(?i:the)\s+)?(.+)$",
        match_phrase.strip(),
    )
    if not prep_match:
        return False

    entity = prep_match.group(1).strip().strip("\"'“”‘’")
    entity_lower = entity.lower()

    # Reject singular and plural possessives (e.g., "Microsoft's", "Tuesday's", "analysts'")
    if "'s" in entity_lower or "’s" in entity_lower or entity_lower.endswith("'") or entity_lower.endswith("’"):
        return False

    # Reject exact non-spatial entities
    if entity_lower in _NON_SPATIAL_ENTITIES:
        return False

    # Reject if starting with a non-spatial entity or keyword (e.g. "Tuesday call", "Microsoft report", "Chapter 3")
    first_word = entity_lower.split()[0] if entity_lower.split() else ""
    if first_word in _NON_SPATIAL_ENTITIES or first_word in _NON_SPATIAL_SUFFIXES:
        return False

    for non_sp in _NON_SPATIAL_ENTITIES:
        if entity_lower == non_sp or entity_lower.startswith(non_sp + " "):
            return False

    # Reject corporate, institutional, document, or event suffixes / keywords
    words = re.findall(r"[a-zA-Z0-9]+", entity_lower)
    if any(w in _NON_SPATIAL_SUFFIXES for w in words):
        return False

    # Reject standalone quarters / fiscal terms (e.g., "Q1", "FY2023", "H2")
    if re.match(r"^(?:q[1-4]|h[1-2]|fy\s*\d{2,4})$", entity_lower):
        return False

    return True

# Common factual qualifiers
_QUALIFIER_TERMS = [
    re.compile(r"\bapproximately\b", re.IGNORECASE),
    re.compile(r"\bin vacuum\b", re.IGNORECASE),
    re.compile(r"\bpublicly\b", re.IGNORECASE),
    re.compile(r"\boriginally\b", re.IGNORECASE),
    re.compile(r"\bfirst person to walk on the Moon\b", re.IGNORECASE),
    re.compile(r"\bfirst\b", re.IGNORECASE),
    re.compile(r"\blast\b", re.IGNORECASE),
    re.compile(r"\bonly\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\blargest\b", re.IGNORECASE),
    re.compile(r"\bsmallest\b", re.IGNORECASE),
]

# Consolidated verb phrase patterns for Subject-Predicate-Object decomposition
_VERB_PHRASE_PATTERNS = VERB_PHRASE_PATTERNS


class NLPClaimExtractor(BaseClaimExtractor):
    """Deterministic NLP-based atomic claim extractor.

    Decomposes unstructured LLM outputs into structured, atomic claims using:
    1. Robust sentence segmentation with abbreviation protection.
    2. Compound and coordinate clause splitting.
    3. Cross-sentence subject-object coreference resolution heuristics.
    4. Deterministic entity, temporal anchor, numerical value, and qualifier extraction.
    5. Subject-Predicate-Object (SPO) structural decomposition.
    6. Normalization, deduplication, and deterministic ID indexing.
    """

    def extract_claims(self, response_text: str) -> list[str]:
        """Extract plain string atomic claims from response text."""
        result = self.extract_atomic_claims(response_text)
        return [c.claim_text for c in result.claims]

    def extract_atomic_claims(
        self, response_text: str, query: str = ""
    ) -> ClaimExtractionResult:
        """Decompose response text into structured AtomicClaim objects."""
        self._validate_input(response_text)

        cleaned_text = self._strip_conversational_boilerplate(response_text)
        sentences = self._segment_sentences(cleaned_text)
        sentences = self._stitch_unanchored_antecedents(sentences)

        raw_claims: list[dict[str, Any]] = []
        salient_entities: dict[str, str | None] = {"subject": None, "object": None}
        reviser = SelfContainmentReviser()
        antecedent_memory = AntecedentMemory()
        self._seed_antecedent_memory(cleaned_text, antecedent_memory)

        for sentence in sentences:
            sentence_text = sentence.strip()
            if not sentence_text or sentence_text.lower().startswith("mock response for query:"):
                continue

            # Split compound coordinate clauses and appositives
            clauses = self._split_clauses(sentence_text)

            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue

                # Resolve anaphoric pronouns using salient antecedents
                resolved_clause = self._resolve_coreference(clause, salient_entities)

                # Extract anchors and decomposition
                claim_data = self._analyze_clause(resolved_clause)

                # Update salient entities for downstream coreference
                if claim_data.get("subject"):
                    salient_entities["subject"] = claim_data["subject"]
                if claim_data.get("object"):
                    salient_entities["object"] = claim_data["object"]

                # Apply Submodule 3.6: Self-Containment Revision
                revised_claim_data = reviser.revise_claim(claim_data, antecedent_memory)

                raw_claims.append(revised_claim_data)

        # Deduplicate and format final AtomicClaim models
        atomic_claims = self._build_atomic_claims(raw_claims)

        # Submodule 3.7: Decomposition Quality Metrics
        evaluator = DecompositionQualityEvaluator()
        decomposition_metrics = evaluator.evaluate(response_text, atomic_claims)

        return ClaimExtractionResult(
            claims=atomic_claims,
            extraction_method="nlp_rule_based",
            raw_response_text=response_text,
            decomposition_metrics=decomposition_metrics,
            extracted_at=datetime.now(timezone.utc),
        )

    # -----------------------------------------------------------------------
    # Internal Pipeline Stages
    # -----------------------------------------------------------------------

    def _validate_input(self, text: Any) -> None:
        """Validate that input text is a non-empty string."""
        if not isinstance(text, str):
            raise ValueError(f"Expected response_text to be a string, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("Response text must contain at least one non-whitespace character.")

    def _strip_conversational_boilerplate(self, text: str) -> str:
        """Strip opening greetings, markdown bullets, and assistant preambles."""
        cleaned = text.strip()
        while True:
            matched = False
            # Strip bullets / list markers
            bullet_sub = re.sub(r"^[\*\-\•]\s+|\d+\.\s+", "", cleaned)
            if bullet_sub != cleaned:
                cleaned = bullet_sub.strip()
                matched = True

            for pattern in _CONVERSATIONAL_PREFIXES:
                if pattern.search(cleaned):
                    cleaned = pattern.sub("", cleaned).strip()
                    matched = True
            if not matched:
                break
        return cleaned

    def _segment_sentences(self, text: str) -> list[str]:
        """Segment text into individual sentences while preserving abbreviations and numbers."""
        # Replace decimal dots temporarily
        decimal_marker = "___DECIMAL___"
        protected = re.sub(r"(\d)\.(\d)", r"\1" + decimal_marker + r"\2", text)

        # Protect known abbreviations with exact casing
        for abbr in _ABBREVIATIONS:
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            protected = pattern.sub(lambda m: m.group(0).replace(".", "___DOT___"), protected)

        # Split on sentence boundaries (. ! ?) followed by space, or newlines preceding bullet markers / capitalized words
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+(?=[\*\-\•]|\d+[\.\)]|[A-Z])", protected)

        sentences: list[str] = []
        for s in raw_sentences:
            # Restore protected dots
            restored = s.replace("___DOT___", ".").replace(decimal_marker, ".").strip()
            if restored:
                sentences.append(restored)

        return sentences

    def _has_finite_verb(self, text: str) -> bool:
        """Inspect whether text contains a finite verb, auxiliary, or predicate inflection.

        Pillar 1 Universal Finite Verb Detector:
        has_finite_verb(s) = exists w in s matching aux, modal, or inflectional verb patterns.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # 1. Copula, auxiliary, aspectual, modal verbs
        if _FINITE_VERB_PATTERN.search(cleaned):
            return True

        # 2. General verb lexicon from app.utils.verbs
        if get_has_verb_pattern().search(cleaned):
            return True

        # 3. Regular inflectional verb endings: past (-ed) and 3rd person singular (-s/-es)
        tokens = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", cleaned)
        for tok in tokens:
            t_lower = tok.lower()
            if t_lower in _FINITE_VERB_WHITELIST or t_lower in ALL_VERBS:
                return True
            # Check 3rd-person singular present forms against base verb lexicon
            if len(t_lower) >= 3 and t_lower.endswith("s"):
                if t_lower[:-1] in ALL_VERBS or (t_lower.endswith("es") and t_lower[:-2] in ALL_VERBS):
                    return True
            # Check inflectional past -ed (excluding common non-verb words)
            if len(t_lower) >= 4 and t_lower.endswith("ed") and t_lower not in _NON_VERB_ED_WORDS:
                return True
            # Check common predicate verb endings (-izes, -ises, -ates, -bits, -duces, etc.)
            if re.search(
                r"(?:izes|ises|ates|bits|duces|vides|quires|tains|sists|tures|lates|cretes|motes|acts|serves|binds|treats|lowers|helps|leads|works|functions)$",
                t_lower,
            ):
                return True

        return False

    def _starts_with_active_verb(self, text: str) -> bool:
        """Check if clause begins with an active finite verb (rather than a passive participle)."""
        first_word_match = re.match(r"^([a-zA-Z]+)\b", text.strip())
        if not first_word_match:
            return False
        first_word = first_word_match.group(1).lower()

        # Auxiliaries/modals that start a predicate
        if first_word in {
            "is", "was", "are", "were", "has", "have", "had", "can", "will", "could", "would", "does", "did"
        }:
            return True

        # 3rd-person singular present verbs: ends in -s, -es, -ies (excluding nouns/determiners like 'this', 'as', etc.)
        if len(first_word) >= 3 and first_word.endswith("s") and first_word not in {"this", "as", "his", "yes", "plus", "thus", "minus"}:
            return True

        # Common base action verbs
        if first_word in {
            "run", "runs", "inhibit", "inhibits", "reduce", "reduces", "stimulate", "stimulates",
            "treat", "treats", "cause", "causes", "bind", "binds", "prevent", "prevents",
            "work", "works", "act", "acts", "serve", "serves", "contain", "contains",
            "feature", "features", "require", "requires", "provide", "provides",
        }:
            return True

        return False

    def _has_overt_subject_before_verb(self, text: str) -> bool:
        """Check if clause contains an overt subject preceding a finite verb."""
        t = text.strip()

        # 1. Strip infinitives "to <verb>" so they do not falsely register as main-clause finite verbs
        t_no_infinitive = re.sub(r"\bto\s+[a-z]+\b", "___INF___", t, flags=re.IGNORECASE)

        # 2. If the clause has no finite verb at all, it cannot have an overt subject before a verb
        if not self._has_finite_verb(t_no_infinitive):
            return False

        first_word_match = re.match(r"^([a-zA-Z]+)\b", t)
        if not first_word_match:
            return False
        first_word = first_word_match.group(1).lower()

        # If the first word is a verb/participle or preposition or conjunction, there is NO overt subject before it
        if self._has_finite_verb(first_word) or first_word in {
            "for", "in", "on", "at", "with", "by", "from", "as", "to", "and", "or", "but"
        }:
            return False

        # Check if subject split pattern matches a noun phrase preceding a verb
        subj_split_pat = get_subject_split_pattern()
        match = subj_split_pat.match(t_no_infinitive)
        if match:
            candidate_subj = match.group(1).strip()
            # If candidate subject ends in a preposition or conjunction, it is not a valid subject noun phrase
            last_subj_word = candidate_subj.split()[-1].lower() if candidate_subj.split() else ""
            if last_subj_word in {"to", "with", "for", "in", "on", "of", "by", "from", "at", "as", "and", "or", "but"}:
                return False

            c_tokens = candidate_subj.split()
            if len(c_tokens) >= 2 or candidate_subj.lower() in {"the", "a", "an", "this", "that"}:
                return True

        return False

    def _stitch_clause_with_subject(self, subject: str, clause: str) -> str:
        """Compose a grammatically complete proposition by binding subject with clause."""
        c = clause.strip()
        bullet_match = re.match(r"^([\*\-\•]\s+|\d+[\.\)]\s+)", c)
        bullet_prefix = bullet_match.group(1) if bullet_match else ""
        c_body = c[len(bullet_prefix):].strip()

        if not c_body:
            return f"{subject}."

        # Case 1: Clause already starts with the subject itself
        if c_body.lower().startswith(subject.lower()):
            return c_body

        # Case 2: Clause starts with a pronoun ("It is...", "He was...", "It has...", "They are...")
        pronoun_match = re.match(r"^(?:It|He|She|They)\b\s*", c_body, re.IGNORECASE)
        if pronoun_match:
            rest = c_body[pronoun_match.end():]
            return f"{subject} {rest}".strip()

        # Case 3: Check if clause already has its own complete overt subject before a finite verb
        if self._has_overt_subject_before_verb(c_body):
            return c_body

        # Case 4: Clause starts with an active finite verb
        if self._starts_with_active_verb(c_body):
            first_word_match = re.match(r"^([a-zA-Z]+)(.*)$", c_body, re.DOTALL)
            if first_word_match:
                verb = first_word_match.group(1).lower()
                rest = first_word_match.group(2)
                return f"{subject} {verb}{rest}".strip()
            return f"{subject} {c_body}".strip()

        # Case 5: Participle, adjective, noun complement, or preposition:
        first_word_match = re.match(r"^([a-zA-Z]+)(.*)$", c_body, re.DOTALL)
        if first_word_match:
            word = first_word_match.group(1).lower()
            rest = first_word_match.group(2)
            has_past_temporal = bool(re.search(r"\b(?:in|on|at|around|circa|by)\s+(?:1[0-9]{3}|20[0-2][0-9])\b", rest))
            copula = "was" if (word in {"born", "created", "founded", "built", "discovered", "invented", "released", "established"} and has_past_temporal) else "is"
            return f"{subject} {copula} {word}{rest}".strip()

        return f"{subject} is {c_body}".strip()

    def _stitch_unanchored_antecedents(self, sentences: list[str]) -> list[str]:
        """Stitch unanchored antecedent headers/noun phrases to succeeding predicate clauses.

        Pillar 1: Universal Syntactic Sentence Stitching via Grammar Heuristics.
        When text contains list headers, titles, or entity anchors without a finite verb (token count <= 5),
        buffer the antecedent subject and prepend it to succeeding dependent clauses:
        s_{i+1}' = s_i + " is " + s_{i+1} or s_i + " " + s_{i+1}
        """
        if not sentences:
            return []

        stitched: list[str] = []
        active_subject: str | None = None

        for raw_s in sentences:
            s = raw_s.strip()
            if not s:
                continue

            # Check for inline colon header e.g. "* Nateglinide: Used for type 2 diabetes."
            colon_match = re.match(r"^([\*\-\•]?\s*[A-Z0-9][a-zA-Z0-9\s\-\']{1,45}):\s+(.+)$", s)
            if colon_match:
                header_part = colon_match.group(1).strip()
                body_part = colon_match.group(2).strip()
                header_clean = re.sub(r"^[\*\-\•]\s+|\d+[\.\)]\s+", "", header_part).strip().rstrip(":")
                header_tokens = [t for t in header_clean.split() if t]
                if not self._has_finite_verb(header_clean) and len(header_tokens) <= 5:
                    active_subject = header_clean
                    s = self._stitch_clause_with_subject(active_subject, body_part)
                    stitched.append(s)
                    continue

            # Strip leading bullet/list markers for analysis
            core = re.sub(r"^[\*\-\•]\s+|\d+[\.\)]\s+", "", s).strip()
            core_clean = core.rstrip(".!?:")
            tokens = [t for t in re.split(r"\s+", core_clean) if t]
            has_verb = self._has_finite_verb(core)

            # If we already have an active_subject buffered:
            if active_subject:
                # Check whether this segment is a dependent clause / complement of active_subject
                first_w = core_clean.split()[0].lower() if core_clean.split() else ""
                is_dependent_or_complement = (
                    has_verb
                    or self._starts_with_active_verb(core)
                    or first_w in {
                        "the", "a", "an", "this", "that", "in", "on", "at", "for", "with",
                        "by", "from", "as", "first-line", "second-line", "common", "used", "monotherapy"
                    }
                    or len(tokens) > 3
                )
                if is_dependent_or_complement:
                    s_stitched = self._stitch_clause_with_subject(active_subject, s)
                    stitched.append(s_stitched)
                    continue
                else:
                    # New distinct entity header (e.g. "* Category.\n* Mount Everest.")
                    active_subject = core_clean
                    continue

            # Universal condition: Segment contains NO finite verb and token count <= 5
            if not has_verb and len(tokens) <= 5:
                # Unanchored Antecedent Subject
                active_subject = core_clean
                continue

            stitched.append(s)

        # If the text contained only an unanchored subject with no following sentence
        if not stitched and active_subject:
            stitched.append(f"{active_subject}.")

        return stitched

    def _decompose_participial_phrase(self, lead: str, interposed: str) -> list[str]:
        """Decompose a participial appositive into atomic propositions across all embedded facts."""
        # 1. Coordinate participles: e.g. "designed by Maurice Koechlin and built in 1889 by Gustave Eiffel"
        coord_parts = re.split(
            r"\s+and\s+(?=(?:was\s+)?(?:built|launched|discovered|founded|created|born|located|designed|developed|named|directed|written|painted|released|operated|established|commissioned|constructed|engineered|invented|patented|composed|published)\b)",
            interposed,
            flags=re.IGNORECASE,
        )
        if len(coord_parts) > 1:
            results: list[str] = []
            for part in coord_parts:
                results.extend(self._decompose_participial_phrase(lead, part.strip()))
            return results

        # 2. Extract leading verb
        v_match = re.match(
            r"^(?:was\s+)?(built|launched|discovered|founded|created|born|located|designed|developed|named|directed|written|painted|released|operated|established|commissioned|constructed|engineered|invented|patented|composed|published)\b",
            interposed,
            re.IGNORECASE,
        )
        if not v_match:
            return [f"{lead} was {interposed}"]

        verb = v_match.group(1).strip()
        verb_clause = f"{lead} was {verb}"
        rem = interposed[v_match.end():].strip()
        if not rem:
            return [verb_clause]

        # Extract distinct adjuncts from rem:
        # a. Temporal adjunct: "in 1889", "on July 20, 1969", "in 1920"
        temp_pat = re.compile(
            r"\b((?:in|on|at|circa|by|during)\s+(?:the\s+)?(?:\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?|\d{4}(?:s)?|\d{1,2}(?:st|nd|rd|th)\s+century))\b",
            re.IGNORECASE,
        )
        # b. Agent adjunct: "by Gustave Eiffel", "by Edwin Hubble"
        agent_pat = re.compile(
            r"\b(by\s+[A-Z][a-zA-Z\s\.\-]+?)(?=\s+(?:in|at|on|near|circa|under|with|aboard|using)\b|$)",
            re.IGNORECASE,
        )
        # c. Spatial adjunct: "at Mount Wilson Observatory", "in Paris", "in California"
        loc_pat = re.compile(
            r"\b((?:in|at|near|on)\s+(?:the\s+)?[A-Z][a-zA-Z0-9\'-]+(?:\s+[A-Z][a-zA-Z0-9\'-]+)*)\b"
        )
        # d. Vehicle / Instrument adjunct: "aboard an Ariane 5 rocket", "using the Hooker telescope"
        inst_pat = re.compile(
            r"\b((?:aboard|using|under|with)\s+(?:the\s+)?[A-Za-z0-9\s\.\-]+?)(?=\s+(?:in|at|on|by|circa|near)\b|$)",
            re.IGNORECASE,
        )

        found_adjuncts: list[str] = []
        agent_m = agent_pat.search(rem)
        agent_span = agent_m.span() if agent_m else None
        if agent_m:
            found_adjuncts.append(agent_m.group(1).strip())

        temp_m = temp_pat.search(rem)
        temp_span = temp_m.span() if temp_m else None
        if temp_m:
            found_adjuncts.append(temp_m.group(1).strip())

        inst_m = inst_pat.search(rem)
        inst_span = inst_m.span() if inst_m else None
        if inst_m:
            found_adjuncts.append(inst_m.group(1).strip())

        for loc_m in loc_pat.finditer(rem):
            l_span = loc_m.span()
            if agent_span and (agent_span[0] <= l_span[0] < agent_span[1] or l_span[0] <= agent_span[0] < l_span[1]):
                continue
            if temp_span and (temp_span[0] <= l_span[0] < temp_span[1] or l_span[0] <= temp_span[0] < l_span[1]):
                continue
            if inst_span and (inst_span[0] <= l_span[0] < inst_span[1] or l_span[0] <= inst_span[0] < l_span[1]):
                continue
            loc_str = loc_m.group(1).strip()
            # Exclude if it's a month or year
            if not re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4})\b", loc_str, re.IGNORECASE):
                if _is_valid_spatial_entity(loc_str):
                    found_adjuncts.append(loc_str)

        if len(found_adjuncts) >= 2:
            return [f"{verb_clause} {adj}" for adj in found_adjuncts]
        else:
            return [f"{lead} was {interposed}"]

    def _decompose_appositive_content(self, lead: str, interposed: str) -> list[str]:
        """Decompose an interposed appositive segment into discrete atomic propositions."""
        interposed = interposed.strip()
        if not interposed:
            return []

        # 1. Relative clauses starting with who, which, that, whose
        if re.match(r"^(?:which|who|whom|that)\b", interposed, re.IGNORECASE):
            return [re.sub(r"^(?:which|who|whom|that)\s+", f"{lead} ", interposed, flags=re.IGNORECASE)]
        if re.match(r"^whose\b", interposed, re.IGNORECASE):
            return [re.sub(r"^whose\s+", f"{lead}'s ", interposed, flags=re.IGNORECASE)]

        # 2. Noun phrase with embedded relative clause: e.g. "a Polish-born physicist who discovered radium in 1898"
        np_rel_match = re.match(
            r"^(?:a|an|the)\s+([a-zA-Z0-9\s\-]+?)\s+(?:who|which|that)\s+(.+)$",
            interposed,
            re.IGNORECASE,
        )
        if np_rel_match:
            head_noun = np_rel_match.group(1).strip()
            rel_clause = np_rel_match.group(2).strip()
            article = "an" if head_noun[0].lower() in "aeiou" else "a"
            return [
                f"{lead} is {article} {head_noun}",
                f"{lead} {rel_clause}",
            ]

        # 3. Participial phrase with embedded facts: "built in 1889 by Gustave Eiffel"
        starts_participle = bool(
            re.match(
                r"^(?:was\s+)?(?:built|launched|discovered|founded|created|born|located|designed|developed|named|directed|written|painted|released|operated|established|commissioned|constructed|engineered|invented|patented|composed|published)\b",
                interposed,
                re.IGNORECASE,
            )
        )
        if starts_participle:
            return self._decompose_participial_phrase(lead, interposed)

        # 4. Role or title appositive: "CEO of Microsoft since 2014" or "founder of Apple"
        if _ROLE_PATTERNS.match(interposed):
            return [f"{lead} is {interposed}"]

        # 5. Simple noun phrase appositive: "a prominent astronomer", "the capital of France"
        if re.match(r"^(?:a|an|the)\s+[a-zA-Z0-9\s\-]+$", interposed, re.IGNORECASE):
            return [f"{lead} is {interposed}"]

        # 6. Fallback if clause has a verb
        has_verb_pat = get_has_verb_pattern()
        if has_verb_pat.search(interposed):
            return [f"{lead} {interposed}"]

        return [f"{lead} is {interposed}"]

    def _split_appositives(self, text: str) -> list[str]:
        """Split non-restrictive appositives, participial clauses, and parentheticals into atomic claims."""
        # 1. Dash-delimited appositives: "Subject — [interposed] — [rest]" or "--" or "–"
        dash_match = re.search(r"\s*(?:—|--|–)\s*([^—–-]+?)\s*(?:—|--|–)\s*", text)
        if dash_match:
            lead = text[:dash_match.start()].strip().rstrip(",")
            dash_content = dash_match.group(1).strip()
            rest = text[dash_match.end():].strip().lstrip(",").rstrip(".!? ")
            if lead and dash_content and rest:
                main_claim = f"{lead} {rest}"
                sub_claims = self._decompose_appositive_content(lead, dash_content)
                if sub_claims:
                    return self._split_appositives(main_claim) + sub_claims

        # 2. Parenthetical phrases: e.g. "The Hubble Space Telescope (launched in 1990) orbits Earth."
        paren_match = re.search(r"\(([^)]+)\)", text)
        if paren_match:
            lead = text[:paren_match.start()].strip().rstrip(",")
            paren_content = paren_match.group(1).strip()
            rest = text[paren_match.end():].strip().lstrip(",").rstrip(".!? ")
            if lead and paren_content and rest:
                main_claim = f"{lead} {rest}"
                sub_claims = self._decompose_appositive_content(lead, paren_content)
                if sub_claims:
                    return self._split_appositives(main_claim) + sub_claims

        # 3. Comma-delimited appositives: "Subject, [interposed], [main predicate/rest]"
        trimmed = text.rstrip(".!? ").strip()
        comma_match = re.match(r"^([^,]+?),\s*([^,]+?),\s*(.+)$", trimmed)
        if comma_match:
            lead = comma_match.group(1).strip()
            interposed = comma_match.group(2).strip()
            rest = comma_match.group(3).strip()

            non_appositive_leads = {
                "although", "even though", "though", "while", "whereas", "if", "unless",
                "since", "because", "after", "before", "when", "whenever", "where",
                "wherever", "in addition", "moreover", "however", "furthermore", "therefore",
                "thus", "meanwhile", "as a result", "for example", "for instance",
                "first", "second", "third", "finally", "originally", "initially",
            }
            lead_lower = lead.lower()
            if not any(lead_lower.startswith(w + " ") or lead_lower == w for w in non_appositive_leads):
                has_verb_pat = get_has_verb_pattern()
                if has_verb_pat.search(rest) or get_starts_with_verb_pattern().match(rest) or re.search(r"\b[a-zA-Z]{3,}ed\b", rest):
                    sub_claims = self._decompose_appositive_content(lead, interposed)
                    if sub_claims:
                        main_claim = f"{lead} {rest}"
                        # Recurse on main_claim to handle any further embedded appositives
                        return self._split_appositives(main_claim) + sub_claims

        return [text]

    def _split_clauses(self, sentence: str) -> list[str]:
        """Split appositives, parentheticals, and coordinate clauses."""
        # Stage 1: Split appositive / parenthetical clauses (Rank 4)
        appositive_segments = self._split_appositives(sentence)

        valid_clauses: list[str] = []
        for segment in appositive_segments:
            trimmed = segment.rstrip(".!? ").strip()

            # Stage 2: Coordinate patterns: ' and ', ' but ', ' while ', ' ; '
            clause_delimiter = re.compile(
                r"(?:,\s*(?:and|but|while|whereas)\s+|\s+(?:and|but|while|whereas)\s+|;\s*)",
                re.IGNORECASE,
            )

            parts = clause_delimiter.split(trimmed)
            if len(parts) <= 1:
                valid_clauses.append(segment)
                continue

            # Extract leading subject from the first clause to propagate if second clause is a compound predicate
            subj_split_pat = get_subject_split_pattern()
            subject_match = subj_split_pat.match(parts[0])
            inherited_subject = subject_match.group(1).strip() if subject_match else None

            has_verb_pat = get_has_verb_pattern()
            starts_with_verb_pat = get_starts_with_verb_pattern()

            for idx, part in enumerate(parts):
                p = part.strip()
                if not p:
                    continue

                has_verb = bool(has_verb_pat.search(p))
                if not has_verb and idx > 0:
                    if valid_clauses:
                        valid_clauses[-1] = f"{valid_clauses[-1]} and {p}"
                    continue

                if idx > 0 and starts_with_verb_pat.match(p) and inherited_subject:
                    p = f"{inherited_subject} {p}"

                valid_clauses.append(p)

        return valid_clauses if valid_clauses else [sentence]

    def _resolve_coreference(
        self, clause: str, salient_entities: dict[str, str | None]
    ) -> str:
        """Resolve third-person pronouns to the active antecedent entities."""
        resolved = clause

        subj_entity = salient_entities.get("subject")
        obj_entity = salient_entities.get("object")

        # Resolve sentence-initial pronouns
        if subj_entity:
            # "He / She / It / They" at the beginning
            resolved = re.sub(
                r"^(?:He|She|It|They)\s+",
                f"{subj_entity} ",
                resolved,
                flags=re.IGNORECASE,
            )

        # Resolve object pronoun "it" following a verb if object entity is known
        if obj_entity:
            resolved = re.sub(
                r"(\b(?:created|released|discovered|visited|launched|wrote)\s+)it(\b)",
                rf"\g<1>{obj_entity}\g<2>",
                resolved,
                flags=re.IGNORECASE,
            )
        elif subj_entity:
            resolved = re.sub(
                r"(\b(?:created|released|discovered|visited|launched|wrote)\s+)it(\b)",
                rf"\g<1>{subj_entity}\g<2>",
                resolved,
                flags=re.IGNORECASE,
            )

        return resolved

    def _analyze_clause(self, clause: str) -> dict[str, Any]:
        """Extract SPO, temporal anchors, numerical values, spatial anchors, and qualifiers."""
        text = clause.strip()
        if not text.endswith((".", "!", "?")):
            text = f"{text}."

        # Extract temporal anchors
        temporal_anchors: list[str] = []
        for pattern in _TEMPORAL_PATTERNS:
            for match in pattern.finditer(text):
                val = match.group(0).strip()
                if val not in temporal_anchors:
                    temporal_anchors.append(val)

        temporal_anchor: str | None = temporal_anchors[0] if temporal_anchors else None

        # Extract spatial anchors (precision-checked)
        spatial_anchors: list[str] = []
        for sp_match in _SPATIAL_PREPOSITION_PATTERN.finditer(text):
            sp_val = sp_match.group(0).strip()
            if _is_valid_spatial_entity(sp_val) and sp_val not in spatial_anchors:
                spatial_anchors.append(sp_val)

        spatial_anchor: str | None = spatial_anchors[0] if spatial_anchors else None

        # Extract numerical values and measurements
        numerical_values: list[str] = []
        for num_pat in _NUMERICAL_PATTERNS:
            for match in num_pat.finditer(text):
                num_str = match.group(0)
                # Check for adjacent unit
                unit_match = _UNIT_PATTERN.search(text[match.end(): match.end() + 30])
                has_unit = unit_match is not None
                if has_unit:
                    num_val = f"{num_str} {unit_match.group(0).strip()}"
                else:
                    num_val = num_str

                # Disambiguate bare 4-digit years from numerical values (Rank 10)
                if not has_unit and num_str.isdigit() and len(num_str) == 4 and (1000 <= int(num_str) <= 2099):
                    if any(num_str in t for t in temporal_anchors):
                        continue

                # Disambiguate numbers that are part of proper noun entity designations (e.g. Apollo 11, Boeing 747)
                preceding_text = text[:match.start()].rstrip()
                if not has_unit and preceding_text and re.search(r"\b[A-Z][a-zA-Z]+$", preceding_text):
                    continue

                if not any(num_str in existing for existing in numerical_values):
                    numerical_values.append(num_val)

        numerical_value: str | None = numerical_values[0] if numerical_values else None

        # Extract qualifiers
        qualifiers: list[str] = []
        for q_pat in _QUALIFIER_TERMS:
            q_match = q_pat.search(text)
            if q_match:
                q_val = q_match.group(0).strip()
                if q_val not in qualifiers:
                    qualifiers.append(q_val)

        # Decompose Subject - Predicate - Object
        subject: str | None = None
        predicate: str | None = None
        obj: str | None = None

        for verb_pattern in _VERB_PHRASE_PATTERNS:
            v_match = re.search(verb_pattern, text, re.IGNORECASE)
            if v_match:
                predicate = v_match.group(0).strip()
                subj_candidate = text[: v_match.start()].strip()
                obj_candidate = text[v_match.end():].rstrip(".!? ").strip()

                if subj_candidate:
                    subject = subj_candidate
                if obj_candidate:
                    obj = obj_candidate
                break

        return {
            "claim_text": text,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "temporal_anchor": temporal_anchor,
            "numerical_value": numerical_value,
            "spatial_anchor": spatial_anchor,
            "temporal_anchors": temporal_anchors,
            "numerical_values": numerical_values,
            "spatial_anchors": spatial_anchors,
            "qualifiers": qualifiers,
        }

    def _build_atomic_claims(
        self, raw_claims: list[dict[str, Any]]
    ) -> list[AtomicClaim]:
        """Deduplicate claims and construct validated AtomicClaim models."""
        seen_texts: set[str] = set()
        atomic_claims: list[AtomicClaim] = []

        for item in raw_claims:
            claim_text = item["claim_text"].strip()
            # Normalize for deduplication comparison
            normalized_key = re.sub(r"\s+", " ", claim_text.lower().rstrip(".!?"))
            if not normalized_key or normalized_key in seen_texts:
                continue

            seen_texts.add(normalized_key)
            claim_id = f"c{len(atomic_claims) + 1}"

            atomic_claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    subject=item.get("subject"),
                    predicate=item.get("predicate"),
                    object=item.get("object"),
                    temporal_anchor=item.get("temporal_anchor"),
                    numerical_value=item.get("numerical_value"),
                    spatial_anchor=item.get("spatial_anchor"),
                    temporal_anchors=item.get("temporal_anchors", []),
                    numerical_values=item.get("numerical_values", []),
                    spatial_anchors=item.get("spatial_anchors", []),
                    qualifiers=item.get("qualifiers", []),
                    original_clause=item.get("original_clause"),
                    is_self_contained=item.get("is_self_contained", True),
                )
            )

        return atomic_claims

    def _seed_antecedent_memory(self, text: str, memory: AntecedentMemory) -> None:
        """Scan full text for prominent proper nouns and known entities to pre-seed antecedent memory."""
        # Find proper capitalized phrases like "The Wright brothers", "Wright Flyer", "Guido van Rossum"
        proper_names = re.findall(r"\b(?:The\s+)?[A-Z][a-z]+(?:\s+(?:van|de|von))?(?:\s+[A-Z][a-z]+)+\b", text)
        for name in proper_names:
            memory.register_entity(name)

        # Well-known acronyms and organizations
        acronyms = re.findall(r"\b(?:NASA|ESA|CERN|Apple(?:\s+Inc\.)?|Google|Microsoft|OpenAI|IBM|MIT|Apollo\s+\d+|STS-\d+|Python|Linux|Windows|iPhone)\b", text, re.IGNORECASE)
        for acr in acronyms:
            memory.register_entity(acr)
