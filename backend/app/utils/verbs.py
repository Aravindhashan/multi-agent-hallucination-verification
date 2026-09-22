"""Consolidated verb lexicon and pre-compiled regex patterns for clause splitting, SPO extraction, and coreference resolution."""

import re

# 1. Copula and auxiliary linking verbs
COPULA_AUXILIARY_VERBS: frozenset[str] = frozenset({
    "is", "was", "are", "were", "be", "been", "being",
    "became", "become", "becomes", "remains", "remained",
    "seemed", "seems", "has been", "have been", "had been",
})

# 2. Transitive action and event verbs (past, participle, and present forms)
TRANSITIVE_ACTION_VERBS: frozenset[str] = frozenset({
    "created", "released", "discovered", "built", "founded", "launched",
    "developed", "invented", "wrote", "designed", "tested", "acquired",
    "produced", "verified", "landed", "walked", "authored", "introduced",
    "occurred", "lasted", "operated", "contains", "composed", "located",
    "established", "manufactured", "published", "won", "signed", "purchased",
    "organized", "orbited", "generated", "served", "named", "formed", "joined",
    "completed", "started", "began", "visited", "painted", "composed",
    "constructed", "engineered", "patented", "discovered", "directed",
    "changed", "transformed", "revolutionized", "altered", "influenced",
    "improved", "led", "caused", "announced", "reported", "demonstrated",
    "confirmed", "revealed", "showed", "proved", "earned", "received", "awarded",
    "stands", "stood", "lies", "lay", "sits", "sat", "measures", "measured",
    "weighs", "weighed", "reaches", "reached", "extends", "extended", "spans", "spanned",
    "features", "featured", "includes", "included", "holds", "held", "displays", "displayed",
    "opens", "opened", "spoke", "speaks", "stated", "states", "said", "says", "told",
    "delivered", "presented", "shared", "testified", "addressed", "argued", "noted", "remarked",
})

# 3. All verb forms including base lemmas for broad presence checks
ACTION_VERBS: frozenset[str] = frozenset(
    TRANSITIVE_ACTION_VERBS | {
        "create", "release", "discover", "build", "found", "launch",
        "develop", "invent", "write", "design", "test", "acquire",
        "produce", "verify", "land", "walk", "author", "introduce",
        "occur", "last", "operate", "contain", "compose", "locate",
        "establish", "manufacture", "publish", "win", "sign", "purchase",
        "organize", "orbit", "generate", "serve", "name", "form", "join",
        "complete", "start", "begin", "visit", "direct",
        "change", "transform", "revolutionize", "alter", "influence",
        "improve", "lead", "cause", "announce", "report", "demonstrate",
        "confirm", "reveal", "show", "prove", "earn", "receive", "award",
        "stand", "lie", "sit", "measure", "weigh", "reach", "extend",
        "span", "feature", "include", "hold", "display", "open",
        "speak", "state", "say", "tell", "deliver", "present", "share",
        "testify", "address", "argue", "note", "remark",
    }
)

ALL_VERBS: frozenset[str] = frozenset(COPULA_AUXILIARY_VERBS | ACTION_VERBS)

# 4. Canonical multi-word and single-word verb phrase patterns for SPO extraction
# Ordered longest/most specific first to ensure greedy match precedence
VERB_PHRASE_PATTERNS: list[str] = [
    r"\bbecame the first person to walk on\b",
    r"\bbecame the first person to\b",
    r"\bwas first released in\b",
    r"\bfirst released in\b",
    r"\bfirst released\b",
    r"\bwas created by\b",
    r"\bwas released in\b",
    r"\bwas discovered by\b",
    r"\bwas founded in\b",
    r"\bwas founded by\b",
    r"\bwas founded\b",
    r"\bfounded in\b",
    r"\bfounded by\b",
    r"\bwas designed by\b",
    r"\bwas built in\b",
    r"\bwas built by\b",
    r"\bbuilt in\b",
    r"\bbuilt by\b",
    r"\bwas introduced in\b",
    r"\bintroduced in\b",
    r"\bwas established in\b",
    r"\bwas acquired by\b",
    r"\bwas tested by\b",
    r"\bwas launched on\b",
    r"\bwas launched in\b",
    r"\blaunched on\b",
    r"\blaunched in\b",
    r"\blanded on\b",
    r"\bwalked on\b",
    r"\boccurred in\b",
    r"\boccurred at\b",
    r"\blasted for\b",
    r"\boperates at\b",
    r"\boperates in\b",
    r"\boperated at\b",
    r"\bis the capital of\b",
    r"\bis located in\b",
    r"\bis composed of\b",
    r"\bis approximately\b",
    r"\bgenerated\b",
    r"\bdeveloped\b",
    r"\binvented\b",
    r"\bdesigned\b",
    r"\bauthored\b",
    r"\bproduced\b",
    r"\bmanufactured\b",
    r"\bpublished\b",
    r"\bacquired\b",
    r"\bverified\b",
    r"\bdiscovered\b",
    r"\bcreated\b",
    r"\breleased\b",
    r"\bbuilt\b",
    r"\bfounded\b",
    r"\blaunched\b",
    r"\blanded\b",
    r"\bwalked\b",
    r"\bbecame\b",
    r"\boccurred\b",
    r"\blasted\b",
    r"\borbits\b",
    r"\bcontains\b",
    r"\bis\b",
    r"\bwas\b",
    r"\bare\b",
    r"\bwere\b",
]

# 5. Compiled helper patterns
_VERB_ALTERNATION = "|".join(re.escape(v) for v in sorted(ALL_VERBS, key=len, reverse=True))
_TRANSITIVE_ALTERNATION = "|".join(re.escape(v) for v in sorted(TRANSITIVE_ACTION_VERBS, key=len, reverse=True))

_HAS_VERB_REGEX = re.compile(rf"\b(?:{_VERB_ALTERNATION})\b", re.IGNORECASE)
_STARTS_WITH_VERB_REGEX = re.compile(rf"^(?:{_VERB_ALTERNATION})\b", re.IGNORECASE)
_TRANSITIVE_OBJ_IT_REGEX = re.compile(rf"(\b(?:{_TRANSITIVE_ALTERNATION})\s+)it\b", re.IGNORECASE)
_SUBJECT_SPLIT_REGEX = re.compile(
    rf"^([A-Z0-9][a-zA-Z0-9\s,\'\-]+?)(?:\s+(?:{_VERB_ALTERNATION}))\b",
    re.IGNORECASE,
)


def get_has_verb_pattern() -> re.Pattern[str]:
    """Return regex pattern checking whether a clause contains any known verb."""
    return _HAS_VERB_REGEX


def get_starts_with_verb_pattern() -> re.Pattern[str]:
    """Return regex pattern checking if clause begins with a predicate verb."""
    return _STARTS_WITH_VERB_REGEX


def get_transitive_obj_it_pattern() -> re.Pattern[str]:
    """Return regex pattern matching a transitive action verb followed by pronoun 'it'."""
    return _TRANSITIVE_OBJ_IT_REGEX


def get_subject_split_pattern() -> re.Pattern[str]:
    """Return regex pattern matching subject entity preceding an active or auxiliary verb."""
    return _SUBJECT_SPLIT_REGEX
