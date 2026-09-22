"""Reusable prompt templates for LLM generation, claim extraction, and evidence reasoning."""

import re


class PromptTemplate:
    """Represent a parameterized prompt template with variable validation and injection boundaries."""

    def __init__(
        self,
        template: str,
        input_variables: list[str],
        description: str = "",
    ) -> None:
        """Initialize a prompt template with validation metadata."""
        self.template = template
        self.input_variables = input_variables
        self.description = description

    def format(self, **kwargs: object) -> str:
        """Substitute variables into template after validating presence and content."""
        missing = [
            var for var in self.input_variables if var not in kwargs or kwargs[var] is None
        ]
        if missing:
            raise ValueError(
                f"Missing required prompt template variables: {missing}. Expected: {self.input_variables}"
            )

        # Validate non-empty content
        for var in self.input_variables:
            val_str = str(kwargs[var]).strip()
            if not val_str:
                raise ValueError(
                    f"Prompt variable '{var}' must contain at least one non-whitespace character."
                )

        return self.template.format(**kwargs)


# 1. General LLM Prompt Template
GENERAL_QUERY_PROMPT = PromptTemplate(
    template="""You are an expert factual AI assistant. Answer the user question accurately, concisely, and factually based on verified knowledge.

[BEGIN USER QUERY]
{query}
[END USER QUERY]

Response:""",
    input_variables=["query"],
    description="General-purpose factual response generation prompt.",
)


# 2. Atomic Claim Extraction Prompt Template
CLAIM_EXTRACTION_PROMPT = PromptTemplate(
    template="""You are a precise linguistic fact-verification parser. Your task is to decompose the provided AI-generated response into distinct, independently verifiable atomic factual claims.

RULES:
1. Each extracted claim must be a standalone, self-contained factual assertion.
2. Resolve pronouns and co-references (e.g. replace "he", "it", "they" with the specific named entity).
3. Split compound or multi-clause sentences into separate single-fact claims.
4. Preserve all critical qualifiers: dates, years, numbers, quantities, named entities, locations, and conditions.
5. Exclude conversational filler, greetings, subjective opinions, or meta-commentary (e.g., "Here is what I found").
6. Output ONLY a valid JSON object matching the exact schema below, with no additional text or Markdown code fences.

JSON SCHEMA:
{{
  "claims": [
    "Atomic factual claim statement 1",
    "Atomic factual claim statement 2"
  ]
}}

[BEGIN USER QUERY]
{query}
[END USER QUERY]

[BEGIN RESPONSE DATA]
{response_text}
[END RESPONSE DATA]

JSON Output:""",
    input_variables=["query", "response_text"],
    description="Prompt template for atomic claim decomposition with structured JSON output.",
)


# 3. Evidence Reasoning Prompt Template
EVIDENCE_REASONING_PROMPT = PromptTemplate(
    template="""You are an analytical evidence reasoning assistant in an adversarial hallucination-verification pipeline.

Your task is to examine the logical and factual relationship between a candidate claim and a retrieved evidence passage.

INSTRUCTIONS:
1. Compare the factual assertions in the claim against the explicit content of the evidence.
2. Perform structured comparisons for:
   - Dates and temporal consistency (do the years and chronologies match?).
   - Numerical values and units (are quantities and measurements consistent?).
   - Named entities and relations (are subjects, actions, and objects aligned?).
3. Check for missing, restrictive, or conflicting qualifiers.
4. Clearly distinguish whether the evidence entails, contradicts, or is neutral towards the claim.
5. Base your analysis STRICTLY on the text inside the evidence data boundary. Do not assume facts not stated in the evidence.

[BEGIN CLAIM DATA]
{claim}
[END CLAIM DATA]

[BEGIN EVIDENCE DATA]
{evidence}
[END EVIDENCE DATA]

Reasoning Analysis:""",
    input_variables=["claim", "evidence"],
    description="Prompt template for structured claim-evidence factual reasoning.",
)
