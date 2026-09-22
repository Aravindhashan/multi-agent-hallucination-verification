"""Explicit schema definitions for the reference evidence corpus."""

from datetime import date
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo


class CorpusDocument(BaseModel):
    """Represent one factual reference document and its provenance metadata.

    This model represents raw reference evidence without any retrieval-time or
    model-specific scores (e.g. BM25 scores, embeddings, NLI logits).
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Unique, non-empty identifier for the document.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Non-empty title of the document or passage.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Factual evidence passage.",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="Originating source, encyclopedia, or publisher.",
    )
    source_url: str | None = Field(
        default=None,
        description="Optional URL pointing to the originating source.",
    )
    domain: str | None = Field(
        default=None,
        description="Optional domain associated with the source.",
    )
    published_date: date | None = Field(
        default=None,
        description="Optional publication date of the evidence document.",
    )

    @field_validator("id", "title", "text", "source", mode="before")
    @classmethod
    def validate_non_empty_string(cls, value: object, info: ValidationInfo) -> str:
        """Ensure required string fields are non-empty and not whitespace-only."""
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must contain at least one non-whitespace character.")
        return normalized

    @field_validator("source_url", mode="before")
    @classmethod
    def validate_source_url(cls, value: object) -> str | None:
        """Normalize and validate source_url if provided."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("source_url must be a string or None.")
        normalized = value.strip()
        if not normalized:
            return None
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: '{normalized}'. Must include scheme (e.g. https) and domain.")
        return normalized

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain(cls, value: object) -> str | None:
        """Normalize domain if provided."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("domain must be a string or None.")
        normalized = value.strip().lower()
        return normalized if normalized else None


class Corpus(BaseModel):
    """Represent the structured reference corpus with metadata and document list."""

    version: str = Field(
        ...,
        min_length=1,
        description="Corpus specification or dataset version string.",
    )
    documents: list[CorpusDocument] = Field(
        ...,
        description="Collection of validated reference documents.",
    )

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> str:
        """Ensure version is a valid non-empty string."""
        if not isinstance(value, str):
            raise ValueError("version must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("version must contain at least one non-whitespace character.")
        return normalized

    @field_validator("documents")
    @classmethod
    def validate_unique_document_ids(cls, docs: list[CorpusDocument]) -> list[CorpusDocument]:
        """Ensure document IDs are unique across the corpus.

        Note: Near-duplicate or exact-duplicate text is explicitly permitted
        to enable downstream dependency and evidence pollution research.
        """
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []

        for doc in docs:
            if doc.id in seen_ids:
                duplicate_ids.append(doc.id)
            seen_ids.add(doc.id)

        if duplicate_ids:
            unique_dups = sorted(set(duplicate_ids))
            raise ValueError(
                f"Duplicate document ID(s) found in corpus: {unique_dups}. Every document ID must be unique."
            )

        return docs
