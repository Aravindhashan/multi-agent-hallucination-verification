"""Common data models for evidence retrieval results."""

from datetime import date
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Represent one ranked evidence candidate from an individual retrieval method."""

    document_id: str = Field(description="Unique identifier of the retrieved corpus document.")
    title: str = Field(description="Title of the retrieved document.")
    text: str = Field(description="Textual content of the evidence passage.")
    source: str = Field(description="Originating source name or publication.")
    source_url: str | None = Field(default=None, description="Optional URL of the source.")
    domain: str | None = Field(default=None, description="Optional domain of the source.")
    published_date: date | None = Field(default=None, description="Optional publication date.")
    score: float = Field(description="Raw or normalized similarity/matching score.")
    retrieval_method: str = Field(description="Retrieval method used ('BM25', 'DENSE', or 'HYBRID').")
    rank: int = Field(ge=1, description="1-based rank within this retrieval result set.")


class HybridCandidate(BaseModel):
    """Represent a deduplicated candidate from the merged hybrid retrieval process."""

    document_id: str = Field(description="Unique identifier of the retrieved corpus document.")
    title: str = Field(description="Title of the retrieved document.")
    text: str = Field(description="Textual content of the evidence passage.")
    source: str = Field(description="Originating source name or publication.")
    source_url: str | None = Field(default=None, description="Optional URL of the source.")
    domain: str | None = Field(default=None, description="Optional domain of the source.")
    published_date: date | None = Field(default=None, description="Optional publication date.")
    bm25_score: float | None = Field(default=None, description="BM25 score if retrieved by BM25.")
    bm25_rank: int | None = Field(default=None, ge=1, description="BM25 rank if retrieved by BM25.")
    dense_score: float | None = Field(default=None, description="Dense cosine score if retrieved by Dense retriever.")
    dense_rank: int | None = Field(default=None, ge=1, description="Dense rank if retrieved by Dense retriever.")
    matched_methods: list[str] = Field(
        default_factory=list,
        description="List of methods that found this document (e.g. ['BM25', 'DENSE']).",
    )
    combined_rank: int = Field(ge=1, description="1-based rank in the merged candidate set.")
