"""Versioned API route through which user queries enter the pipeline."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends

from app.dependencies import get_logger
from app.schemas import QueryRequest, QueryResponse
from app.services.session_store import get_session_store

router = APIRouter(prefix="/api/v1", tags=["queries"])


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    logger: logging.Logger = Depends(get_logger),
) -> QueryResponse:
    """Validate and accept a user query for the verification pipeline.

    Pydantic validates the incoming JSON body and normalizes the query before
    this function runs. The endpoint then assigns tracking metadata and returns
    the structured entry-point response. Later modules can consume this same
    validated query without changing the public request contract.
    """

    # Preserve caller-supplied query identifier if present; otherwise generate a new UUID4.
    query_id = request.query_id if request.query_id is not None else uuid4()
    # Store an explicit timezone-aware UTC timestamp to avoid ambiguous times.
    accepted_at = datetime.now(timezone.utc)

    # Avoid logging the raw user query because it may contain sensitive text.
    logger.info("Accepted query %s", query_id)

    # Record in session store for cross-module pipeline correlation
    session_store = get_session_store()
    session_store.create_session(query_id=query_id, query=request.query, created_at=accepted_at)

    return QueryResponse(
        query=request.query,
        query_id=query_id,
        timestamp=accepted_at,
    )
