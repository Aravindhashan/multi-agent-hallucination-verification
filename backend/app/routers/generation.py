"""API route for generating the pipeline's initial LLM response."""

from uuid import uuid4

from fastapi import APIRouter

from app.schemas import LLMGeneratedResponse, QueryRequest
from app.services.llm_service import generate_response
from app.services.session_store import get_session_store

router = APIRouter(prefix="/api/v1", tags=["generation"])


@router.post("/generate", response_model=LLMGeneratedResponse)
async def generate_base_response(request: QueryRequest) -> LLMGeneratedResponse:
    """Generate a base LLM response from a validated user query.

    If a query_id is provided from Module 1, it is preserved for pipeline
    correlation; otherwise a new UUID4 identifier is assigned.
    """

    query_id = request.query_id if request.query_id is not None else uuid4()
    result = generate_response(query_id=query_id, query=request.query)

    # Record generation output in the pipeline session store
    session_store = get_session_store()
    session_store.record_generation(
        query_id=query_id,
        query=request.query,
        response_text=result.response_text,
        model_name=result.model_name,
        generated_at=result.generated_at,
    )

    return result
