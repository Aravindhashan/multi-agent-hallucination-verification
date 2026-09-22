"""Service orchestration for provider-independent LLM response generation."""

from datetime import datetime, timezone
from uuid import UUID

from app.config import settings
from app.schemas import LLMGeneratedResponse
from app.services.llm_client import BaseLLMClient
from app.services.mock_llm_client import MockLLMClient


def get_llm_client() -> BaseLLMClient:
    """Instantiate the LLM client selected by application configuration.

    New providers can be integrated by implementing ``BaseLLMClient`` and
    adding a provider mapping here; routers and response schemas remain stable.
    """

    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "mock":
        return MockLLMClient(model_name=settings.MOCK_MODEL_NAME)
    elif provider in ("openai", "real"):
        from app.services.real_llm_client import RealLLMClient

        return RealLLMClient(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            base_url=settings.LLM_BASE_URL,
        )

    # Fail explicitly rather than silently falling back to a mock provider.
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def _create_llm_client() -> BaseLLMClient:
    """Backward-compatible alias for get_llm_client."""
    return get_llm_client()


def generate_response(query_id: UUID, query: str) -> LLMGeneratedResponse:
    """Generate and package the base LLM response for a validated query.

    The service selects the configured provider, delegates text generation to
    its client, and adds provider-independent tracking metadata for downstream
    pipeline modules.
    """

    client = _create_llm_client()
    response_text = client.generate(query).strip()

    # Guard the service contract in case a future provider returns empty text.
    if not response_text:
        raise RuntimeError("The configured LLM client returned an empty response.")

    return LLMGeneratedResponse(
        query_id=query_id,
        response_text=response_text,
        model_name=client.model_name,
        generated_at=datetime.now(timezone.utc),
    )
