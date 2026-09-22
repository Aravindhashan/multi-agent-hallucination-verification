"""Deterministic mock implementation of the LLM client contract."""

from app.services.llm_client import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Generate predictable placeholder text for development and testing.

    This client performs no network calls and must be replaced by a real LLM
    client before production use.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the mock client with its configured model identifier."""

        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the configured mock model identifier."""

        return self._model_name

    def generate(self, query: str) -> str:
        """Return a deterministic, non-empty response for the supplied query."""

        # The explicit marker prevents this placeholder from being mistaken for
        # a production model response in logs or downstream development work.
        return (
            f"MOCK RESPONSE for query: '{query}'. "
            "This is deterministic placeholder content generated for testing. "
            "It can be passed to downstream claim extraction in a later module."
        )
