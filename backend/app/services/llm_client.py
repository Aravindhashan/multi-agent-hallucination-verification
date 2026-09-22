"""Provider-independent interface for large language model clients."""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base exception for all LLM client and communication errors."""

    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration or credentials are missing or invalid."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication or authorization with the LLM provider fails."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout limit."""

    pass


class LLMAPIError(LLMError):
    """Raised when the LLM provider API returns a non-success error response."""

    pass


class LLMResponseError(LLMError):
    """Raised when the LLM provider returns an empty, unparseable, or malformed response."""

    pass


class BaseLLMClient(ABC):
    """Define the contract implemented by all LLM provider clients.

    Future OpenAI, Hugging Face, or local-model integrations should inherit
    from this class. Keeping this contract provider-independent allows the
    service and API layers to remain unchanged when the provider changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider-specific model identifier."""

    @abstractmethod
    def generate(self, query: str) -> str:
        """Generate response text for a validated user query."""
