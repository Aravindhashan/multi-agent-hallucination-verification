"""Real LLM client implementation supporting OpenAI and OpenAI-compatible providers."""

import openai

from app.config import settings
from app.services.llm_client import (
    BaseLLMClient,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMResponseError,
    LLMTimeoutError,
)


class RealLLMClient(BaseLLMClient):
    """Production LLM client utilizing OpenAI or OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        """Initialize the client with configurable generation parameters and credentials."""
        self._model = model or settings.LLM_MODEL
        self._temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self._max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.LLM_MAX_OUTPUT_TOKENS
        )
        self._timeout = timeout if timeout is not None else settings.LLM_TIMEOUT
        self._base_url = base_url or settings.LLM_BASE_URL

        resolved_api_key = api_key or settings.LLM_API_KEY
        if client is not None:
            self._client = client
        else:
            if not resolved_api_key or not resolved_api_key.strip():
                raise LLMConfigurationError(
                    "LLM API key is required. Please set the OPENAI_API_KEY or "
                    "LLM_API_KEY environment variable."
                )
            self._client = openai.OpenAI(
                api_key=resolved_api_key.strip(),
                base_url=self._base_url,
                timeout=self._timeout,
            )

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def generate(self, query: str) -> str:
        """Generate response text for a validated prompt query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Prompt must contain at least one non-whitespace character.")

        normalized_prompt = query.strip()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": normalized_prompt}],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                timeout=self._timeout,
            )
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"LLM authentication failed: {e.message}") from e
        except openai.PermissionDeniedError as e:
            raise LLMAuthenticationError(f"LLM authorization failed: {e.message}") from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(
                f"LLM request timed out after {self._timeout}s: {e.message}"
            ) from e
        except openai.APIConnectionError as e:
            raise LLMAPIError(f"LLM network connection error: {e.message}") from e
        except openai.APIError as e:
            raise LLMAPIError(f"LLM provider API error: {e.message}") from e
        except Exception as e:
            raise LLMAPIError(f"Unexpected error during LLM generation: {e}") from e

        if not response or not getattr(response, "choices", None) or len(response.choices) == 0:
            raise LLMResponseError("LLM provider returned an empty or malformed choice list.")

        first_choice = response.choices[0]
        content = getattr(first_choice.message, "content", None)
        if content is None or not str(content).strip():
            raise LLMResponseError("LLM provider returned an empty text response.")

        return str(content).strip()
