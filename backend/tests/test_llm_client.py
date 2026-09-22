"""Unit tests for RealLLMClient, MockLLMClient, and startup configuration gates."""

from unittest.mock import MagicMock
import openai
import pytest

from app.config import settings
from app.services.llm_client import (
    BaseLLMClient,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.services.mock_llm_client import MockLLMClient
from app.services.real_llm_client import RealLLMClient
from app.services.startup_gate import validate_startup_configuration


def test_real_client_implements_interface():
    """Verify RealLLMClient inherits from BaseLLMClient."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)
    assert isinstance(client, BaseLLMClient)
    assert client.model_name == settings.LLM_MODEL


def test_mock_client_compatibility():
    """Verify existing MockLLMClient remains functional without network calls."""
    mock_client = MockLLMClient(model_name="mock-test-v1")
    assert isinstance(mock_client, BaseLLMClient)
    assert mock_client.model_name == "mock-test-v1"
    response = mock_client.generate("What is the speed of light?")
    assert "MOCK RESPONSE" in response
    assert "speed of light" in response


def test_real_client_missing_api_key_raises():
    """Verify RealLLMClient refuses to initialize without an API key."""
    # Temporarily ensure no API key is in settings/env
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLMClient(api_key="")
    assert "LLM API key is required" in str(excinfo.value)


def test_real_client_empty_prompt_rejected():
    """Verify empty or whitespace-only prompt raises ValueError."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)

    with pytest.raises(ValueError):
        client.generate("")

    with pytest.raises(ValueError):
        client.generate("   \n\t  ")


def test_real_client_successful_response():
    """Verify clean text extraction from a valid provider response."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "  Paris is the capital of France.  "
    mock_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_response

    client = RealLLMClient(
        api_key="sk-test-fake-key",
        model="gpt-4o-mini",
        temperature=0.2,
        max_output_tokens=500,
        timeout=15.0,
        client=mock_openai,
    )

    result = client.generate("What is the capital of France?")
    assert result == "Paris is the capital of France."
    mock_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.2,
        max_tokens=500,
        timeout=15.0,
    )


def test_real_client_empty_response_raises():
    """Verify empty content from provider raises LLMResponseError."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "   "
    mock_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_response

    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)
    with pytest.raises(LLMResponseError) as excinfo:
        client.generate("Test prompt")
    assert "empty text response" in str(excinfo.value)


def test_real_client_authentication_error():
    """Verify provider 401/403 maps to LLMAuthenticationError."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    mock_openai.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid API key provided",
        response=MagicMock(status_code=401),
        body=None,
    )

    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)
    with pytest.raises(LLMAuthenticationError) as excinfo:
        client.generate("Test prompt")
    assert "authentication failed" in str(excinfo.value)


def test_real_client_timeout_error():
    """Verify provider timeout maps to LLMTimeoutError."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    mock_openai.chat.completions.create.side_effect = openai.APITimeoutError(
        request=MagicMock()
    )

    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)
    with pytest.raises(LLMTimeoutError) as excinfo:
        client.generate("Test prompt")
    assert "timed out" in str(excinfo.value)


def test_real_client_api_error():
    """Verify provider API error maps to LLMAPIError."""
    mock_openai = MagicMock(spec=openai.OpenAI)
    mock_openai.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )

    client = RealLLMClient(api_key="sk-test-fake-key", client=mock_openai)
    with pytest.raises(LLMAPIError) as excinfo:
        client.generate("Test prompt")
    assert "network connection error" in str(excinfo.value)


def test_startup_gate_mock_mode_allowed():
    """Verify startup gate allows application startup when in mock mode without API key."""
    assert validate_startup_configuration(provider="mock", api_key=None) is True


def test_startup_gate_real_mode_missing_key_blocked():
    """Verify startup gate blocks application startup when in real mode without API key."""
    with pytest.raises(RuntimeError) as excinfo:
        validate_startup_configuration(provider="openai", api_key=None)
    assert "startup blocked" in str(excinfo.value).lower()


def test_startup_gate_real_mode_with_key_allowed():
    """Verify startup gate allows real mode when API key is provided."""
    assert (
        validate_startup_configuration(provider="openai", api_key="sk-test-valid-key")
        is True
    )
