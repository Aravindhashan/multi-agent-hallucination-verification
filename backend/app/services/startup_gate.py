"""Application startup validation and API key gating."""

import logging
from app.config import settings

logger = logging.getLogger("hallucination_verifier")


def validate_startup_configuration(
    provider: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Validate runtime environment and credentials before application startup.

    When running in real LLM mode (e.g. provider is 'openai' or 'real'), an API
    key must be present in the configuration/environment. If missing, startup is
    blocked and a descriptive error is raised without exposing any secret details.

    When running in 'mock' mode, startup proceeds without requiring external keys.
    """
    resolved_provider = (
        provider if provider is not None else settings.LLM_PROVIDER
    ).strip().lower()
    resolved_api_key = (
        api_key if api_key is not None else settings.LLM_API_KEY
    )

    if resolved_provider in ("openai", "real"):
        if not resolved_api_key or not resolved_api_key.strip():
            msg = (
                f"\n{'='*70}\n"
                f"[STARTUP ERROR] Real LLM provider '{resolved_provider}' is configured,\n"
                f"but no API key was found in the environment.\n"
                f"Please set the OPENAI_API_KEY or LLM_API_KEY environment variable.\n"
                f"{'='*70}\n"
            )
            print(msg)
            logger.error("Startup blocked: Real LLM provider requires an API key.")
            raise RuntimeError(
                f"Application startup blocked: Real LLM provider '{resolved_provider}' requires an API key."
            )

    return True

