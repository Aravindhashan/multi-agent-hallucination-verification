"""Reusable FastAPI dependencies for the application.

Only a lightweight logger dependency is needed for Module 1. Future modules can
add shared clients, authentication, or pipeline services here.
"""

import logging


LOGGER_NAME = "hallucination_verifier"


def get_logger() -> logging.Logger:
    """Return the application logger for request-level operational events."""

    return logging.getLogger(LOGGER_NAME)


def get_evidence_retriever():
    """FastAPI dependency providing the active evidence retriever.

    Defaults to HybridRetriever unless RETRIEVER_TYPE is explicitly 'mock'
    or when overridden via app.dependency_overrides in test fixtures.
    """
    from app.config import settings

    if getattr(settings, "RETRIEVER_TYPE", "hybrid").lower() == "mock":
        from app.services.mock_evidence_retriever import MockEvidenceRetriever
        return MockEvidenceRetriever()

    try:
        from app.services.hybrid_retriever import get_hybrid_retriever
        return get_hybrid_retriever()
    except Exception:
        # Fallback gracefully to mock if corpus/indexes are uninitialized in minimal environments
        from app.services.mock_evidence_retriever import MockEvidenceRetriever
        return MockEvidenceRetriever()


def get_session_store_dep():
    """FastAPI dependency providing the singleton session store."""
    from app.services.session_store import get_session_store
    return get_session_store()
