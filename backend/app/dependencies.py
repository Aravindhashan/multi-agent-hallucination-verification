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

    Resolution semantics:
    - If RETRIEVER_TYPE is explicitly 'mock', returns MockEvidenceRetriever (for offline mock testing).
    - If RETRIEVER_TYPE is 'hybrid' (standard configuration):
      Attempts to initialize and return the singleton HybridRetriever.
      If HybridRetriever initialization fails, an explicit RuntimeError is raised
      (never silently substituting synthetic mock evidence for real corpus documents,
      which would compromise hallucination detection integrity).
    """
    from app.config import settings

    retriever_type = getattr(settings, "RETRIEVER_TYPE", "hybrid").lower()

    if retriever_type == "mock":
        from app.services.mock_evidence_retriever import MockEvidenceRetriever
        return MockEvidenceRetriever()

    if retriever_type in ("federated", "live_web", "web", "scifact", "fever"):
        from app.services.federated_evidence_retriever import get_federated_evidence_retriever
        return get_federated_evidence_retriever()

    try:
        from app.services.hybrid_retriever import get_hybrid_retriever
        return get_hybrid_retriever()
    except Exception as exc:
        logger = logging.getLogger(LOGGER_NAME)
        logger.error("HybridRetriever failed to initialize: %s", exc)
        raise RuntimeError(f"HybridRetriever backend unavailable: {exc}") from exc



def get_session_store_dep():
    """FastAPI dependency providing the singleton session store."""
    from app.services.session_store import get_session_store
    return get_session_store()
