"""FastAPI application entry point for the hallucination verifier."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.routers.analysis import router as analysis_router
from app.routers.claims import router as claims_router
from app.routers.defender import router as defender_router
from app.routers.evidence_quality import router as evidence_quality_router
from app.routers.generation import router as generation_router
from app.routers.prosecutor import router as prosecutor_router
from app.routers.query import router as query_router
from app.services.startup_gate import validate_startup_configuration


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run application startup validation and resource initialization."""
    validate_startup_configuration()
    yield


# Create the shared ASGI application and expose metadata in the OpenAPI docs.
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Register the versioned routes for Modules 1 through 6 and Evidence Quality.
app.include_router(query_router)
app.include_router(generation_router)
app.include_router(claims_router)
app.include_router(analysis_router)
app.include_router(evidence_quality_router)
app.include_router(defender_router)
app.include_router(prosecutor_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    """Return a simple message confirming that the API is available."""

    return {"message": "Hallucination Verification API"}
