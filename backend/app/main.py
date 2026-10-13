"""FastAPI application entry point for the hallucination verifier."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.routers.adjudication import router as adjudication_router
from app.routers.analysis import router as analysis_router
from app.routers.claims import router as claims_router
from app.routers.debate import router as debate_router
from app.routers.defender import router as defender_router
from app.routers.evidence_quality import router as evidence_quality_router
from app.routers.generation import router as generation_router
from app.routers.prosecutor import router as prosecutor_router
from app.routers.query import router as query_router
from app.routers.verification import router as verification_router
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

# Set up static directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register the versioned routes for Modules 1 through 11.
app.include_router(query_router)
app.include_router(generation_router)
app.include_router(claims_router)
app.include_router(analysis_router)
app.include_router(evidence_quality_router)
app.include_router(defender_router)
app.include_router(prosecutor_router)
app.include_router(debate_router)
app.include_router(adjudication_router)
app.include_router(verification_router)


@app.get("/")
async def read_root():
    """Serve the interactive frontend UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
