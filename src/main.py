"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.core.logging_config import setup_logging
from src.core.constants import APIEndpoints
from src.core.observability import observability
from src.core.logging_config import get_logger

# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Explainability Assistant Backend",
    description="LLM-powered assistant for ML model explanations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Startup tasks."""
    if observability.initialize():
        logger.info("Langfuse tracing enabled")
    else:
        logger.info(
            "Langfuse tracing disabled. Set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL to enable it."
        )


@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks."""
    observability.flush()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Explainability Assistant Backend",
        "version": "2.0.0",
        "docs": "/docs",
        "health": APIEndpoints.READY,
    }
