"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.core.logging_config import setup_logging, get_logger
from src.core.constants import APIEndpoints

# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="XAI LLM Chat Backend",
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
    logger.info("Starting XAI LLM Chat Backend")
    logger.info("FastAPI application initialized")
    logger.info(f"API endpoints: {APIEndpoints.READY}, {APIEndpoints.ASSISTANT_RESPONSE}")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks."""
    logger.info("Shutting down XAI LLM Chat Backend")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "XAI LLM Chat Backend",
        "version": "2.0.0",
        "docs": "/docs",
        "health": APIEndpoints.READY,
    }

