"""Langfuse tracing helpers with a safe no-op fallback."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TraceContext:
    """Optional request attributes that improve trace correlation."""

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class _NoopObservation:
    """Fallback observation used when tracing is disabled."""

    def update(self, **_: Any) -> None:
        """Ignore updates when tracing is disabled."""


class LangfuseObservability:
    """Thin wrapper around Langfuse that keeps tracing optional."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._propagate_attributes: Optional[Any] = None
        self._import_attempted = False
        self._noop_observation = _NoopObservation()

    def _get_base_url(self) -> Optional[str]:
        return (
            os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or settings.langfuse_base_url
            or settings.langfuse_host
        )

    def _get_tracing_environment(self) -> Optional[str]:
        return (
            os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
            or settings.langfuse_tracing_environment
        )

    def _has_credentials(self) -> bool:
        return bool(
            (os.getenv("LANGFUSE_PUBLIC_KEY") or settings.langfuse_public_key)
            and (os.getenv("LANGFUSE_SECRET_KEY") or settings.langfuse_secret_key)
            and self._get_base_url()
        )

    def _ensure_client(self) -> Optional[Any]:
        if not self._has_credentials():
            return None

        if self._client is not None:
            return self._client

        if self._import_attempted:
            return None

        self._import_attempted = True

        try:
            base_url = self._get_base_url()
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or settings.langfuse_public_key
            secret_key = os.getenv("LANGFUSE_SECRET_KEY") or settings.langfuse_secret_key
            tracing_environment = self._get_tracing_environment()

            if public_key:
                os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
            if secret_key:
                os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
            if base_url:
                os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
            if tracing_environment:
                os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", tracing_environment)

            from langfuse import get_client, propagate_attributes

            self._client = get_client()
            self._propagate_attributes = propagate_attributes
            return self._client
        except ImportError:
            logger.warning(
                "Langfuse tracing disabled because the 'langfuse' package is not installed."
            )
            return None
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse tracing: %s", exc)
            return None

    def initialize(self) -> bool:
        """Initialize the Langfuse client if credentials are configured."""
        return self._ensure_client() is not None

    def start_observation(self, *, name: str, as_type: str = "span", **kwargs: Any) -> Any:
        """Start a Langfuse observation, or return a no-op context manager."""
        client = self._ensure_client()
        if client is None:
            return nullcontext(self._noop_observation)

        try:
            return client.start_as_current_observation(
                name=name,
                as_type=as_type,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Failed to start Langfuse observation '%s': %s", name, exc)
            return nullcontext(self._noop_observation)

    def propagate_attributes(self, **kwargs: Any) -> Any:
        """Propagate tags, metadata, and correlation ids to child observations."""
        client = self._ensure_client()
        attrs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, "", [], {})
        }

        if client is None or not attrs or self._propagate_attributes is None:
            return nullcontext()

        try:
            return self._propagate_attributes(**attrs)
        except Exception as exc:
            logger.warning("Failed to propagate Langfuse attributes: %s", exc)
            return nullcontext()

    def flush(self) -> None:
        """Flush pending Langfuse events."""
        client = self._ensure_client()
        if client is None:
            return

        try:
            client.flush()
        except Exception as exc:
            logger.warning("Failed to flush Langfuse events: %s", exc)


def slugify_trace_tag(value: str) -> str:
    """Convert a label into a stable tag fragment."""
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def truncate_for_trace(value: Optional[str], limit: int = 4_000) -> str:
    """Keep trace payloads readable and bounded."""
    if not value:
        return ""

    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized

    return f"{normalized[: limit - 3]}..."


def get_last_user_message(conversation: list[dict[str, str]]) -> str:
    """Return the most recent user message from a conversation."""
    for message in reversed(conversation):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def build_trace_metadata(
    *,
    usecase: str,
    model: str,
    conversation_length: int,
    trace_context: Optional[TraceContext] = None,
) -> dict[str, str]:
    """Build compact trace metadata values."""
    metadata = {
        "usecase": usecase,
        "model": model,
        "conversationLength": str(conversation_length),
    }

    if trace_context and trace_context.request_id:
        metadata["requestId"] = truncate_for_trace(trace_context.request_id, limit=200)

    return metadata


observability = LangfuseObservability()
