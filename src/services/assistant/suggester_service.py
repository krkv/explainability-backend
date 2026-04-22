"""Heart-only suggester service for dynamic follow-up prompts."""

import json
import re
from typing import Any, Dict, List, Optional

from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException
from src.core.logging_config import get_logger
from src.core.observability import (
    TraceContext,
    build_trace_metadata,
    get_last_user_message,
    observability,
    slugify_trace_tag,
    truncate_for_trace,
)
from src.domain.interfaces.llm_provider import AgentRole
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.services.llm.llm_factory import get_google_gemini_provider

logger = get_logger(__name__)

_FUNCTION_CALL_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(")
_SUGGESTER_MODEL_NAME = "gemini-3.1-flash-lite-preview"
_SUGGESTER_LOCATION = "global"


class SuggesterService:
    """Generate follow-up suggestions for the heart use case."""

    def __init__(self, usecase_registry: UseCaseRegistry):
        self.usecase_registry = usecase_registry

    async def generate_follow_ups(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase,
        limit: Optional[int] = None,
        exclude_suggestions: Optional[List[str]] = None,
        trace_context: Optional[TraceContext] = None,
    ) -> Optional[List[str]]:
        """
        Generate dynamic heart follow-up prompts.

        Returns `None` for non-heart use cases or when generation fails.
        """
        if usecase != UseCase.HEART:
            return None

        target_limit = max(1, min(limit or 5, 5))
        minimum_required = 1 if target_limit == 1 else 3

        latest_user_message = truncate_for_trace(get_last_user_message(conversation))
        trace_tags = [
            "suggester-api",
            f"usecase:{slugify_trace_tag(usecase.value)}",
            f"model:{slugify_trace_tag(_SUGGESTER_MODEL_NAME)}",
        ]
        trace_metadata = build_trace_metadata(
            usecase=usecase.value,
            model=_SUGGESTER_MODEL_NAME,
            conversation_length=len(conversation),
            trace_context=trace_context,
        )

        with observability.propagate_attributes(
            session_id=truncate_for_trace(trace_context.session_id, limit=200) if trace_context else None,
            user_id=truncate_for_trace(trace_context.user_id, limit=200) if trace_context else None,
            tags=trace_tags,
            metadata=trace_metadata,
        ):
            with observability.start_observation(
                name="assistant-follow-up-suggester",
                as_type="span",
            ) as root_span:
                root_span.update(
                    input={
                        "latest_user_message": latest_user_message,
                        "conversation_length": len(conversation),
                    }
                )

                try:
                    llm_provider = get_google_gemini_provider(
                        _SUGGESTER_MODEL_NAME,
                        location=_SUGGESTER_LOCATION,
                    )
                    llm_response = await llm_provider.generate_response(
                        conversation=conversation,
                        usecase=usecase,
                        agent_role=AgentRole.SUGGESTER,
                    )
                    suggestions = self._parse_and_validate_suggestions(
                        llm_response,
                        limit=target_limit,
                        minimum_required=minimum_required,
                        exclude_suggestions=exclude_suggestions,
                    )

                    root_span.update(
                        output={
                            "suggestion_count": len(suggestions),
                            "used_fallback": False,
                        }
                    )
                    return suggestions
                except Exception as exc:
                    logger.warning("Failed to generate dynamic follow-up suggestions: %s", exc)
                    root_span.update(
                        level="WARNING",
                        status_message=str(exc),
                        output={
                            "suggestion_count": 0,
                            "used_fallback": False,
                            "error": truncate_for_trace(str(exc)),
                        },
                    )
                    return None

    def _parse_and_validate_suggestions(
        self,
        llm_response: str,
        limit: int = 5,
        minimum_required: int = 3,
        exclude_suggestions: Optional[List[str]] = None,
    ) -> List[str]:
        """Parse and validate suggester JSON output."""
        try:
            response_data = json.loads(llm_response)
        except json.JSONDecodeError as exc:
            raise LLMProviderException(f"Invalid JSON response from suggester: {exc}") from exc

        suggestions = response_data.get("suggestions")
        if not isinstance(suggestions, list):
            raise LLMProviderException("Invalid suggester response: 'suggestions' must be a list")

        normalized: List[str] = []
        seen: set[str] = set()
        excluded = {
            suggestion.strip().lower()
            for suggestion in (exclude_suggestions or [])
            if isinstance(suggestion, str) and suggestion.strip()
        }

        for raw_suggestion in suggestions:
            if not isinstance(raw_suggestion, str):
                continue

            suggestion = raw_suggestion.strip()
            normalized_key = suggestion.lower()

            if not suggestion or normalized_key in seen:
                continue
            if normalized_key in excluded:
                continue
            if _FUNCTION_CALL_PATTERN.search(suggestion):
                continue
            if "```" in suggestion or "json" in normalized_key:
                continue

            seen.add(normalized_key)
            normalized.append(suggestion)

        if len(normalized) < minimum_required:
            raise LLMProviderException(
                f"Suggester returned fewer than {minimum_required} usable suggestions"
            )

        return normalized[:limit]
