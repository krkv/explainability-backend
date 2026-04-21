"""Heart-only suggester service for dynamic follow-up prompts."""

import json
import re
from typing import Any, Dict, List, Optional

from src.core.constants import Model, UseCase
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
from src.domain.entities.assistant_response import AssistantResponse
from src.domain.interfaces.llm_provider import AgentRole
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.services.assistant.follow_up_defaults import HEART_DEFAULT_FOLLOW_UPS
from src.services.llm.llm_factory import get_llm_provider

logger = get_logger(__name__)

_FUNCTION_CALL_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(")


class SuggesterService:
    """Generate follow-up suggestions for the heart use case."""

    def __init__(self, usecase_registry: UseCaseRegistry):
        self.usecase_registry = usecase_registry

    async def generate_follow_ups(
        self,
        conversation: List[Dict[str, str]],
        assistant_response: AssistantResponse,
        usecase: UseCase,
        model: Model,
        trace_context: Optional[TraceContext] = None,
    ) -> Optional[List[str]]:
        """
        Generate dynamic heart follow-up prompts.

        Returns `None` for non-heart use cases. For heart, always returns either
        validated dynamic suggestions or the heart defaults.
        """
        if usecase != UseCase.HEART:
            return None

        latest_user_message = truncate_for_trace(get_last_user_message(conversation))
        trace_tags = [
            "suggester-api",
            f"usecase:{slugify_trace_tag(usecase.value)}",
            f"model:{slugify_trace_tag(model.value)}",
        ]
        trace_metadata = build_trace_metadata(
            usecase=usecase.value,
            model=model.value,
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

                fallback_suggestions = list(HEART_DEFAULT_FOLLOW_UPS)

                try:
                    llm_provider = get_llm_provider(model)
                    llm_response = await llm_provider.generate_response(
                        conversation=conversation,
                        usecase=usecase,
                        agent_role=AgentRole.SUGGESTER,
                        generation_context={
                            "latest_assistant_response": assistant_response.freeform_response,
                            "function_calls": assistant_response.function_calls,
                            "parse": assistant_response.parse,
                        },
                    )
                    suggestions = self._parse_and_validate_suggestions(llm_response)

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
                            "suggestion_count": len(fallback_suggestions),
                            "used_fallback": True,
                            "error": truncate_for_trace(str(exc)),
                        },
                    )
                    return fallback_suggestions

    def _parse_and_validate_suggestions(self, llm_response: str) -> List[str]:
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

        for raw_suggestion in suggestions:
            if not isinstance(raw_suggestion, str):
                continue

            suggestion = raw_suggestion.strip()
            normalized_key = suggestion.lower()

            if not suggestion or normalized_key in seen:
                continue
            if _FUNCTION_CALL_PATTERN.search(suggestion):
                continue
            if "```" in suggestion or "json" in normalized_key:
                continue

            seen.add(normalized_key)
            normalized.append(suggestion)

        if len(normalized) < 3:
            raise LLMProviderException("Suggester returned fewer than 3 usable suggestions")

        return normalized[:5]
