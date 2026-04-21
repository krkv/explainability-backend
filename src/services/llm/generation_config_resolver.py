"""Helpers for resolving role-aware prompt/schema configs for LLM calls."""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.constants import UseCase
from src.domain.interfaces.llm_provider import (
    AgentRole,
    StructuredGenerationConfig,
    build_response_schema,
)


def resolve_generation_config(
    conversation: List[Dict[str, str]],
    usecase: Union[UseCase, str],
    agent_role: AgentRole = AgentRole.ASSISTANT,
    generation_config: Optional[StructuredGenerationConfig] = None,
    generation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[UseCase, StructuredGenerationConfig]:
    """Resolve the prompt/schema config for a provider request."""
    from src.services.service_factory import get_usecase_registry

    usecase_enum = usecase if isinstance(usecase, UseCase) else UseCase.from_string(usecase)
    if generation_config is not None:
        return usecase_enum, generation_config

    registry = get_usecase_registry()
    usecase_instance = getattr(registry, "get_usecase_instance", lambda _: None)(usecase_enum)

    if usecase_instance is not None and hasattr(usecase_instance, "get_generation_config"):
        return (
            usecase_enum,
            usecase_instance.get_generation_config(
                conversation=conversation,
                agent_role=agent_role,
                context=generation_context,
            ),
        )

    return (
        usecase_enum,
        StructuredGenerationConfig(
            system_prompt=registry.get_system_prompt(usecase_enum, conversation),
            response_schema=build_response_schema(agent_role),
        ),
    )
