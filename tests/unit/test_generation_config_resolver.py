"""Unit tests for LLM generation config resolution."""

from unittest.mock import Mock, patch

from src.core.constants import UseCase
from src.domain.interfaces.llm_provider import (
    AgentRole,
    StructuredGenerationConfig,
    build_response_schema,
)
from src.services.llm.generation_config_resolver import resolve_generation_config


def test_build_response_schema_omits_gemini_unsupported_json_schema_fields():
    """Gemini response_schema rejects some Draft 7 keys, so we must not emit them."""
    assistant_schema = build_response_schema(AgentRole.ASSISTANT)
    suggester_schema = build_response_schema(AgentRole.SUGGESTER)

    for schema in (assistant_schema, suggester_schema):
        assert "$schema" not in schema
        assert "additionalProperties" not in schema

    assert "uniqueItems" not in suggester_schema["properties"]["suggestions"]


def test_resolve_generation_config_prefers_usecase_generation_config():
    """Role-aware use case configs should take precedence over registry prompt fallback."""
    usecase_instance = Mock()
    usecase_instance.get_generation_config.return_value = StructuredGenerationConfig(
        system_prompt="Heart suggester prompt",
        response_schema=build_response_schema(AgentRole.SUGGESTER),
    )
    registry = Mock()
    registry.get_usecase_instance.return_value = usecase_instance

    with patch(
        "src.services.service_factory.get_usecase_registry",
        return_value=registry,
    ):
        usecase_enum, generation_config = resolve_generation_config(
            conversation=[{"role": "user", "content": "What next?"}],
            usecase=UseCase.HEART.value,
            agent_role=AgentRole.SUGGESTER,
            generation_context={"latest_assistant_response": "Patient 10 is high risk."},
        )

    assert usecase_enum == UseCase.HEART
    assert generation_config.system_prompt == "Heart suggester prompt"
    assert generation_config.response_schema == build_response_schema(AgentRole.SUGGESTER)
    usecase_instance.get_generation_config.assert_called_once_with(
        conversation=[{"role": "user", "content": "What next?"}],
        agent_role=AgentRole.SUGGESTER,
        context={"latest_assistant_response": "Patient 10 is high risk."},
    )


def test_resolve_generation_config_falls_back_to_registry_prompt_and_default_schema():
    """When a use case instance is unavailable, resolver should fall back safely."""
    registry = Mock()
    registry.get_usecase_instance.return_value = None
    registry.get_system_prompt.return_value = "Fallback assistant prompt"

    with patch(
        "src.services.service_factory.get_usecase_registry",
        return_value=registry,
    ):
        usecase_enum, generation_config = resolve_generation_config(
            conversation=[{"role": "user", "content": "Hello"}],
            usecase=UseCase.ENERGY.value,
            agent_role=AgentRole.ASSISTANT,
        )

    assert usecase_enum == UseCase.ENERGY
    assert generation_config.system_prompt == "Fallback assistant prompt"
    assert generation_config.response_schema == build_response_schema(AgentRole.ASSISTANT)
    registry.get_system_prompt.assert_called_once_with(
        UseCase.ENERGY,
        [{"role": "user", "content": "Hello"}],
    )
