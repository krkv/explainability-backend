"""OpenAI LLM provider implementation."""

import asyncio
import os
from importlib import import_module
from typing import Any, Dict, List, Optional

from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException
from src.core.logging_config import get_logger
from src.core.observability import get_last_user_message, observability, truncate_for_trace
from src.domain.interfaces.llm_provider import (
    AgentRole,
    LLMProvider,
    StructuredGenerationConfig,
)
from src.services.llm.generation_config_resolver import resolve_generation_config

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation using the Responses API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout_ms: int = 30_000,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout_ms = timeout_ms
        self._client = None
        self._openai_module = None
        self._is_available = False

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client lazily so local tests do not require the package."""
        try:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not configured")

            self._openai_module = import_module("openai")
            self._client = self._openai_module.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout_ms / 1000,
            )
            self._is_available = True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self._is_available = False

    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase,
        agent_role: AgentRole = AgentRole.ASSISTANT,
        generation_config: Optional[StructuredGenerationConfig] = None,
        generation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a structured JSON response from OpenAI."""
        if not self._is_available:
            raise LLMProviderException("OpenAI provider is not available")

        try:
            response = await self._generate_openai_response(
                conversation=conversation,
                usecase=usecase.value,
                agent_role=agent_role,
                generation_config=generation_config,
                generation_context=generation_context,
            )
            if not response or not response.strip():
                raise LLMProviderException("Empty response content from OpenAI API")

            logger.debug("Generated response from OpenAI: %s characters", len(response))
            return response
        except LLMProviderException:
            raise
        except Exception as e:
            if self._is_rate_limit_error(e):
                raise UpstreamRateLimitException(
                    "OpenAI rate limit exceeded. Please retry shortly."
                ) from e
            logger.error("OpenAI generation failed: %s", e)
            raise LLMProviderException(f"Failed to generate response: {e}") from e

    async def _generate_openai_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str,
        agent_role: AgentRole = AgentRole.ASSISTANT,
        generation_config: Optional[StructuredGenerationConfig] = None,
        generation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response using the OpenAI Responses API."""
        usecase_enum, resolved_generation_config = resolve_generation_config(
            conversation=conversation,
            usecase=usecase,
            agent_role=agent_role,
            generation_config=generation_config,
            generation_context=generation_context,
        )
        system_prompt = resolved_generation_config.system_prompt
        user_input = conversation[len(conversation) - 1]["content"]
        latest_user_message = truncate_for_trace(get_last_user_message(conversation))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        response_schema = self._normalize_schema_for_openai(
            resolved_generation_config.response_schema
        )
        response_format_name = response_schema.get("title", f"{agent_role.value}_response")

        logger.info(
            "Sending OpenAI request [model=%s usecase=%s agent_role=%s conversation_length=%s]",
            self.model_name,
            usecase_enum.value,
            agent_role.value,
            len(conversation),
        )

        with observability.start_observation(
            name="openai-responses-create",
            as_type="generation",
            model=self.model_name,
        ) as generation:
            generation.update(
                input={
                    "latest_user_message": latest_user_message,
                    "conversation_length": len(conversation),
                    "message_count": len(messages),
                    "response_format": "json_schema",
                },
                metadata={
                    "provider": "openai",
                    "usecase": usecase_enum.value,
                    "agent_role": agent_role.value,
                },
            )

            raw_response = await asyncio.to_thread(
                self._client.responses.create,
                model=self.model_name,
                input=messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": response_format_name,
                        "strict": True,
                        "schema": response_schema,
                    }
                },
            )

            response_text = self._extract_response_text(raw_response)
            update_payload = {
                "output": truncate_for_trace(response_text),
                "metadata": {
                    "provider": "openai",
                    "usecase": usecase_enum.value,
                    "agent_role": agent_role.value,
                },
            }

            usage_details = self._extract_usage_details(getattr(raw_response, "usage", None))
            if usage_details:
                update_payload["usage_details"] = usage_details

            generation.update(**update_payload)

            logger.info(
                "Received OpenAI response [model=%s usecase=%s agent_role=%s response_length=%s]",
                self.model_name,
                usecase_enum.value,
                agent_role.value,
                len(response_text),
            )
            return response_text

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from an OpenAI Responses API payload."""
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            output = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
                refusal = getattr(part, "refusal", None)
                if isinstance(refusal, str) and refusal.strip():
                    return refusal.strip()

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return ""

    def _normalize_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Add OpenAI strict-schema defaults to the existing Gemini-compatible schema."""
        def normalize(node: Any) -> Any:
            if isinstance(node, dict):
                normalized = {key: normalize(value) for key, value in node.items()}
                if normalized.get("type") == "object":
                    normalized.setdefault("additionalProperties", False)
                return normalized
            if isinstance(node, list):
                return [normalize(item) for item in node]
            return node

        return normalize(schema)

    def _extract_usage_details(self, usage: Any) -> Optional[Dict[str, int]]:
        """Map OpenAI usage payloads to Langfuse usage details."""
        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        usage_details = {}
        if input_tokens is not None:
            usage_details["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            usage_details["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            usage_details["total_tokens"] = int(total_tokens)

        return usage_details or None

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Best-effort detection for OpenAI rate limits."""
        error_text = str(error).upper()
        if "RATE LIMIT" in error_text:
            return True
        if "429" in error_text:
            return True

        if self._openai_module is not None:
            rate_limit_error = getattr(self._openai_module, "RateLimitError", None)
            if rate_limit_error and isinstance(error, rate_limit_error):
                return True

        status_code = getattr(error, "status_code", None)
        return status_code == 429

    def get_model_name(self) -> str:
        return self.model_name

    def is_available(self) -> bool:
        return self._is_available
