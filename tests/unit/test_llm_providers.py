"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.core.constants import Model
from src.domain.interfaces.llm_provider import (
    AgentRole,
    StructuredGenerationConfig,
    build_response_schema,
)
from src.core.config import settings
from src.services.llm.google_gemini_provider import GoogleGeminiProvider
from src.services.llm.openai_provider import OpenAIProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException
from src.services.llm.llm_factory import get_llm_provider, clear_providers


class TestGoogleGeminiProvider:
    """Test cases for GoogleGeminiProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a GoogleGeminiProvider instance."""
        with patch('src.services.llm.google_gemini_provider.genai.Client'):
            provider = GoogleGeminiProvider(
                model_name="gemini-3.1-flash-lite-preview",
                project_id="test-project",
                location="global",
                api_key="test-key"
            )
            provider._client = Mock()
            provider._client.models.generate_content = AsyncMock(
                return_value=Mock(
                    text='{"function_calls": [], "freeform_response": "Response"}'
                )
            )
            return provider
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        # Mock the internal method that returns a string
        provider._generate_google_cloud_response = AsyncMock(
            return_value='{"function_calls": [], "freeform_response": "Test response"}'
        )
        
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        
        response = await provider.generate_response(conversation, UseCase.HEART)
        
        assert isinstance(response, str)
        assert "function_calls" in response
        assert "freeform_response" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_with_function_calls(self, provider):
        """Test response generation with function calls."""
        # Mock the internal method that returns a string
        provider._generate_google_cloud_response = AsyncMock(
            return_value='{"function_calls": ["show_one(id=1)"], "freeform_response": "Result"}'
        )
        
        conversation = [{"role": "user", "content": "Show record 1"}]
        response = await provider.generate_response(conversation, UseCase.HEART)
        
        assert "show_one(id=1)" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_client_error(self, provider):
        """Test handling of client errors."""
        provider._generate_google_cloud_response = AsyncMock(
            side_effect=Exception("API error")
        )
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.HEART)
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_content(self, provider):
        """Test handling of empty response content."""
        provider._generate_google_cloud_response = AsyncMock(return_value="")
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.HEART)

    @pytest.mark.asyncio
    async def test_generate_google_cloud_response_disables_automatic_function_calling(self, provider):
        """Test automatic function calling is disabled on Gemini requests."""
        mock_registry = Mock()
        mock_registry.get_system_prompt.return_value = "System prompt"
        mock_registry.get_usecase_instance.return_value = None
        provider._generate_sync = Mock(
            return_value=Mock(
                text='{"function_calls": [], "freeform_response": "Response"}',
                usage_metadata=None,
                function_calls=[],
                automatic_function_calling_history=[],
            )
        )

        conversation = [{"role": "user", "content": "Test"}]

        with patch('src.services.service_factory.get_usecase_registry', return_value=mock_registry):
            response = await provider._generate_google_cloud_response(conversation, UseCase.HEART.value)

        config = provider._generate_sync.call_args.args[1]

        assert "freeform_response" in response
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.mode == "NONE"

    @pytest.mark.asyncio
    async def test_generate_response_retries_resource_exhausted_then_succeeds(self, provider):
        """Test transient Vertex AI rate limits are retried."""
        provider._generate_google_cloud_response = AsyncMock(
            side_effect=[
                Exception("429 RESOURCE_EXHAUSTED"),
                '{"function_calls": [], "freeform_response": "Recovered"}',
            ]
        )

        conversation = [{"role": "user", "content": "Test"}]

        with patch('src.services.llm.google_gemini_provider.asyncio.sleep', new=AsyncMock()) as mock_sleep:
            response = await provider.generate_response(conversation, UseCase.HEART)

        assert "Recovered" in response
        assert provider._generate_google_cloud_response.await_count == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_response_resource_exhausted_raises_rate_limit_exception(self, provider):
        """Test persistent Vertex AI rate limits raise an explicit upstream exception."""
        provider._generate_google_cloud_response = AsyncMock(
            side_effect=Exception("429 RESOURCE_EXHAUSTED")
        )

        conversation = [{"role": "user", "content": "Test"}]

        with patch('src.services.llm.google_gemini_provider.asyncio.sleep', new=AsyncMock()):
            with pytest.raises(UpstreamRateLimitException) as exc_info:
                await provider.generate_response(conversation, UseCase.HEART)

        assert "RESOURCE_EXHAUSTED (429)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_google_cloud_response_uses_custom_suggester_schema(self, provider):
        """Test custom generation config overrides Gemini prompt/schema for suggester calls."""
        provider._generate_sync = Mock(
            return_value=Mock(
                text='{"suggestions": ["A", "B", "C"]}',
                usage_metadata=None,
                function_calls=[],
                automatic_function_calling_history=[],
            )
        )

        conversation = [{"role": "user", "content": "What should I ask next?"}]

        response = await provider._generate_google_cloud_response(
            conversation=conversation,
            usecase=UseCase.HEART.value,
            agent_role=AgentRole.SUGGESTER,
            generation_config=StructuredGenerationConfig(
                system_prompt="Suggest the next follow-up questions.",
                response_schema=build_response_schema(AgentRole.SUGGESTER),
            ),
        )

        config = provider._generate_sync.call_args.args[1]

        assert '"suggestions"' in response
        assert config.system_instruction == "Suggest the next follow-up questions."
        assert config.response_schema == build_response_schema(AgentRole.SUGGESTER)
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    @pytest.fixture
    def provider(self):
        """Create an OpenAIProvider instance."""
        mock_client = Mock()
        mock_client.responses.create = Mock(
            return_value=Mock(
                output_text='{"function_calls": [], "freeform_response": "Response"}',
                output=[],
                usage=None,
            )
        )
        mock_module = Mock()
        mock_module.OpenAI = Mock(return_value=mock_client)
        mock_module.RateLimitError = type("RateLimitError", (Exception,), {})

        with patch("src.services.llm.openai_provider.import_module", return_value=mock_module):
            provider = OpenAIProvider(
                model_name="gpt-5.4-mini",
                api_key="test-key",
            )
            provider._client = mock_client
            provider._openai_module = mock_module
            return provider

    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        provider._generate_openai_response = AsyncMock(
            return_value='{"function_calls": [], "freeform_response": "Test response"}'
        )

        response = await provider.generate_response(
            [{"role": "user", "content": "Hello"}],
            UseCase.HEART,
        )

        assert isinstance(response, str)
        assert "function_calls" in response
        assert "freeform_response" in response

    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_raises_upstream_exception(self, provider):
        """Test OpenAI rate limits map to explicit upstream errors."""
        provider._generate_openai_response = AsyncMock(
            side_effect=Exception("429 rate limit exceeded")
        )

        with pytest.raises(UpstreamRateLimitException):
            await provider.generate_response(
                [{"role": "user", "content": "Hello"}],
                UseCase.HEART,
            )

    @pytest.mark.asyncio
    async def test_generate_openai_response_uses_structured_outputs(self, provider):
        """Test OpenAI requests use strict JSON schema outputs."""
        response = await provider._generate_openai_response(
            conversation=[{"role": "user", "content": "Hello"}],
            usecase=UseCase.HEART.value,
            generation_config=StructuredGenerationConfig(
                system_prompt="Answer in JSON.",
                response_schema=build_response_schema(AgentRole.ASSISTANT),
            ),
        )

        create_kwargs = provider._client.responses.create.call_args.kwargs
        assert '"freeform_response"' in response
        assert create_kwargs["model"] == "gpt-5.4-mini"
        assert create_kwargs["input"] == [
            {"role": "system", "content": "Answer in JSON."},
            {"role": "user", "content": "Hello"},
        ]
        assert create_kwargs["text"]["format"]["type"] == "json_schema"
        assert create_kwargs["text"]["format"]["strict"] is True
        assert (
            create_kwargs["text"]["format"]["schema"]["additionalProperties"] is False
        )
        assert (
            create_kwargs["text"]["format"]["schema"]["properties"]["function_calls"]["items"]["type"]
            == "string"
        )

    def test_extract_response_text_prefers_first_output_content_over_output_text(self, provider):
        """Test provider avoids concatenated output_text when structured content is available."""
        response = Mock(
            output_text=(
                '{"function_calls": [], "freeform_response": "First"}'
                '{"function_calls": [], "freeform_response": "Second"}'
            ),
            output=[
                Mock(
                    content=[
                        Mock(
                            text='{"function_calls": [], "freeform_response": "First"}',
                            refusal=None,
                        )
                    ]
                )
            ],
        )

        extracted = provider._extract_response_text(response)

        assert extracted == '{"function_calls": [], "freeform_response": "First"}'


class TestLLMFactory:
    """Test cases for the LLM factory."""

    def teardown_method(self):
        """Reset singleton provider cache between tests."""
        clear_providers()

    def test_get_llm_provider_uses_hard_coded_google_project_and_location(self):
        """Test Gemini providers use the hard-coded project and location."""
        with patch('src.services.llm.llm_factory.GoogleGeminiProvider') as mock_provider_class:
            get_llm_provider(Model.GEMINI_3_1_FLASH_LITE_PREVIEW)

        mock_provider_class.assert_called_once_with(
            model_name="gemini-3.1-flash-lite-preview",
            project_id="explainability-assistant",
            location="global",
        )

    def test_get_llm_provider_supports_gpt_5_4_mini(self):
        """Test GPT-5.4 mini is routed through the OpenAI provider."""
        with patch('src.services.llm.llm_factory.OpenAIProvider') as mock_provider_class:
            get_llm_provider(Model.GPT_5_4_MINI)

        mock_provider_class.assert_called_once_with(
            model_name="gpt-5.4-mini",
            api_key=settings.openai_api_key,
        )
