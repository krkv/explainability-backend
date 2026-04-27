"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.core.constants import Model
from src.domain.interfaces.llm_provider import (
    AgentRole,
    StructuredGenerationConfig,
    build_response_schema,
)
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_gemini_provider import GoogleGeminiProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException
from src.services.llm.llm_factory import get_llm_provider, clear_providers


class TestHuggingFaceProvider:
    """Test cases for HuggingFaceProvider."""
    
    @pytest.fixture
    def mock_inference_client(self):
        """Create a mock InferenceClient."""
        client = Mock()
        client.chat_completion = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content='{"function_calls": [], "freeform_response": "Test"}'))
                ]
            )
        )
        return client
    
    @pytest.fixture
    def provider(self):
        """Create a HuggingFaceProvider instance."""
        with patch('src.services.llm.huggingface_provider.InferenceClient'):
            provider = HuggingFaceProvider(
                model_name="meta-llama/Llama-3.3-70B-Instruct",
                api_token="test-token"
            )
            provider._client = Mock()
            provider._client.chat_completion = AsyncMock(
                return_value=Mock(
                    choices=[
                        Mock(message=Mock(content='{"function_calls": [], "freeform_response": "Response"}'))
                    ]
                )
            )
            return provider
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        # Mock the internal method that returns a string
        provider._generate_hugging_face_response = AsyncMock(
            return_value='{"function_calls": [], "freeform_response": "Test response"}'
        )
        
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        
        response = await provider.generate_response(conversation, UseCase.ENERGY)
        
        assert isinstance(response, str)
        assert "function_calls" in response
        assert "freeform_response" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_with_function_calls(self, provider):
        """Test response generation with function calls."""
        # Mock the internal method that returns a string
        provider._generate_hugging_face_response = AsyncMock(
            return_value='{"function_calls": ["count_all()"], "freeform_response": "OK"}'
        )
        
        conversation = [{"role": "user", "content": "Count records"}]
        response = await provider.generate_response(conversation, UseCase.ENERGY)
        
        assert "count_all()" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_client_error(self, provider):
        """Test handling of client errors."""
        provider._client.chat_completion = AsyncMock(
            side_effect=Exception("Connection error")
        )
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.ENERGY)
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_content(self, provider):
        """Test handling of empty response content."""
        provider._client.chat_completion = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content=""))
                ]
            )
        )
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.ENERGY)

    @pytest.mark.asyncio
    async def test_generate_hugging_face_response_uses_custom_suggester_prompt(self, provider):
        """Test custom generation config overrides the system prompt for suggester calls."""
        provider._client = Mock()
        provider._client.chat = Mock()
        provider._client.chat.completions = Mock()
        provider._client.chat.completions.create = Mock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content='{"suggestions": ["A", "B", "C"]}'))
                ],
                usage=None,
            )
        )

        response = await provider._generate_hugging_face_response(
            conversation=[{"role": "user", "content": "What next?"}],
            usecase=UseCase.HEART.value,
            agent_role=AgentRole.SUGGESTER,
            generation_config=StructuredGenerationConfig(
                system_prompt="Suggest the next follow-up questions.",
                response_schema=build_response_schema(AgentRole.SUGGESTER),
            ),
        )

        assert '"suggestions"' in response
        create_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert create_kwargs["messages"][0]["role"] == "system"
        assert create_kwargs["messages"][0]["content"] == "Suggest the next follow-up questions."


class TestGoogleGeminiProvider:
    """Test cases for GoogleGeminiProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a GoogleGeminiProvider instance."""
        with patch('src.services.llm.google_gemini_provider.genai.Client'):
            provider = GoogleGeminiProvider(
                model_name="gemini-2.0-flash-exp",
                project_id="test-project",
                location="us-central1",
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


class TestLLMFactory:
    """Test cases for the LLM factory."""

    def teardown_method(self):
        """Reset singleton provider cache between tests."""
        clear_providers()

    def test_get_llm_provider_uses_hard_coded_google_project_and_location(self):
        """Test Gemini providers use the hard-coded project and location."""
        with patch('src.services.llm.llm_factory.GoogleGeminiProvider') as mock_provider_class:
            get_llm_provider(Model.GEMINI_2_0_FLASH)

        mock_provider_class.assert_called_once_with(
            model_name="gemini-2.0-flash-001",
            project_id="explainability-assistant",
            location="us-central1",
        )
