"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_gemini_provider import GoogleGeminiProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException


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
        provider._generate_sync = Mock(
            return_value='{"function_calls": [], "freeform_response": "Response"}'
        )

        conversation = [{"role": "user", "content": "Test"}]

        with patch('src.services.service_factory.get_usecase_registry', return_value=mock_registry):
            response = await provider._generate_google_cloud_response(conversation, UseCase.HEART.value)

        config = provider._generate_sync.call_args.args[1]

        assert "freeform_response" in response
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True

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
