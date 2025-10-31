"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_gemini_provider import GoogleGeminiProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException


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
        provider._client.models.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.HEART)
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_content(self, provider):
        """Test handling of empty response content."""
        provider._client.models.generate_content = AsyncMock(
            return_value=Mock(text="")
        )
        
        conversation = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(LLMProviderException):
            await provider.generate_response(conversation, UseCase.HEART)

