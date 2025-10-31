"""Unit tests for assistant service."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from src.services.assistant.assistant_service import AssistantService
from src.core.constants import UseCase, Model
from src.core.exceptions import LLMProviderException, FunctionExecutionException
from src.domain.entities.assistant_response import AssistantResponse


class TestAssistantService:
    """Test cases for AssistantService."""
    
    @pytest.fixture
    def mock_function_executor(self):
        """Create a mock function executor."""
        executor = Mock()
        executor.execute_calls = Mock(return_value="Function execution result")
        return executor
    
    @pytest.fixture
    def mock_usecase_registry(self):
        """Create a mock use case registry."""
        registry = Mock()
        registry.get_system_prompt = Mock(return_value="Test prompt")
        return registry
    
    @pytest.fixture
    def assistant_service(self, mock_function_executor, mock_usecase_registry):
        """Create an AssistantService instance with mocked dependencies."""
        return AssistantService(
            function_executor=mock_function_executor,
            usecase_registry=mock_usecase_registry
        )
    
    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation."""
        return [
            {"role": "user", "content": "What is the dataset size?"},
            {"role": "assistant", "content": "The dataset has 100 records."},
        ]
    
    @pytest.mark.asyncio
    async def test_process_message_success_no_function_calls(
        self, assistant_service, sample_conversation, mock_function_executor
    ):
        """Test processing a message with no function calls."""
        # Mock LLM provider
        llm_response = '{"function_calls": [], "freeform_response": "Here is the answer"}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=llm_response)
            mock_get_provider.return_value = mock_provider
            
            response = await assistant_service.process_message(
                conversation=sample_conversation,
                usecase=UseCase.ENERGY,
                model=Model.LLAMA_3_3_70B
            )
            
            assert isinstance(response, AssistantResponse)
            assert response.function_calls == []
            assert response.freeform_response == "Here is the answer"
            assert response.parse == ""
            # Function executor should not be called
            mock_function_executor.execute_calls.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_message_success_with_function_calls(
        self, assistant_service, sample_conversation, mock_function_executor
    ):
        """Test processing a message with function calls."""
        llm_response = '{"function_calls": ["count_all()", "show_one(id=5)"], "freeform_response": "Results:"}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=llm_response)
            mock_get_provider.return_value = mock_provider
            
            response = await assistant_service.process_message(
                conversation=sample_conversation,
                usecase=UseCase.ENERGY,
                model=Model.LLAMA_3_3_70B
            )
            
            assert isinstance(response, AssistantResponse)
            assert len(response.function_calls) == 2
            assert response.function_calls == ["count_all()", "show_one(id=5)"]
            assert response.freeform_response == "Results:"
            assert response.parse == "Function execution result"
            # Function executor should be called
            mock_function_executor.execute_calls.assert_called_once_with(
                ["count_all()", "show_one(id=5)"],
                UseCase.ENERGY
            )
    
    @pytest.mark.asyncio
    async def test_process_message_function_execution_failure(
        self, assistant_service, sample_conversation, mock_function_executor
    ):
        """Test processing a message when function execution fails."""
        llm_response = '{"function_calls": ["invalid_func()"], "freeform_response": "Error occurred"}'
        mock_function_executor.execute_calls.side_effect = FunctionExecutionException("Function error")
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=llm_response)
            mock_get_provider.return_value = mock_provider
            
            response = await assistant_service.process_message(
                conversation=sample_conversation,
                usecase=UseCase.ENERGY,
                model=Model.LLAMA_3_3_70B
            )
            
            # Should still return response but with error in parse
            assert response.function_calls == ["invalid_func()"]
            assert "Error executing functions" in response.parse
    
    @pytest.mark.asyncio
    async def test_process_message_invalid_json_response(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message with invalid JSON response from LLM."""
        invalid_response = "Not valid JSON"
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=invalid_response)
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "Invalid JSON response" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_message_missing_function_calls_field(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message with missing function_calls field."""
        invalid_response = '{"freeform_response": "Response"}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=invalid_response)
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "Missing 'function_calls' field" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_message_missing_freeform_response_field(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message with missing freeform_response field."""
        invalid_response = '{"function_calls": []}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=invalid_response)
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "Missing 'freeform_response' field" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_message_wrong_type_for_function_calls(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message with wrong type for function_calls."""
        invalid_response = '{"function_calls": "not a list", "freeform_response": "Response"}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=invalid_response)
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "'function_calls' must be a list" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_message_wrong_type_for_freeform_response(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message with wrong type for freeform_response."""
        invalid_response = '{"function_calls": [], "freeform_response": 123}'
        
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(return_value=invalid_response)
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "'freeform_response' must be a string" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_message_llm_provider_exception(
        self, assistant_service, sample_conversation
    ):
        """Test processing a message when LLM provider raises an exception."""
        with patch('src.services.assistant.assistant_service.get_llm_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_response = AsyncMock(side_effect=LLMProviderException("LLM error"))
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMProviderException) as exc_info:
                await assistant_service.process_message(
                    conversation=sample_conversation,
                    usecase=UseCase.ENERGY,
                    model=Model.LLAMA_3_3_70B
                )
            assert "Failed to process message" in str(exc_info.value)

