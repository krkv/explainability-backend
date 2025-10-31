"""Assistant service for processing user messages."""

import json
from typing import Dict, Any, List
from src.core.constants import UseCase, Model
from src.core.exceptions import LLMProviderException, FunctionExecutionException
from src.core.logging_config import get_logger
from src.domain.entities.assistant_response import AssistantResponse
from src.domain.interfaces.function_executor import FunctionExecutor
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.services.llm.llm_factory import get_llm_provider

logger = get_logger(__name__)


class AssistantService:
    """
    Manages the overall assistant logic, including LLM interaction and function execution.
    Conversation history is managed by the frontend and passed with each request.
    """
    
    def __init__(self, function_executor: FunctionExecutor, usecase_registry: UseCaseRegistry):
        """
        Initialize the AssistantService.
        
        Args:
            function_executor: The service responsible for executing functions.
            usecase_registry: The service responsible for managing use cases and their functions/prompts.
        """
        self.function_executor = function_executor
        self.usecase_registry = usecase_registry
    
    async def process_message(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase,
        model: Model
    ) -> AssistantResponse:
        """
        Process a conversation and generate an assistant response.
        
        Args:
            conversation: Full conversation history from frontend (list of message dicts with 'role' and 'content')
            usecase: The use case context
            model: The LLM model to use
            
        Returns:
            AssistantResponse with function calls and freeform response
            
        Raises:
            LLMProviderException: If LLM generation fails
            FunctionExecutionException: If function execution fails
        """
        try:
            # Log conversation for debugging
            logger.info("=== ASSISTANT SERVICE CALL ===")
            logger.info(f"Conversation length: {len(conversation)} messages")
            logger.info(f"Conversation history: {conversation}")
            
            # Get LLM provider
            llm_provider = get_llm_provider(model)
            
            # Generate response from LLM (should return structured JSON)
            # Pass conversation directly - providers will handle system prompts internally
            llm_response = await llm_provider.generate_response(conversation, usecase)
            
            # Parse structured JSON response
            structured_response = self._parse_structured_response(llm_response)
            
            # Extract function calls and freeform response from structured data
            function_calls = structured_response.get("function_calls", [])
            freeform_response = structured_response.get("freeform_response", "")
            
            # Execute function calls if any
            parse_result = ""
            if function_calls:
                try:
                    parse_result = self.function_executor.execute_calls(function_calls, usecase)
                    logger.info(f"Executed {len(function_calls)} function calls")
                except FunctionExecutionException as e:
                    logger.error(f"Function execution failed: {e}")
                    parse_result = f"Error executing functions: {e}"
            
            # Create assistant response
            response = AssistantResponse(
                function_calls=function_calls,
                freeform_response=freeform_response,
                parse=parse_result
            )
            
            logger.info(f"Processed message for usecase {usecase.value}, model {model.value}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise LLMProviderException(f"Failed to process message: {e}")
    
    def _parse_structured_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse structured JSON response from LLM.
        
        Args:
            llm_response: Raw LLM response (should be JSON)
            
        Returns:
            Parsed response dictionary
            
        Raises:
            LLMProviderException: If JSON parsing fails
        """
        try:
            # Parse JSON response
            response_data = json.loads(llm_response)
            
            # Validate required fields
            if "function_calls" not in response_data:
                raise ValueError("Missing 'function_calls' field in response")
            if "freeform_response" not in response_data:
                raise ValueError("Missing 'freeform_response' field in response")
            
            # Validate types
            if not isinstance(response_data["function_calls"], list):
                raise ValueError("'function_calls' must be a list")
            if not isinstance(response_data["freeform_response"], str):
                raise ValueError("'freeform_response' must be a string")
            
            return response_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise LLMProviderException(f"Invalid JSON response from LLM: {e}")
        except ValueError as e:
            logger.error(f"Invalid response structure: {e}")
            raise LLMProviderException(f"Invalid response structure: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise LLMProviderException(f"Failed to parse LLM response: {e}")
    
