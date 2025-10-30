"""Assistant service for processing user messages and managing conversations."""

import json
import uuid
from typing import Dict, Any, List, Optional
from src.core.constants import UseCase, Model, MessageRole
from src.core.exceptions import LLMProviderException, FunctionExecutionException
from src.core.logging_config import get_logger
from src.domain.entities.message import Conversation, Message
from src.domain.entities.assistant_response import AssistantResponse
from src.domain.interfaces.function_executor import FunctionExecutor
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.services.llm.llm_factory import get_llm_provider

logger = get_logger(__name__)


class AssistantService:
    """
    Manages the overall assistant logic, including conversation flow,
    LLM interaction, and function execution.
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
        self._conversations: Dict[str, Conversation] = {}
    
    async def process_message(
        self,
        user_message: str,
        usecase: UseCase,
        model: Model,
        conversation_id: Optional[str] = None
    ) -> AssistantResponse:
        """
        Process a user message and generate an assistant response.
        
        Args:
            user_message: The user's message
            usecase: The use case context
            model: The LLM model to use
            conversation_id: Optional conversation ID for context
            
        Returns:
            AssistantResponse with function calls and freeform response
            
        Raises:
            LLMProviderException: If LLM generation fails
            FunctionExecutionException: If function execution fails
        """
        try:
            # Get or create conversation
            conversation = self._get_or_create_conversation(conversation_id)
            
            # Add user message
            conversation.add_message("user", user_message)
            
            # Get LLM provider
            llm_provider = get_llm_provider(model)
            
            # Prepare conversation for LLM (using original system prompts)
            llm_conversation = self._prepare_conversation_for_llm(conversation, usecase)
            
            # Generate response from LLM (should return structured JSON)
            llm_response = await llm_provider.generate_response(llm_conversation, usecase)
            
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
            
            # Add assistant response to conversation
            conversation.add_message("assistant", freeform_response)
            
            logger.info(f"Processed message for usecase {usecase.value}, model {model.value}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise LLMProviderException(f"Failed to process message: {e}")
    
    def _get_or_create_conversation(self, conversation_id: Optional[str]) -> Conversation:
        """
        Get existing conversation or create a new one.
        """
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        
        new_conversation_id = conversation_id if conversation_id else str(uuid.uuid4())
        conversation = Conversation(id=new_conversation_id, messages=[])
        self._conversations[new_conversation_id] = conversation
        logger.info(f"Created new conversation: {new_conversation_id}")
        return conversation
    
    def _prepare_conversation_for_llm(self, conversation: Conversation, usecase: UseCase) -> List[Dict[str, str]]:
        """
        Prepare conversation history for LLM input.
        The LLM providers will handle system prompts internally using original functions.
        """
        messages = []
        for message in conversation.messages:
            messages.append({
                "role": message.role.value,
                "content": message.content
            })
        return messages
    
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
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        """
        return self._conversations.get(conversation_id)
    
    def get_all_conversations(self) -> Dict[str, Conversation]:
        """
        Get all active conversations.
        """
        return self._conversations
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about conversations.
        """
        return {
            "total_conversations": len(self._conversations),
            "total_messages": sum(len(conv.messages) for conv in self._conversations.values()),
            "active_conversations": list(self._conversations.keys())
        }
