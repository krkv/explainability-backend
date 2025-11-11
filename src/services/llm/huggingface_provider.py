"""HuggingFace LLM provider implementation."""

import asyncio
import os
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient
from src.domain.interfaces.llm_provider import LLMProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class HuggingFaceProvider(LLMProvider):
    """HuggingFace LLM provider implementation using chat.completions.create() API."""
    
    def __init__(self, model_name: str, api_token: Optional[str] = None):
        """
        Initialize the HuggingFace provider.
        
        Args:
            model_name: Name of the HuggingFace model
            api_token: HuggingFace API token (optional)
        """
        self.model_name = model_name
        self.api_token = api_token or os.getenv('HF_TOKEN')
        self._client = None
        self._is_available = False
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the HuggingFace InferenceClient."""
        try:
            # Initialize client without model - model will be specified in API calls
            self._client = InferenceClient(
                token=self.api_token,
            )
            self._is_available = True
            logger.info(f"HuggingFace provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
            self._is_available = False
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase
    ) -> str:
        """
        Generate a response from the HuggingFace model using chat.completions.create() API.
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys
            usecase: The use case context (energy or heart)
            
        Returns:
            Raw response string from the LLM (structured JSON)
            
        Raises:
            LLMProviderException: If the LLM call fails
        """
        if not self._is_available:
            raise LLMProviderException("HuggingFace provider is not available")
        
        try:
            response = await self._generate_hugging_face_response(conversation, usecase.value)
            
            logger.debug(f"Generated response from HuggingFace: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise LLMProviderException(f"Failed to generate response: {e}")
    
    async def _generate_hugging_face_response(self, conversation: List[Dict[str, str]], usecase: str) -> str:
        """
        Generate response using chat.completions.create() API.
        
        Args:
            conversation: List of message dictionaries
            usecase: The use case context as string
            
        Returns:
            Generated response (structured JSON)
        """
        # Get system prompt from use case registry
        from src.services.service_factory import get_usecase_registry
        from src.core.constants import UseCase
        
        # Convert string usecase to enum (handles both frontend and backend formats)
        usecase_enum = UseCase.from_string(usecase)
        registry = get_usecase_registry()
        system_prompt = registry.get_system_prompt(usecase_enum, conversation)
        
        # Build messages array with system prompt and conversation history
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history (excluding the last user message if it's already in history)
        # The conversation list should already contain the full history
        for msg in conversation:
            # Skip system messages from conversation as we already have the system prompt
            if msg.get("role") != "system":
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Generate response using chat.completions.create()
        try:
            logger.debug(f"Calling HuggingFace API with model: {self.model_name}, messages count: {len(messages)}")
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self.model_name,
                messages=messages
            )
            
            logger.debug(f"Completion received: {completion}")
            
            # Extract response text from completion
            if not completion.choices or len(completion.choices) == 0:
                logger.error(f"No choices in completion: {completion}")
                raise LLMProviderException("No choices returned from HuggingFace API")
            
            message = completion.choices[0].message
            if not message:
                logger.error(f"No message in completion choices: {completion.choices[0]}")
                raise LLMProviderException("No message in completion choices")
            
            response = message.content
            if not response:
                logger.error(f"Empty response from HuggingFace. Completion object: {completion}, Message: {message}")
                raise LLMProviderException("Empty response content from HuggingFace API")
            
            logger.debug(f"HuggingFace response length: {len(response)} characters, first 200 chars: {response[:200]}")
            
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            elif response.startswith("```"):
                response = response[3:]   # Remove ```
            if response.endswith("```"):
                response = response[:-3]   # Remove closing ```
            response = response.strip()
            
            return response
            
        except LLMProviderException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"HuggingFace API error: {error_msg}, Type: {type(e)}")
            if "410" in error_msg or "api-inference.huggingface.co" in error_msg:
                logger.error(
                    "HuggingFace API endpoint deprecated. Please upgrade huggingface_hub: "
                    "pip install --upgrade huggingface_hub"
                )
            raise LLMProviderException(f"HuggingFace API call failed: {e}") from e
    
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name string
        """
        return self.model_name
    
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and configured.
        
        Returns:
            True if available, False otherwise
        """
        return self._is_available
