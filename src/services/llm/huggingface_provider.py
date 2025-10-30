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
    """HuggingFace LLM provider implementation using exact patterns from huggingface.py."""
    
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
        """Initialize the HuggingFace client exactly as in huggingface.py."""
        try:
            self._client = InferenceClient(
                self.model_name,
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
        Generate a response from the HuggingFace model using exact pattern from huggingface.py.
        
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
            # Use the exact same pattern as huggingface.py
            response = await self._generate_hugging_face_response(conversation, usecase.value)
            
            logger.debug(f"Generated response from HuggingFace: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise LLMProviderException(f"Failed to generate response: {e}")
    
    async def _generate_hugging_face_response(self, conversation: List[Dict[str, str]], usecase: str) -> str:
        """
        Generate response using the exact same pattern as huggingface.py
        
        Args:
            conversation: List of message dictionaries
            usecase: The use case context as string
            
        Returns:
            Generated response (structured JSON)
        """
        # Get system prompt from use case registry
        from src.services.service_factory import get_usecase_registry
        from src.core.constants import UseCase
        
        usecase_enum = UseCase.ENERGY if usecase == "energy" else UseCase.HEART
        registry = get_usecase_registry()
        system_prompt = registry.get_system_prompt(usecase_enum, conversation)
        
        # Get user input from the last message
        user_input = conversation[len(conversation) - 1]['content']
        
        # Format prompt exactly as in huggingface.py
        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  
  {system_prompt}
  
  <|eot_id|><|start_header_id|>user<|end_header_id|>
  
  {user_input}
  
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Generate response using async wrapper
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self._client.text_generation, 
            llama_prompt
        )
        return response.strip()
    
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
