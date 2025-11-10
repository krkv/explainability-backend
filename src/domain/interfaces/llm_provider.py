"""LLM provider interface for different language models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.constants import UseCase


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys
            usecase: The use case context (energy or heart)
            
        Returns:
            Raw response string from the LLM
            
        Raises:
            LLMProviderException: If the LLM call fails
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name string
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and configured.
        
        Returns:
            True if available, False otherwise
        """
        pass
