"""Use case registry interface for managing different use cases."""

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any, List, Optional

from src.core.constants import UseCase
from src.domain.interfaces.llm_provider import AgentRole, StructuredGenerationConfig


class UseCaseRegistry(ABC):
    """Abstract base class for use case registry."""
    
    @abstractmethod
    def register_usecase(
        self, 
        usecase: UseCase, 
        functions: Dict[str, Callable]
    ) -> None:
        """
        Register functions for a use case.
        
        Args:
            usecase: The use case to register
            functions: Dictionary mapping function names to callable functions
        """
        pass
    
    @abstractmethod
    def get_functions(self, usecase: UseCase) -> Dict[str, Callable]:
        """
        Get functions for a use case.
        
        Args:
            usecase: The use case to get functions for
            
        Returns:
            Dictionary of function names to callable functions
        """
        pass
    
    @abstractmethod
    def get_system_prompt(
        self, 
        usecase: UseCase, 
        conversation: List[Dict[str, str]]
    ) -> str:
        """
        Get system prompt for a use case.
        
        Args:
            usecase: The use case context
            conversation: Conversation history
            
        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_generation_config(
        self,
        usecase: UseCase,
        agent_role: AgentRole,
        conversation: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> StructuredGenerationConfig:
        """
        Get the generation config for a use case and agent role.

        Args:
            usecase: The use case context
            agent_role: The backend agent role
            conversation: Conversation history
            context: Optional extra context for prompt generation

        Returns:
            Structured generation configuration including system prompt and schema
        """
        pass
    
    @abstractmethod
    def is_usecase_registered(self, usecase: UseCase) -> bool:
        """
        Check if a use case is registered.
        
        Args:
            usecase: The use case to check
            
        Returns:
            True if registered, False otherwise
        """
        pass
