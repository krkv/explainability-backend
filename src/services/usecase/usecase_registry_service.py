"""Use case registry service for managing use cases, functions, and system prompts."""

from typing import Dict, Callable, Any, List
from src.core.constants import UseCase
from src.core.exceptions import FunctionExecutionException
from src.core.logging_config import get_logger
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.domain.entities.message import Message

logger = get_logger(__name__)


class UseCaseRegistryService(UseCaseRegistry):
    """
    Manages the registration and retrieval of functions and system prompts
    for different use cases.
    """
    
    def __init__(self):
        self._registries: Dict[UseCase, Dict[str, Callable]] = {}
        self._system_prompts: Dict[UseCase, str] = {}
        self._initialize_default_prompts()
        logger.info("UseCaseRegistryService initialized")
    
    def register_usecase(self, usecase: UseCase, functions: Dict[str, Callable]) -> None:
        """
        Register functions for a use case.
        
        Args:
            usecase: The use case to register
            functions: Dictionary mapping function names to callable functions
        """
        if not isinstance(usecase, UseCase):
            raise ValueError("Usecase must be an instance of UseCase enum.")
        if not isinstance(functions, dict):
            raise ValueError("Functions must be a dictionary.")
        
        self._registries[usecase] = functions
        logger.info(f"Registered {len(functions)} functions for usecase '{usecase.value}'")
    
    def register_usecase_functions(self, usecase: UseCase, functions: Dict[str, Callable]):
        """
        Registers a dictionary of functions for a specific use case.
        (Alias for register_usecase for backward compatibility)
        
        Args:
            usecase: The name of the use case (e.g., UseCase.ENERGY).
            functions: A dictionary where keys are function names (str) and values are callable functions.
        """
        self.register_usecase(usecase, functions)
    
    def get_function(self, usecase: UseCase, function_name: str) -> Callable:
        """
        Retrieves a specific function for a given use case.
        
        Args:
            usecase: The name of the use case.
            function_name: The name of the function to retrieve.
            
        Returns:
            The callable function.
            
        Raises:
            FunctionExecutionException: If the use case or function is not found.
        """
        if usecase not in self._registries:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' not registered.")
        
        if function_name not in self._registries[usecase]:
            raise FunctionExecutionException(f"Function '{function_name}' not found in usecase '{usecase.value}'.")
        
        return self._registries[usecase][function_name]
    
    def get_functions(self, usecase: UseCase) -> Dict[str, Callable]:
        """
        Retrieves all registered functions for a specific use case.
        
        Args:
            usecase: The name of the use case.
            
        Returns:
            A dictionary of callable functions.
            
        Raises:
            FunctionExecutionException: If the use case is not found.
        """
        if usecase not in self._registries:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' not registered.")
        return self._registries[usecase]
    
    def get_system_prompt(self, usecase: UseCase, conversation: List[Dict[str, str]]) -> str:
        """
        Retrieves the system prompt for a given use case.
        Note: This method is kept for interface compatibility, but the actual
        system prompts are handled by the original prompt functions in the LLM providers.
        
        Args:
            usecase: The use case for which to retrieve the system prompt.
            conversation: The list of past messages in the conversation.
            
        Returns:
            A placeholder system prompt (actual prompts handled by LLM providers).
        """
        # The actual system prompts are handled by the original functions
        # in instances/energy/prompt.py and instances/heart/prompt.py
        # This is just for interface compatibility
        return f"System prompt for {usecase.value} use case (handled by LLM providers)"
    
    def is_usecase_registered(self, usecase: UseCase) -> bool:
        """Check if a use case is registered."""
        return usecase in self._registries
    
    def get_registered_usecases(self) -> List[UseCase]:
        """Get a list of all registered use cases."""
        return list(self._registries.keys())
    
    def clear_all_usecases(self) -> None:
        """Clear all registered use cases."""
        self._registries.clear()
        logger.info("Cleared all use cases")
    
    def _initialize_default_prompts(self) -> None:
        """
        Initialize default system prompts for each use case.
        Note: The actual system prompts are handled by the original functions
        in instances/energy/prompt.py and instances/heart/prompt.py
        """
        # These are placeholder prompts - the real ones are in the original files
        self._system_prompts[UseCase.ENERGY] = "Energy analysis assistant (original prompt in instances/energy/prompt.py)"
        self._system_prompts[UseCase.HEART] = "Heart disease analysis assistant (original prompt in instances/heart/prompt.py)"
    
    def _get_default_prompt(self) -> str:
        """
        Get default system prompt for unknown use cases.
        """
        return "You are a helpful AI assistant."
