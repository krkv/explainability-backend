"""Use case registry service for managing use cases, functions, and system prompts."""

from typing import Dict, Callable, Any, List, Optional
from src.core.constants import UseCase
from src.core.exceptions import FunctionExecutionException
from src.core.logging_config import get_logger
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.domain.entities.message import Message
from src.usecases.base.base_usecase import BaseUseCase

logger = get_logger(__name__)


class UseCaseRegistryService(UseCaseRegistry):
    """
    Manages the registration and retrieval of functions and system prompts
    for different use cases.
    """
    
    def __init__(self):
        self._registries: Dict[UseCase, Dict[str, Callable]] = {}
        self._system_prompts: Dict[UseCase, str] = {}
        self._usecase_instances: Dict[UseCase, BaseUseCase] = {}
        self._initialize_usecases()
        logger.info("UseCaseRegistryService initialized")
    
    def _initialize_usecases(self) -> None:
        """Initialize use case instances and register their functions."""
        from src.infrastructure.factory import get_model_loader, get_data_loader, get_explainer_loader
        from src.usecases.energy.energy_usecase import EnergyUseCase
        from src.usecases.heart.heart_usecase import HeartUseCase
        
        model_loader = get_model_loader()
        data_loader = get_data_loader()
        explainer_loader = get_explainer_loader()
        
        # Initialize energy use case
        energy_usecase = EnergyUseCase(
            model_loader=model_loader,
            data_loader=data_loader,
            explainer_loader=explainer_loader,
        )
        self._usecase_instances[UseCase.ENERGY] = energy_usecase
        self._registries[UseCase.ENERGY] = energy_usecase.get_functions()
        logger.info(f"Registered energy use case with {len(self._registries[UseCase.ENERGY])} functions")
        
        # Initialize heart use case
        heart_usecase = HeartUseCase(
            model_loader=model_loader,
            data_loader=data_loader,
            explainer_loader=explainer_loader,
        )
        self._usecase_instances[UseCase.HEART] = heart_usecase
        self._registries[UseCase.HEART] = heart_usecase.get_functions()
        logger.info(f"Registered heart use case with {len(self._registries[UseCase.HEART])} functions")
    
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
        Retrieves the system prompt for a given use case with embedded data and functions.
        
        Args:
            usecase: The use case for which to retrieve the system prompt.
            conversation: The list of past messages in the conversation.
            
        Returns:
            The complete system prompt with embedded data and functions.
        """
        if usecase in self._usecase_instances:
            return self._usecase_instances[usecase].get_system_prompt(conversation)
        else:
            return self._get_default_prompt()
    
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
    
    def get_usecase_instance(self, usecase: UseCase) -> Optional[BaseUseCase]:
        """
        Get the use case instance for a given use case.
        
        Args:
            usecase: The use case to get
            
        Returns:
            BaseUseCase instance or None if not found
        """
        return self._usecase_instances.get(usecase)
    
    def _get_default_prompt(self) -> str:
        """
        Get default system prompt for unknown use cases.
        """
        return "You are a helpful AI assistant."
