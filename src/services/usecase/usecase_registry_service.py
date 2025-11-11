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
    
    Use cases are loaded lazily - only when first requested, not at initialization.
    This prevents loading expensive resources (models, datasets, explainers) for
    use cases that may never be used.
    """
    
    def __init__(self):
        self._registries: Dict[UseCase, Dict[str, Callable]] = {}
        self._system_prompts: Dict[UseCase, str] = {}
        self._usecase_instances: Dict[UseCase, BaseUseCase] = {}
        # Track which use cases are available (but not yet loaded)
        self._available_usecases = {UseCase.ENERGY, UseCase.HEART}
        logger.info("UseCaseRegistryService initialized (lazy loading enabled)")
    
    def _get_or_create_usecase(self, usecase: UseCase) -> BaseUseCase:
        """
        Get or create a use case instance (lazy loading).
        
        Args:
            usecase: The use case to get or create
            
        Returns:
            BaseUseCase instance
            
        Raises:
            FunctionExecutionException: If use case is not available
        """
        # Check if already loaded
        if usecase in self._usecase_instances:
            return self._usecase_instances[usecase]
        
        # Check if use case is available
        if usecase not in self._available_usecases:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' is not available.")
        
        # Lazy load the use case
        logger.info(f"Lazy loading use case: {usecase.value}")
        from src.infrastructure.factory import get_model_loader, get_data_loader, get_explainer_loader
        
        model_loader = get_model_loader()
        data_loader = get_data_loader()
        explainer_loader = get_explainer_loader()
        
        if usecase == UseCase.ENERGY:
            from src.usecases.energy.energy_usecase import EnergyUseCase
            usecase_instance = EnergyUseCase(
                model_loader=model_loader,
                data_loader=data_loader,
                explainer_loader=explainer_loader,
            )
        elif usecase == UseCase.HEART:
            from src.usecases.heart.heart_usecase import HeartUseCase
            usecase_instance = HeartUseCase(
                model_loader=model_loader,
                data_loader=data_loader,
                explainer_loader=explainer_loader,
            )
        else:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' is not implemented.")
        
        # Store instance and register functions
        self._usecase_instances[usecase] = usecase_instance
        self._registries[usecase] = usecase_instance.get_functions()
        logger.info(f"Loaded and registered {usecase.value} use case with {len(self._registries[usecase])} functions")
        
        return usecase_instance
    
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
        # Lazy load use case if not already loaded
        self._get_or_create_usecase(usecase)
        
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
        # Lazy load use case if not already loaded
        self._get_or_create_usecase(usecase)
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
        # Lazy load use case if not already loaded
        usecase_instance = self._get_or_create_usecase(usecase)
        return usecase_instance.get_system_prompt(conversation)
    
    def is_usecase_registered(self, usecase: UseCase) -> bool:
        """
        Check if a use case is registered (either loaded or available for lazy loading).
        
        Args:
            usecase: The use case to check
            
        Returns:
            True if registered or available, False otherwise
        """
        return usecase in self._registries or usecase in self._available_usecases
    
    def get_registered_usecases(self) -> List[UseCase]:
        """
        Get a list of all available use cases (both loaded and available for lazy loading).
        
        Returns:
            List of available use cases
        """
        return list(self._available_usecases)
    
    def clear_all_usecases(self) -> None:
        """Clear all registered use cases and instances."""
        self._registries.clear()
        self._usecase_instances.clear()
        logger.info("Cleared all use cases")
    
    def get_usecase_instance(self, usecase: UseCase) -> Optional[BaseUseCase]:
        """
        Get the use case instance for a given use case (lazy loads if needed).
        
        Args:
            usecase: The use case to get
            
        Returns:
            BaseUseCase instance or None if not available
        """
        try:
            return self._get_or_create_usecase(usecase)
        except FunctionExecutionException:
            return None
    
    def _get_default_prompt(self) -> str:
        """
        Get default system prompt for unknown use cases.
        """
        return "You are a helpful AI assistant."
