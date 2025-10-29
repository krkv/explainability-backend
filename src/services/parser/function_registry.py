"""Function registry for managing available functions per usecase."""

from typing import Dict, Callable, Optional


class FunctionRegistry:
    """Registry of available functions per usecase."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._registries: Dict[str, Dict[str, Callable]] = {}
    
    def register_usecase(
        self, 
        usecase: str, 
        functions: Dict[str, Callable]
    ) -> None:
        """
        Register functions for a usecase.
        
        Args:
            usecase: The usecase identifier (e.g., "energy", "heart")
            functions: Dictionary mapping function names to callable functions
        """
        if not isinstance(functions, dict):
            raise TypeError("functions must be a dictionary")
        
        # Validate all values are callable
        for name, func in functions.items():
            if not callable(func):
                raise TypeError(f"Function '{name}' is not callable")
        
        self._registries[usecase] = functions
    
    def get_registry(self, usecase: str) -> Dict[str, Callable]:
        """
        Get function registry for a usecase.
        
        Args:
            usecase: The usecase identifier
            
        Returns:
            Dictionary of function name to callable
            
        Raises:
            ValueError: If usecase is not registered
        """
        if usecase not in self._registries:
            raise ValueError(f"Unknown usecase: {usecase}")
        return self._registries[usecase]
    
    def is_registered(self, usecase: str) -> bool:
        """Check if a usecase is registered."""
        return usecase in self._registries
    
    def get_function(self, usecase: str, function_name: str) -> Optional[Callable]:
        """
        Get a specific function by name.
        
        Args:
            usecase: The usecase identifier
            function_name: Name of the function
            
        Returns:
            The callable function or None if not found
        """
        registry = self.get_registry(usecase)
        return registry.get(function_name)

