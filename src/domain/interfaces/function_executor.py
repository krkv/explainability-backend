"""Function execution interface for safe function calling."""

from abc import ABC, abstractmethod
from typing import List
from src.core.constants import UseCase


class FunctionExecutor(ABC):
    """Abstract base class for function execution."""
    
    @abstractmethod
    def execute_calls(
        self, 
        function_calls: List[str], 
        usecase: UseCase
    ) -> str:
        """
        Execute a list of function calls safely.
        
        Args:
            function_calls: List of function call strings
            usecase: The use case context (energy or heart)
            
        Returns:
            Concatenated results from function execution
            
        Raises:
            FunctionExecutionException: If function execution fails
        """
        pass
    
    @abstractmethod
    def get_available_functions(self, usecase: UseCase) -> List[str]:
        """
        Get list of available function names for a usecase.
        
        Args:
            usecase: The use case context
            
        Returns:
            List of function names
        """
        pass
    
    @abstractmethod
    def validate_function_call(self, function_call: str, usecase: UseCase) -> bool:
        """
        Validate if a function call is valid for the given usecase.
        
        Args:
            function_call: Function call string to validate
            usecase: The use case context
            
        Returns:
            True if valid, False otherwise
        """
        pass
