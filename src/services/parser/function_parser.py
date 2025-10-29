"""Safe function parser that executes function calls without using eval()."""

from typing import Callable, Dict, Any, List
from src.services.parser.ast_parser import ASTParser
from src.services.parser.function_registry import FunctionRegistry
from src.core.exceptions import FunctionExecutionException


class FunctionParser:
    """Safely parses and executes function calls without eval()."""
    
    def __init__(self, function_registry: Dict[str, Callable]):
        """
        Initialize parser with a function registry.
        
        Args:
            function_registry: Dictionary mapping function names to callable functions
        """
        if not isinstance(function_registry, dict):
            raise TypeError("function_registry must be a dictionary")
        
        # Validate all registry entries are callable
        for name, func in function_registry.items():
            if not callable(func):
                raise TypeError(f"Function '{name}' in registry is not callable")
        
        self.function_registry = function_registry
    
    def parse_and_execute(self, function_call_str: str) -> Any:
        """
        Parse a function call string and execute it safely.
        
        Args:
            function_call_str: String like "count_all()" or "show_one(id=5)"
            
        Returns:
            Result of function execution
            
        Raises:
            FunctionExecutionException: If function doesn't exist or execution fails
        """
        # Parse the function call using AST
        func_name, kwargs = ASTParser.parse_function_call(function_call_str)
        
        # Check if function exists in registry
        if func_name not in self.function_registry:
            available = ", ".join(sorted(self.function_registry.keys()))
            raise FunctionExecutionException(
                f"Unknown function '{func_name}'. Available functions: {available}"
            )
        
        # Get the function from registry
        func = self.function_registry[func_name]
        
        # Execute the function with the parsed arguments
        try:
            result = func(**kwargs)
            return result
        except TypeError as e:
            # Handle argument errors (missing required args, too many args, etc.)
            raise FunctionExecutionException(
                f"Error calling '{func_name}': {e}"
            )
        except Exception as e:
            # Re-raise FunctionExecutionException as-is
            if isinstance(e, FunctionExecutionException):
                raise
            # Wrap other exceptions
            raise FunctionExecutionException(
                f"Error executing '{func_name}': {e}"
            )
    
    def parse_calls(self, function_calls: List[str]) -> str:
        """
        Parse multiple function calls and return concatenated results.
        
        Args:
            function_calls: List of function call strings
            
        Returns:
            Newline-separated string of results
            
        Raises:
            FunctionExecutionException: If any function call fails
        """
        if not function_calls:
            raise FunctionExecutionException("No function calls provided")
        
        results = []
        for call in function_calls:
            if not isinstance(call, str):
                raise FunctionExecutionException(
                    f"Function call must be a string, got {type(call).__name__}"
                )
            
            result = self.parse_and_execute(call)
            
            # Handle different return types
            # Energy functions return strings, heart functions return dicts with "text" key
            if isinstance(result, dict) and "text" in result:
                results.append(result["text"])
            elif isinstance(result, str):
                results.append(result)
            else:
                # Convert to string as fallback
                results.append(str(result))
        
        return '\n'.join(results)

