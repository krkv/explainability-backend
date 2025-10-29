"""AST-based parser for safely parsing function call strings."""

import ast
from typing import Dict, Any, Tuple
from src.core.exceptions import FunctionExecutionException


class ASTParser:
    """AST-based function call parser for safe parsing without eval()."""
    
    @staticmethod
    def parse_function_call(function_call_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a function call string and extract function name and arguments.
        
        Args:
            function_call_str: String like "count_all()" or "show_one(id=5)"
            
        Returns:
            Tuple of (function_name, keyword_arguments_dict)
            
        Raises:
            FunctionExecutionException: If parsing fails or invalid format
        """
        try:
            # Parse the string into an AST
            # We use 'eval' mode to parse expressions, but we won't execute it
            parsed = ast.parse(function_call_str, mode='eval')
            
            # Validate it's a function call
            if not isinstance(parsed.body, ast.Call):
                raise FunctionExecutionException(
                    f"Invalid function call format: expected function call, got {type(parsed.body).__name__}"
                )
            
            call_node = parsed.body
            
            # Extract function name
            if not isinstance(call_node.func, ast.Name):
                raise FunctionExecutionException(
                    f"Invalid function name: only simple function names are allowed"
                )
            
            func_name = call_node.func.id
            
            # Extract keyword arguments
            kwargs = {}
            for keyword in call_node.keywords:
                if keyword.arg is None:
                    raise FunctionExecutionException(
                        "Positional arguments are not allowed, only keyword arguments"
                    )
                
                # Safely evaluate the value using ast.literal_eval
                # This only works for literals (numbers, strings, lists, dicts, etc.)
                # and prevents arbitrary code execution
                try:
                    value = ast.literal_eval(keyword.value)
                    kwargs[keyword.arg] = value
                except ValueError as e:
                    raise FunctionExecutionException(
                        f"Invalid argument value for '{keyword.arg}': {e}"
                    )
            
            # Check for positional arguments (not allowed for security)
            if call_node.args:
                raise FunctionExecutionException(
                    "Positional arguments are not allowed, only keyword arguments"
                )
            
            return func_name, kwargs
            
        except SyntaxError as e:
            raise FunctionExecutionException(
                f"Syntax error in function call '{function_call_str}': {e}"
            )
        except Exception as e:
            if isinstance(e, FunctionExecutionException):
                raise
            raise FunctionExecutionException(
                f"Error parsing function call '{function_call_str}': {e}"
            )

