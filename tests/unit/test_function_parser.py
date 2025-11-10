"""Unit tests for function parser."""

import pytest
from typing import Dict, Callable
from src.services.parser.function_parser import FunctionParser
from src.core.exceptions import FunctionExecutionException


class TestFunctionParser:
    """Test cases for FunctionParser."""
    
    def test_init_with_valid_registry(self, mock_function_registry):
        """Test initializing parser with valid registry."""
        parser = FunctionParser(mock_function_registry)
        assert parser.function_registry == mock_function_registry
    
    def test_init_with_invalid_registry_type(self):
        """Test initializing parser with invalid registry type."""
        with pytest.raises(TypeError) as exc_info:
            FunctionParser("not_a_dict")
        assert "must be a dictionary" in str(exc_info.value)
    
    def test_init_with_non_callable_in_registry(self):
        """Test initializing parser with non-callable in registry."""
        invalid_registry = {
            "valid_func": lambda x: x,
            "invalid_func": "not callable",
        }
        with pytest.raises(TypeError) as exc_info:
            FunctionParser(invalid_registry)
        assert "not callable" in str(exc_info.value)
    
    def test_parse_and_execute_simple_function(self, function_parser):
        """Test parsing and executing a simple function."""
        result = function_parser.parse_and_execute("simple_func()")
        assert result == "Simple result"
    
    def test_parse_and_execute_function_with_args(self, function_parser):
        """Test parsing and executing a function with arguments."""
        result = function_parser.parse_and_execute("test_func(x=10, y='hello')")
        assert "10" in result
        assert "hello" in result
    
    def test_parse_and_execute_unknown_function(self, function_parser):
        """Test parsing and executing an unknown function."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            function_parser.parse_and_execute("unknown_func()")
        assert "Unknown function" in str(exc_info.value)
        assert "unknown_func" in str(exc_info.value)
    
    def test_parse_and_execute_function_with_wrong_args(self, function_parser):
        """Test parsing and executing a function with wrong arguments."""
        # show_one requires an id, but we're not providing it correctly
        with pytest.raises(FunctionExecutionException):
            function_parser.parse_and_execute("show_one(wrong_param=5)")
    
    def test_parse_calls_empty_list(self, function_parser):
        """Test parsing an empty list of function calls."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            function_parser.parse_calls([])
        assert "No function calls provided" in str(exc_info.value)
    
    def test_parse_calls_single_function(self, function_parser):
        """Test parsing a single function call."""
        result = function_parser.parse_calls(["simple_func()"])
        assert result == "Simple result"
    
    def test_parse_calls_multiple_functions(self, function_parser):
        """Test parsing multiple function calls."""
        result = function_parser.parse_calls(["simple_func()", "count_all()"])
        assert "Simple result" in result
        assert "Total: 100" in result
        assert "\n" in result  # Should be newline-separated
    
    def test_parse_calls_with_non_string(self, function_parser):
        """Test parsing function calls with non-string input."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            function_parser.parse_calls([123])  # Not a string
        assert "must be a string" in str(exc_info.value)
    
    def test_parse_calls_handles_dict_result(self):
        """Test parsing calls that return dictionary with 'text' key."""
        def dict_func():
            return {"text": "Dict result"}
        
        registry = {"dict_func": dict_func}
        parser = FunctionParser(registry)
        
        result = parser.parse_calls(["dict_func()"])
        assert result == "Dict result"
    
    def test_parse_calls_handles_non_string_result(self):
        """Test parsing calls that return non-string results."""
        def int_func():
            return 42
        
        registry = {"int_func": int_func}
        parser = FunctionParser(registry)
        
        result = parser.parse_calls(["int_func()"])
        assert result == "42"  # Should be converted to string
    
    def test_parse_and_execute_function_exception_propagates(self):
        """Test that FunctionExecutionException from function is propagated."""
        def failing_func():
            raise FunctionExecutionException("Custom error")
        
        registry = {"failing_func": failing_func}
        parser = FunctionParser(registry)
        
        with pytest.raises(FunctionExecutionException) as exc_info:
            parser.parse_and_execute("failing_func()")
        assert "Custom error" in str(exc_info.value)
    
    def test_parse_and_execute_function_wraps_generic_exception(self):
        """Test that generic exceptions are wrapped in FunctionExecutionException."""
        def failing_func():
            raise ValueError("Some error")
        
        registry = {"failing_func": failing_func}
        parser = FunctionParser(registry)
        
        with pytest.raises(FunctionExecutionException) as exc_info:
            parser.parse_and_execute("failing_func()")
        assert "Error executing" in str(exc_info.value) or "Some error" in str(exc_info.value)
    
    def test_parse_and_execute_type_error_wrapped(self):
        """Test that TypeError is wrapped properly."""
        def func_with_args(x: int):
            return x
        
        registry = {"func_with_args": func_with_args}
        parser = FunctionParser(registry)
        
        # Call without required argument or with wrong type
        with pytest.raises(FunctionExecutionException) as exc_info:
            parser.parse_and_execute("func_with_args(wrong='string')")
        # Should wrap the TypeError
        assert "Error calling" in str(exc_info.value)

