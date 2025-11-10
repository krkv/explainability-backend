"""Unit tests for AST parser."""

import pytest
from src.services.parser.ast_parser import ASTParser
from src.core.exceptions import FunctionExecutionException


class TestASTParser:
    """Test cases for ASTParser."""
    
    def test_parse_simple_function_call_no_args(self):
        """Test parsing a simple function call with no arguments."""
        func_name, kwargs = ASTParser.parse_function_call("count_all()")
        assert func_name == "count_all"
        assert kwargs == {}
    
    def test_parse_function_call_with_keyword_args(self):
        """Test parsing a function call with keyword arguments."""
        func_name, kwargs = ASTParser.parse_function_call("show_one(id=5)")
        assert func_name == "show_one"
        assert kwargs == {"id": 5}
    
    def test_parse_function_call_multiple_args(self):
        """Test parsing a function call with multiple keyword arguments."""
        func_name, kwargs = ASTParser.parse_function_call("predict_new(age=25, energy=100)")
        assert func_name == "predict_new"
        assert kwargs == {"age": 25, "energy": 100}
    
    def test_parse_function_call_with_string_arg(self):
        """Test parsing a function call with string arguments."""
        func_name, kwargs = ASTParser.parse_function_call('show_group(type="summer")')
        assert func_name == "show_group"
        assert kwargs == {"type": "summer"}
    
    def test_parse_function_call_with_list_arg(self):
        """Test parsing a function call with list arguments."""
        func_name, kwargs = ASTParser.parse_function_call("predict_group(ids=[1, 2, 3])")
        assert func_name == "predict_group"
        assert kwargs == {"ids": [1, 2, 3]}
    
    def test_parse_function_call_with_dict_arg(self):
        """Test parsing a function call with dictionary arguments."""
        func_name, kwargs = ASTParser.parse_function_call('predict_new(data={"age": 25})')
        assert func_name == "predict_new"
        assert kwargs == {"data": {"age": 25}}
    
    def test_parse_function_call_invalid_syntax(self):
        """Test parsing an invalid function call syntax."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            ASTParser.parse_function_call("count_all(")
        assert "Syntax error" in str(exc_info.value) or "Error parsing" in str(exc_info.value)
    
    def test_parse_function_call_not_a_call(self):
        """Test parsing something that's not a function call."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            ASTParser.parse_function_call("x = 5")
        # Should raise either syntax error or invalid format error
        error_msg = str(exc_info.value)
        assert "Invalid function call format" in error_msg or "Syntax error" in error_msg or "Error parsing" in error_msg
    
    def test_parse_function_call_with_positional_args_not_allowed(self):
        """Test that positional arguments are not allowed."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            ASTParser.parse_function_call("show_one(5)")
        assert "Positional arguments are not allowed" in str(exc_info.value)
    
    def test_parse_function_call_complex_attribute_not_allowed(self):
        """Test that complex attribute access is not allowed."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            ASTParser.parse_function_call("obj.method()")
        assert "Invalid function name" in str(exc_info.value)
    
    def test_parse_function_call_invalid_literal(self):
        """Test parsing with invalid literal value."""
        with pytest.raises(FunctionExecutionException) as exc_info:
            ASTParser.parse_function_call("test_func(x=invalid_var)")
        assert "Invalid argument value" in str(exc_info.value) or "Error parsing" in str(exc_info.value)
    
    def test_parse_function_call_float_arg(self):
        """Test parsing a function call with float arguments."""
        func_name, kwargs = ASTParser.parse_function_call("predict_new(price=99.99)")
        assert func_name == "predict_new"
        assert kwargs == {"price": 99.99}
    
    def test_parse_function_call_bool_arg(self):
        """Test parsing a function call with boolean arguments."""
        func_name, kwargs = ASTParser.parse_function_call("filter_data(active=True)")
        assert func_name == "filter_data"
        assert kwargs == {"active": True}
    
    def test_parse_function_call_none_arg(self):
        """Test parsing a function call with None argument."""
        func_name, kwargs = ASTParser.parse_function_call("process_data(value=None)")
        assert func_name == "process_data"
        assert kwargs == {"value": None}
    
    def test_parse_function_call_empty_string_arg(self):
        """Test parsing a function call with empty string argument."""
        func_name, kwargs = ASTParser.parse_function_call('test_func(name="")')
        assert func_name == "test_func"
        assert kwargs == {"name": ""}
    
    def test_parse_function_call_nested_structure(self):
        """Test parsing a function call with nested data structures."""
        func_name, kwargs = ASTParser.parse_function_call(
            'complex_func(data={"items": [1, 2, 3], "meta": {"count": 3}})'
        )
        assert func_name == "complex_func"
        assert "data" in kwargs
        assert kwargs["data"]["items"] == [1, 2, 3]

