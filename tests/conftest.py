"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Callable, List
from src.services.parser.function_parser import FunctionParser
from src.services.parser.ast_parser import ASTParser
from src.core.constants import UseCase, Model
from src.core.exceptions import FunctionExecutionException


@pytest.fixture
def mock_function():
    """Create a simple mock function for testing."""
    def test_func(x: int = 1, y: str = "test") -> str:
        return f"Result: {x}, {y}"
    return test_func


@pytest.fixture
def mock_function_registry(mock_function):
    """Create a mock function registry."""
    return {
        "test_func": mock_function,
        "count_all": lambda: "Total: 100",
        "show_one": lambda id: f"Item {id}",
        "simple_func": lambda: "Simple result",
    }


@pytest.fixture
def function_parser(mock_function_registry):
    """Create a FunctionParser instance with mock registry."""
    return FunctionParser(mock_function_registry)


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.generate_response = AsyncMock(
        return_value='{"function_calls": ["count_all()"], "freeform_response": "Test response"}'
    )
    return provider


@pytest.fixture
def mock_function_executor():
    """Create a mock function executor."""
    executor = Mock()
    executor.execute_calls = Mock(return_value="Function result")
    executor.get_stats = Mock(return_value={"executed_calls": 10})
    return executor


@pytest.fixture
def mock_usecase_registry():
    """Create a mock use case registry."""
    registry = Mock()
    registry.get_system_prompt = Mock(return_value="Test system prompt")
    registry.get_functions = Mock(return_value={"test_func": lambda: "test"})
    registry.get_usecase = Mock(return_value=UseCase.ENERGY)
    return registry


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return [
        {"role": "user", "content": "What is the dataset size?"},
        {"role": "assistant", "content": "The dataset has 100 records."},
        {"role": "user", "content": "Show me record 5"},
    ]


@pytest.fixture
def sample_llm_response():
    """Create a sample LLM JSON response."""
    return '{"function_calls": ["count_all()", "show_one(id=5)"], "freeform_response": "Here are the results"}'

