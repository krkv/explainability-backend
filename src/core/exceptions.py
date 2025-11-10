"""Custom exception hierarchy for the application."""


class ExplainabilityException(Exception):
    """Base exception for all application errors."""
    pass


class InvalidModelException(ExplainabilityException):
    """Raised when invalid model is specified."""
    pass


class InvalidUseCaseException(ExplainabilityException):
    """Raised when invalid usecase is specified."""
    pass


class FunctionExecutionException(ExplainabilityException):
    """Raised when function execution fails."""
    pass


class LLMProviderException(ExplainabilityException):
    """Raised when LLM provider fails."""
    pass


class ModelLoadException(ExplainabilityException):
    """Raised when model fails to load."""
    pass


class DataLoadException(ExplainabilityException):
    """Raised when dataset fails to load."""
    pass

