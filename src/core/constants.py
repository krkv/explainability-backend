"""Constants and enums for the application."""

from enum import Enum


class Model(str, Enum):
    """Supported LLM models."""
    LLAMA_3_3_70B = "Llama-3.3-70B-Instruct"
    GEMINI_2_0_FLASH = "Gemini-2.0-Flash"


class UseCase(str, Enum):
    """Supported use cases."""
    ENERGY = "energy"
    HEART = "heart"
    
    @classmethod
    def from_string(cls, value: str) -> "UseCase":
        """
        Map frontend string format to enum.
        
        Handles both frontend format ("Energy Consumption", "Heart Disease")
        and backend format ("energy", "heart").
        
        Args:
            value: String value from frontend or backend
            
        Returns:
            UseCase enum value
            
        Raises:
            ValueError: If value cannot be mapped to a valid UseCase
        """
        mapping = {
            "Energy Consumption": cls.ENERGY,
            "Heart Disease": cls.HEART,
            "energy": cls.ENERGY,
            "heart": cls.HEART,
        }
        result = mapping.get(value)
        if result is None:
            # Try case-insensitive match
            value_lower = value.lower()
            if value_lower in ["energy consumption", "energy"]:
                return cls.ENERGY
            elif value_lower in ["heart disease", "heart"]:
                return cls.HEART
            raise ValueError(f"Invalid usecase value: {value}. Expected 'Energy Consumption', 'Heart Disease', 'energy', or 'heart'")
        return result


class MessageRole(str, Enum):
    """Message roles in conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# API Endpoints
class APIEndpoints:
    """API endpoint constants."""
    READY = "/ready"
    ASSISTANT_RESPONSE = "/getAssistantResponse"


# Error Messages
class ErrorMessages:
    """Common error messages."""
    INVALID_MODEL = "Invalid model specified"
    INVALID_USECASE = "Invalid usecase specified"
    FUNCTION_EXECUTION_FAILED = "Function execution failed"
    LLM_PROVIDER_ERROR = "LLM provider error"
    MODEL_LOAD_FAILED = "Model failed to load"
    DATA_LOAD_FAILED = "Dataset failed to load"
    NO_FUNCTION_CALLS = "No function calls provided"
    UNKNOWN_FUNCTION = "Unknown function"
    INVALID_SYNTAX = "Invalid syntax in function call"


# File Extensions
class FileExtensions:
    """Common file extensions."""
    PICKLE = ".pkl"
    CSV = ".csv"
    JSON = ".json"


# Default Values
class Defaults:
    """Default configuration values."""
    CACHE_SIZE = 10
    LOG_LEVEL = "INFO"
    API_HOST = "0.0.0.0"
    API_PORT = 8080
