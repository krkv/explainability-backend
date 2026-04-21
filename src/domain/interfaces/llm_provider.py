"""LLM provider interface for different language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

from src.core.constants import UseCase


class AgentRole(str, Enum):
    """Supported LLM agent roles."""

    ASSISTANT = "assistant"
    SUGGESTER = "suggester"


@dataclass(frozen=True)
class StructuredGenerationConfig:
    """Role-aware prompt and schema configuration for one LLM call."""

    system_prompt: str
    response_schema: Dict[str, Any]


def build_response_schema(agent_role: AgentRole) -> Dict[str, Any]:
    """Return a Gemini-compatible structured response schema for an agent role."""
    if agent_role == AgentRole.SUGGESTER:
        return {
            "title": "SuggestedFollowUps",
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 3,
                    "maxItems": 5,
                    "description": "Three to five unique next prompts the user could ask.",
                }
            },
            "required": ["suggestions"],
        }

    return {
        "title": "AssistantResponse",
        "type": "object",
        "properties": {
            "function_calls": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "A list of function calls in string format."
            },
            "freeform_response": {
                "type": "string",
                "description": "A free-form response that strictly follows the rules of the assistant."
            }
        },
        "required": ["function_calls", "freeform_response"],
    }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase,
        agent_role: AgentRole = AgentRole.ASSISTANT,
        generation_config: Optional[StructuredGenerationConfig] = None,
        generation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys
            usecase: The use case context (energy or heart)
            agent_role: Role-specific prompt/schema behavior for the request
            generation_config: Optional explicit prompt/schema override
            generation_context: Optional additional context used to build a prompt
            
        Returns:
            Raw response string from the LLM
            
        Raises:
            LLMProviderException: If the LLM call fails
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name string
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and configured.
        
        Returns:
            True if available, False otherwise
        """
        pass
