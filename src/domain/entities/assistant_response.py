"""Assistant response entity for LLM responses."""

from pydantic import BaseModel, Field
from typing import List, Optional
from src.domain.entities.function_call import FunctionCall


class AssistantResponse(BaseModel):
    """Represents a response from the assistant."""
    
    function_calls: List[str] = Field(default_factory=list, description="List of function call strings")
    freeform_response: str = Field(..., description="Free-form text response from the assistant")
    parse: Optional[str] = Field(None, description="Parsed results from function execution")
    
    def has_function_calls(self) -> bool:
        """Check if the response contains function calls."""
        return len(self.function_calls) > 0
    
    def get_function_call_objects(self) -> List[FunctionCall]:
        """Convert function call strings to FunctionCall objects."""
        # This would be implemented by parsing the function call strings
        # For now, return empty list - will be implemented in the parser
        return []
