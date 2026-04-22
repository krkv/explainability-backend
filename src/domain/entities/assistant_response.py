"""Assistant response entity for LLM responses."""

from typing import List, Optional

from pydantic import BaseModel, Field

from src.domain.entities.function_call import FunctionCall


class AssistantResponse(BaseModel):
    """Represents a response from the assistant."""
    
    function_calls: List[str] = Field(default_factory=list, description="List of function call strings")
    freeform_response: str = Field(..., description="Free-form text response from the assistant")
    parse: Optional[str] = Field(None, description="Parsed results from function execution")
    trace_id: Optional[str] = Field(
        default=None,
        description="Langfuse trace ID associated with this assistant turn",
    )
    suggested_follow_ups: Optional[List[str]] = Field(
        default=None,
        description="Optional next prompts suggested for the user",
    )
    
    def has_function_calls(self) -> bool:
        """Check if the response contains function calls."""
        return len(self.function_calls) > 0
    
    def get_function_call_objects(self) -> List[FunctionCall]:
        """Convert function call strings to FunctionCall objects."""
        # This would be implemented by parsing the function call strings
        # For now, return empty list - will be implemented in the parser
        return []
