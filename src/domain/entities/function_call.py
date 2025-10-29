"""Function call entity for LLM function calling."""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class FunctionCall(BaseModel):
    """Represents a function call with its parameters."""
    
    name: str = Field(..., description="Name of the function to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the function call")
    
    def to_string(self) -> str:
        """Convert function call to string format for LLM."""
        if not self.parameters:
            return f"{self.name}()"
        
        param_strs = []
        for key, value in self.parameters.items():
            if isinstance(value, str):
                param_strs.append(f"{key}='{value}'")
            else:
                param_strs.append(f"{key}={value}")
        
        return f"{self.name}({', '.join(param_strs)})"


class FunctionResult(BaseModel):
    """Represents the result of a function call."""
    
    success: bool = Field(..., description="Whether the function call was successful")
    result: Optional[str] = Field(None, description="Result of the function call")
    error: Optional[str] = Field(None, description="Error message if the call failed")
    
    @classmethod
    def success_result(cls, result: str) -> "FunctionResult":
        """Create a successful function result."""
        return cls(success=True, result=result)
    
    @classmethod
    def error_result(cls, error: str) -> "FunctionResult":
        """Create an error function result."""
        return cls(success=False, error=error)
