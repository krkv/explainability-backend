"""API schemas for request/response validation."""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from src.domain.entities.assistant_response import AssistantResponse


class ConversationMessage(BaseModel):
    """Message in conversation for API requests."""

    model_config = ConfigDict(populate_by_name=True)
    
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    is_function_call: bool = Field(
        default=False,
        alias="isFunctionCall",
        description="Whether the assistant message represents function calls.",
    )


class AssistantRequest(BaseModel):
    """Request schema for assistant response endpoint."""
    
    conversation: List[ConversationMessage] = Field(..., description="Full conversation history from frontend")
    model: str = Field(..., description="LLM model name")
    usecase: str = Field(..., description="Use case: 'Energy Consumption' or 'Heart Disease'")


class SuggestedFollowUpsRequest(BaseModel):
    """Request schema for suggested follow-ups endpoint."""

    conversation: List[ConversationMessage] = Field(..., description="Full conversation history from frontend")
    usecase: str = Field(..., description="Use case: 'Energy Consumption' or 'Heart Disease'")
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Maximum number of suggestions to return",
    )
    exclude_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions that should be excluded from the response",
    )


class AssistantResponseWrapper(BaseModel):
    """Response wrapper matching legacy Flask API format."""
    
    assistantResponse: AssistantResponse = Field(..., description="Assistant response with function calls and freeform response")


class SuggestedFollowUpsResponse(BaseModel):
    """Response payload for suggested follow-up prompts."""

    suggested_follow_ups: Optional[List[str]] = Field(
        default=None,
        description="Suggested next prompts for the user, if available",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
