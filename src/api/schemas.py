"""API schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional
from src.domain.entities.message import Message
from src.domain.entities.assistant_response import AssistantResponse


class ConversationMessage(BaseModel):
    """Message in conversation for API requests."""
    
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class AssistantRequest(BaseModel):
    """Request schema for assistant response endpoint."""
    
    conversation: List[ConversationMessage] = Field(..., description="Conversation history")
    model: str = Field(..., description="LLM model name")
    usecase: str = Field(..., description="Use case: 'Energy Consumption' or 'Heart Disease'")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")


class AssistantResponseWrapper(BaseModel):
    """Response wrapper matching legacy Flask API format."""
    
    assistantResponse: AssistantResponse = Field(..., description="Assistant response with function calls and freeform response")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")

