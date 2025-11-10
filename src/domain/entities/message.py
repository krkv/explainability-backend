"""Message entity for conversation handling."""

from pydantic import BaseModel, Field
from typing import Literal
from src.core.constants import MessageRole


class Message(BaseModel):
    """Represents a message in a conversation."""
    
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class Conversation(BaseModel):
    """Represents a conversation as a list of messages."""
    
    id: str = Field(..., description="Unique identifier for the conversation")
    messages: list[Message] = Field(default_factory=list, description="List of messages in the conversation")
    
    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation."""
        message = Message(role=role, content=content)
        self.messages.append(message)
    
    def get_last_message(self) -> Message | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None
    
    def get_user_messages(self) -> list[Message]:
        """Get all user messages in the conversation."""
        return [msg for msg in self.messages if msg.role == MessageRole.USER]
    
    def get_assistant_messages(self) -> list[Message]:
        """Get all assistant messages in the conversation."""
        return [msg for msg in self.messages if msg.role == MessageRole.ASSISTANT]
