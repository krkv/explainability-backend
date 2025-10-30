"""FastAPI routes for API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import (
    AssistantRequest,
    AssistantResponseWrapper,
    AssistantResponse,
    HealthResponse,
)
from src.api.dependencies import (
    validate_model,
    validate_usecase,
    AssistantServiceDep,
)
from src.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/ready", response_model=HealthResponse, tags=["Health"])
async def ready():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    return HealthResponse(status="OK")


@router.post("/getAssistantResponse", response_model=AssistantResponseWrapper, tags=["Assistant"])
async def get_assistant_response(
    request: AssistantRequest,
    assistant_service: AssistantServiceDep,
):
    """
    Generate assistant response with function execution.
    
    This endpoint processes a user message, generates an LLM response,
    executes any function calls, and returns the complete response.
    
    Args:
        request: Assistant request with conversation, model, and usecase
        assistant_service: Injected assistant service instance
        
    Returns:
        Assistant response wrapper containing function calls, freeform response, and parse results
        
    Raises:
        HTTPException: If request validation fails or processing fails
    """
    try:
        logger.info(f"Processing request - Model: {request.model}, UseCase: {request.usecase}")
        
        # Validate model and usecase
        model = validate_model(request.model)
        usecase = validate_usecase(request.usecase)
        
        # Extract user message from conversation
        # The last message in the conversation should be the user's current message
        if not request.conversation:
            raise HTTPException(
                status_code=400,
                detail="Conversation cannot be empty. At least one message is required."
            )
        
        # Get the last message as the user input
        last_message = request.conversation[-1]
        user_message = last_message.content
        
        # Optionally validate that last message is from user
        if last_message.role != "user":
            logger.warning(f"Last message role is '{last_message.role}', expected 'user'")
        
        # Process message using assistant service
        response = await assistant_service.process_message(
            user_message=user_message,
            usecase=usecase,
            model=model,
            conversation_id=request.conversation_id,
        )
        
        # Wrap response in legacy format for backward compatibility
        return AssistantResponseWrapper(assistantResponse=response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate assistant response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

