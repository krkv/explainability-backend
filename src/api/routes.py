"""FastAPI routes for API endpoints."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from src.api.schemas import (
    AssistantRequest,
    AssistantResponseWrapper,
    HealthResponse,
    SuggestedFollowUpsRequest,
    SuggestedFollowUpsResponse,
)
from src.api.dependencies import (
    AssistantServiceDep,
    SuggesterServiceDep,
    validate_model,
    validate_usecase,
)
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException
from src.core.logging_config import get_logger
from src.core.observability import TraceContext

logger = get_logger(__name__)
router = APIRouter()


def _convert_assistant_conversation(request_conversation):
    """Preserve assistant function-call markers needed for prompt compaction."""
    conversation = []
    for message in request_conversation:
        serialized = {"role": message.role, "content": message.content}
        if message.is_function_call:
            serialized["is_function_call"] = True
        conversation.append(serialized)
    return conversation


def _convert_basic_conversation(request_conversation):
    """Convert request conversation messages to simple role/content dictionaries."""
    return [{"role": msg.role, "content": msg.content} for msg in request_conversation]


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
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
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
        # Validate model and usecase
        model = validate_model(request.model)
        usecase = validate_usecase(request.usecase)
        
        # Validate conversation is not empty
        if not request.conversation:
            raise HTTPException(
                status_code=400,
                detail="Conversation cannot be empty. At least one message is required."
            )
        
        # Convert conversation from API format to simple dict format
        conversation_dict = _convert_assistant_conversation(request.conversation)
        
        # Process conversation using assistant service (frontend manages conversation history)
        response = await assistant_service.process_message(
            conversation=conversation_dict,
            usecase=usecase,
            model=model,
            trace_context=TraceContext(
                session_id=x_session_id,
                user_id=x_user_id,
                request_id=x_request_id,
            ),
        )
        
        # Wrap response in legacy format for backward compatibility
        return AssistantResponseWrapper(assistantResponse=response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except UpstreamRateLimitException as e:
        logger.warning(f"LLM provider temporarily rate limited: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except LLMProviderException as e:
        logger.error(f"LLM provider error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate assistant response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/getSuggestedFollowUps", response_model=SuggestedFollowUpsResponse, tags=["Assistant"])
async def get_suggested_follow_ups(
    request: SuggestedFollowUpsRequest,
    suggester_service: SuggesterServiceDep,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    """
    Generate optional heart follow-up suggestions.

    This endpoint is intentionally best-effort and returns no suggestions when
    generation fails, so it can be called in parallel with the main assistant
    response without affecting user-facing latency.
    """
    try:
        usecase = validate_usecase(request.usecase)

        if not request.conversation:
            raise HTTPException(
                status_code=400,
                detail="Conversation cannot be empty. At least one message is required."
            )

        conversation_dict = _convert_basic_conversation(request.conversation)

        suggestions = await suggester_service.generate_follow_ups(
            conversation=conversation_dict,
            usecase=usecase,
            limit=request.limit,
            exclude_suggestions=request.exclude_suggestions,
            trace_context=TraceContext(
                session_id=x_session_id,
                user_id=x_user_id,
                request_id=x_request_id,
            ),
        )

        return SuggestedFollowUpsResponse(suggested_follow_ups=suggestions)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("Failed to generate suggested follow-ups: %s", e, exc_info=True)
        return SuggestedFollowUpsResponse(suggested_follow_ups=None)
