"""FastAPI dependencies for dependency injection."""

from fastapi import Depends, HTTPException
from typing import Annotated
from src.core.constants import Model, UseCase
from src.core.logging_config import get_logger
from src.services.service_factory import get_assistant_service
from src.services.assistant.assistant_service import AssistantService

logger = get_logger(__name__)


def validate_model(model_str: str) -> Model:
    """
    Validate and convert model string to Model enum.
    
    Args:
        model_str: Model name string
        
    Returns:
        Model enum value
        
    Raises:
        HTTPException: If model is invalid
    """
    try:
        # Match by enum value
        for model_enum in Model:
            if model_enum.value == model_str:
                return model_enum
        
        raise ValueError(f"Invalid model: {model_str}")
        
    except ValueError as e:
        logger.error(f"Invalid model specified: {model_str}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model specified: {model_str}. Supported models: {[m.value for m in Model]}"
        )


def validate_usecase(usecase_str: str) -> UseCase:
    """
    Validate and convert usecase string to UseCase enum.
    
    Args:
        usecase_str: Use case name string
        
    Returns:
        UseCase enum value
        
    Raises:
        HTTPException: If usecase is invalid
    """
    try:
        usecase = UseCase.from_string(usecase_str)
        return usecase
    except ValueError as e:
        logger.error(f"Invalid usecase specified: {usecase_str}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid usecase specified: {usecase_str}. Supported usecases: 'Energy Consumption', 'Heart Disease'"
        )


def get_assistant_service_dependency() -> AssistantService:
    """
    Dependency for getting assistant service instance.
    
    Returns:
        AssistantService instance (singleton)
    """
    return get_assistant_service()


# Type aliases for dependency injection
AssistantServiceDep = Annotated[AssistantService, Depends(get_assistant_service_dependency)]

