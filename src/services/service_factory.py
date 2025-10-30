"""Factory for creating and managing singleton instances of all service layer components."""

from typing import Any, Dict, Optional
from src.core.logging_config import get_logger
from src.services.usecase.usecase_registry_service import UseCaseRegistryService
from src.services.function.function_executor_service import FunctionExecutorService
from src.services.assistant.assistant_service import AssistantService
from src.services.llm.llm_factory import get_provider_info

logger = get_logger(__name__)

# Singleton instances
_usecase_registry: Optional[UseCaseRegistryService] = None
_function_executor: Optional[FunctionExecutorService] = None
_assistant_service: Optional[AssistantService] = None


def get_usecase_registry() -> UseCaseRegistryService:
    """
    Get or create the use case registry service.
    
    Returns:
        UseCaseRegistryService instance
    """
    global _usecase_registry
    if _usecase_registry is None:
        _usecase_registry = UseCaseRegistryService()
        logger.info("Created UseCaseRegistryService singleton")
    return _usecase_registry


def get_function_executor() -> FunctionExecutorService:
    """
    Get or create the function executor service.
    
    Returns:
        FunctionExecutorService instance
    """
    global _function_executor
    if _function_executor is None:
        usecase_registry = get_usecase_registry()
        _function_executor = FunctionExecutorService(usecase_registry)
        logger.info("Created FunctionExecutorService singleton")
    return _function_executor


def get_assistant_service() -> AssistantService:
    """
    Get or create the assistant service.
    
    Returns:
        AssistantService instance
    """
    global _assistant_service
    if _assistant_service is None:
        usecase_registry = get_usecase_registry()
        function_executor = get_function_executor()
        _assistant_service = AssistantService(function_executor, usecase_registry)
        logger.info("Created AssistantService singleton")
    return _assistant_service


def get_service_stats() -> Dict[str, Any]:
    """
    Get statistics about all services.
    
    Returns:
        Dictionary with service statistics
    """
    stats = {
        "usecase_registry": {
            "registered_usecases": len(get_usecase_registry().get_registered_usecases()),
            "usecases": [uc.value for uc in get_usecase_registry().get_registered_usecases()]
        },
        "function_executor": get_function_executor().get_parser_stats(),
        "assistant_service": get_assistant_service().get_conversation_stats(),
        "llm_providers": get_provider_info()
    }
    return stats


def clear_all_services() -> None:
    """Clear all service instances (useful for testing)."""
    global _usecase_registry, _function_executor, _assistant_service
    
    _usecase_registry = None
    _function_executor = None
    _assistant_service = None
    
    # Also clear LLM providers
    from src.services.llm.llm_factory import clear_providers
    clear_providers()
    
    logger.info("Cleared all service instances")
