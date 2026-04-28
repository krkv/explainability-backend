"""Factory for creating and managing LLM provider instances."""

from typing import Dict, Any, Optional, Union
from src.core.config import GOOGLE_LOCATION, GOOGLE_PROJECT_ID, settings
from src.core.constants import Model
from src.core.logging_config import get_logger
from src.services.llm.google_gemini_provider import GoogleGeminiProvider
from src.services.llm.openai_provider import OpenAIProvider

logger = get_logger(__name__)

# Singleton instances
_providers: Dict[Union[Model, str], Any] = {}


def get_llm_provider(model: Model) -> Any:
    """
    Get or create an LLM provider instance for the specified model.
    
    Args:
        model: The model to get a provider for
        
    Returns:
        LLM provider instance
        
    Raises:
        ValueError: If the model is not supported
    """
    if model in _providers:
        return _providers[model]
    
    if model == Model.GEMINI_3_1_FLASH_LITE_PREVIEW:
        provider = GoogleGeminiProvider(
            model_name="gemini-3.1-flash-lite-preview",
            project_id=GOOGLE_PROJECT_ID,
            location=GOOGLE_LOCATION
        )
    elif model == Model.GPT_5_4_MINI:
        provider = OpenAIProvider(
            model_name="gpt-5.4-mini",
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    _providers[model] = provider
    return provider


def get_google_gemini_provider(
    model_name: str,
    location: Optional[str] = None,
) -> GoogleGeminiProvider:
    """
    Get or create a Google Gemini provider instance for a raw model name.

    Args:
        model_name: Exact Vertex/Gemini model name
        location: Optional Vertex AI location override

    Returns:
        GoogleGeminiProvider instance
    """
    resolved_location = location or GOOGLE_LOCATION
    cache_key = f"google:{model_name}:{resolved_location}"
    if cache_key in _providers:
        return _providers[cache_key]

    provider = GoogleGeminiProvider(
        model_name=model_name,
        project_id=GOOGLE_PROJECT_ID,
        location=resolved_location,
    )
    _providers[cache_key] = provider
    return provider


def get_provider_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available providers.
    
    Returns:
        Dictionary with provider information
    """
    info = {}
    
    for model in Model:
        try:
            provider = get_llm_provider(model)
            info[model.value] = {
                "available": provider.is_available(),
                "model_name": provider.get_model_name(),
                "type": provider.__class__.__name__
            }
        except Exception as e:
            logger.error(f"Failed to get info for model {model.value}: {e}")
            info[model.value] = {
                "available": False,
                "model_name": model.value,
                "type": "Unknown",
                "error": str(e)
            }
    
    return info


def clear_providers() -> None:
    """Clear all cached provider instances."""
    global _providers
    _providers.clear()
