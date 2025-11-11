"""Factory for creating and managing LLM provider instances."""

from typing import Dict, Any, Optional
from src.core.constants import Model
from src.core.logging_config import get_logger
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_gemini_provider import GoogleGeminiProvider

logger = get_logger(__name__)

# Singleton instances
_providers: Dict[Model, Any] = {}


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
    
    if model == Model.LLAMA_3_3_70B:
        provider = HuggingFaceProvider(
            model_name="meta-llama/Llama-3.3-70B-Instruct"
        )
    elif model == Model.GEMINI_2_0_FLASH:
        provider = GoogleGeminiProvider(
            model_name="gemini-2.0-flash-001",
            project_id="explainability-app",
            location="europe-north1"
        )
    elif model == Model.GEMINI_2_5_FLASH:
        provider = GoogleGeminiProvider(
            model_name="gemini-2.5-flash",
            project_id="explainability-app",
            location="europe-north1"
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    _providers[model] = provider
    logger.info(f"Created LLM provider for model: {model.value}")
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
    logger.info("Cleared all LLM provider instances")
