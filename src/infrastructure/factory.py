"""Factory functions for creating infrastructure components."""

from typing import Optional
from src.core.config import settings
from src.core.logging_config import get_logger
from src.infrastructure.loaders.model_loader import CachedModelLoader
from src.infrastructure.loaders.data_loader import CachedDataLoader
from src.infrastructure.loaders.explainer_loader import ExplainerLoader
from src.infrastructure.caching.cache_manager import CacheManager

logger = get_logger(__name__)

# Global instances (singleton pattern)
_model_loader: Optional[CachedModelLoader] = None
_data_loader: Optional[CachedDataLoader] = None
_explainer_loader: Optional[ExplainerLoader] = None
_cache_manager: Optional[CacheManager] = None


def get_model_loader() -> CachedModelLoader:
    """
    Get the global model loader instance.
    
    Returns:
        CachedModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = CachedModelLoader(max_cache_size=10)  # Default cache size
        logger.info("Model loader created")
    return _model_loader


def get_data_loader() -> CachedDataLoader:
    """
    Get the global data loader instance.
    
    Returns:
        CachedDataLoader instance
    """
    global _data_loader
    if _data_loader is None:
        _data_loader = CachedDataLoader(max_cache_size=10)  # Default cache size
        logger.info("Data loader created")
    return _data_loader


def get_explainer_loader() -> ExplainerLoader:
    """
    Get the global explainer loader instance.
    
    Returns:
        ExplainerLoader instance
    """
    global _explainer_loader
    if _explainer_loader is None:
        model_loader = get_model_loader()
        data_loader = get_data_loader()
        _explainer_loader = ExplainerLoader(
            model_loader=model_loader,
            data_loader=data_loader,
            max_cache_size=10  # Default cache size
        )
        logger.info("Explainer loader created")
    return _explainer_loader


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        model_loader = get_model_loader()
        data_loader = get_data_loader()
        explainer_loader = get_explainer_loader()
        _cache_manager = CacheManager(
            model_loader=model_loader,
            data_loader=data_loader,
            explainer_loader=explainer_loader
        )
        logger.info("Cache manager created")
    return _cache_manager


def reset_infrastructure() -> None:
    """Reset all infrastructure components (useful for testing)."""
    global _model_loader, _data_loader, _explainer_loader, _cache_manager
    
    if _cache_manager:
        _cache_manager.clear_all_caches()
    
    _model_loader = None
    _data_loader = None
    _explainer_loader = None
    _cache_manager = None
    
    logger.info("Infrastructure components reset")
