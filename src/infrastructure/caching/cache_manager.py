"""Cache manager for coordinating all application caches."""

from typing import Dict, Any, Optional
from src.core.logging_config import get_logger
from src.infrastructure.loaders.model_loader import CachedModelLoader
from src.infrastructure.loaders.data_loader import CachedDataLoader
from src.infrastructure.loaders.explainer_loader import ExplainerLoader

logger = get_logger(__name__)


class CacheManager:
    """Centralized cache manager for all application caches."""
    
    def __init__(
        self,
        model_loader: CachedModelLoader,
        data_loader: CachedDataLoader,
        explainer_loader: ExplainerLoader
    ):
        """
        Initialize the cache manager.
        
        Args:
            model_loader: Model loader instance
            data_loader: Data loader instance
            explainer_loader: Explainer loader instance
        """
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.explainer_loader = explainer_loader
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics from all loaders
        """
        return {
            "models": self.model_loader.get_cache_info(),
            "datasets": self.data_loader.get_cache_info(),
            "explainers": self.explainer_loader.get_cache_info()
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        logger.info("Clearing all caches")
        self.model_loader.clear_cache()
        self.data_loader.clear_cache()
        self.explainer_loader.clear_cache()
        logger.info("All caches cleared")
    
    def clear_model_cache(self) -> None:
        """Clear model cache only."""
        logger.info("Clearing model cache")
        self.model_loader.clear_cache()
    
    def clear_data_cache(self) -> None:
        """Clear dataset cache only."""
        logger.info("Clearing dataset cache")
        self.data_loader.clear_cache()
    
    def clear_explainer_cache(self) -> None:
        """Clear explainer cache only."""
        logger.info("Clearing explainer cache")
        self.explainer_loader.clear_cache()
    
    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """
        Get estimated memory usage for loaded resources.
        
        Returns:
            Dictionary with memory usage estimates
        """
        import sys
        
        model_memory = 0
        for model_path in self.model_loader.get_loaded_models():
            try:
                model = self.model_loader._loaded_models[model_path]
                model_memory += sys.getsizeof(model)
            except:
                pass
        
        dataset_memory = 0
        for dataset_path in self.data_loader.get_loaded_datasets():
            try:
                dataset = self.data_loader._loaded_datasets[dataset_path]
                dataset_memory += dataset.memory_usage(deep=True).sum()
            except:
                pass
        
        explainer_memory = 0
        for explainer_key in self.explainer_loader.get_loaded_explainers():
            try:
                explainer = self.explainer_loader._loaded_explainers[explainer_key]
                explainer_memory += sys.getsizeof(explainer)
            except:
                pass
        
        return {
            "model_memory_bytes": model_memory,
            "dataset_memory_bytes": dataset_memory,
            "explainer_memory_bytes": explainer_memory,
            "total_memory_bytes": model_memory + dataset_memory + explainer_memory,
            "total_memory_mb": (model_memory + dataset_memory + explainer_memory) / (1024 * 1024)
        }
    
    def optimize_caches(self, max_memory_mb: float = 1000) -> None:
        """
        Optimize caches by clearing least recently used items if memory usage is too high.
        
        Args:
            max_memory_mb: Maximum memory usage in MB before optimization
        """
        memory_usage = self.get_memory_usage_estimate()
        
        if memory_usage["total_memory_mb"] > max_memory_mb:
            logger.warning(
                f"Memory usage ({memory_usage['total_memory_mb']:.2f} MB) "
                f"exceeds limit ({max_memory_mb} MB). Clearing caches."
            )
            self.clear_all_caches()
        else:
            logger.info(
                f"Memory usage ({memory_usage['total_memory_mb']:.2f} MB) "
                f"is within limit ({max_memory_mb} MB)"
            )
    
    def get_loaded_resources(self) -> Dict[str, Any]:
        """
        Get information about all loaded resources.
        
        Returns:
            Dictionary with loaded resource information
        """
        return {
            "loaded_models": self.model_loader.get_loaded_models(),
            "loaded_datasets": self.data_loader.get_loaded_datasets(),
            "loaded_explainers": self.explainer_loader.get_loaded_explainers(),
            "total_loaded": (
                len(self.model_loader.get_loaded_models()) +
                len(self.data_loader.get_loaded_datasets()) +
                len(self.explainer_loader.get_loaded_explainers())
            )
        }
