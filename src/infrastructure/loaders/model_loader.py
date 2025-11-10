"""Model loader implementation with lazy loading and caching."""

import joblib
import gplearn
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.domain.interfaces.model_loader import ModelLoader
from src.core.exceptions import ModelLoadException
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CachedModelLoader(ModelLoader):
    """Model loader with caching and lazy loading."""
    
    def __init__(self, max_cache_size: int = 10):
        """
        Initialize the model loader.
        
        Args:
            max_cache_size: Maximum number of models to cache
        """
        self.max_cache_size = max_cache_size
        self._loaded_models: Dict[str, Any] = {}
        self._load_model_cached = lru_cache(maxsize=max_cache_size)(self._load_model_impl)
    
    def load_model(self, model_path: str | Path) -> Any:
        """
        Load a model from the given path with caching.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
            
        Raises:
            ModelLoadException: If model loading fails
        """
        model_path_str = str(model_path)
        
        # Check if already loaded in memory
        if model_path_str in self._loaded_models:
            logger.debug(f"Model already loaded from cache: {model_path_str}")
            return self._loaded_models[model_path_str]
        
        try:
            # Load model using cached implementation
            model = self._load_model_cached(model_path_str)
            self._loaded_models[model_path_str] = model
            logger.info(f"Model loaded successfully: {model_path_str}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path_str}: {e}")
            raise ModelLoadException(f"Failed to load model from {model_path_str}: {e}")
    
    def _load_model_impl(self, model_path: str) -> Any:
        """
        Internal implementation for loading a model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        path = Path(model_path)
        
        if not path.exists():
            raise ModelLoadException(f"Model file not found: {model_path}")
        
        if not path.is_file():
            raise ModelLoadException(f"Path is not a file: {model_path}")
        
        try:
            return joblib.load(path)
        except Exception as e:
            raise ModelLoadException(f"Error loading model file {model_path}: {e}")
    
    def is_model_loaded(self, model_path: str | Path) -> bool:
        """
        Check if a model is already loaded.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded, False otherwise
        """
        return str(model_path) in self._loaded_models
    
    def unload_model(self, model_path: str | Path) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_path: Path to the model file
        """
        model_path_str = str(model_path)
        if model_path_str in self._loaded_models:
            del self._loaded_models[model_path_str]
            logger.info(f"Model unloaded from memory: {model_path_str}")
    
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model paths.
        
        Returns:
            List of loaded model paths
        """
        return list(self._loaded_models.keys())
    
    def clear_cache(self) -> None:
        """Clear all loaded models from cache."""
        self._loaded_models.clear()
        self._load_model_cached.cache_clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = self._load_model_cached.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "loaded_models": len(self._loaded_models)
        }
