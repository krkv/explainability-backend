"""Model loader interface for loading ML models."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pathlib import Path


class ModelLoader(ABC):
    """Abstract base class for model loading."""
    
    @abstractmethod
    def load_model(self, model_path: str | Path) -> Any:
        """
        Load a model from the given path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
            
        Raises:
            ModelLoadException: If model loading fails
        """
        pass
    
    @abstractmethod
    def is_model_loaded(self, model_path: str | Path) -> bool:
        """
        Check if a model is already loaded.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded, False otherwise
        """
        pass
    
    @abstractmethod
    def unload_model(self, model_path: str | Path) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model paths.
        
        Returns:
            List of loaded model paths
        """
        pass
