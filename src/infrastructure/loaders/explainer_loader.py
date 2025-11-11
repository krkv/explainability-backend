"""Explainer loader implementation with lazy loading and caching."""

import shap
import copy
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.core.exceptions import ModelLoadException, DataLoadException
from src.core.logging_config import get_logger
from src.infrastructure.loaders.model_loader import CachedModelLoader
from src.infrastructure.loaders.data_loader import CachedDataLoader

logger = get_logger(__name__)


class ExplainerLoader:
    """Explainer loader with caching and lazy loading for SHAP and DiCE explainers."""
    
    def __init__(
        self, 
        model_loader: CachedModelLoader, 
        data_loader: CachedDataLoader,
        max_cache_size: int = 10
    ):
        """
        Initialize the explainer loader.
        
        Args:
            model_loader: Model loader instance
            data_loader: Data loader instance
            max_cache_size: Maximum number of explainers to cache
        """
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.max_cache_size = max_cache_size
        self._loaded_explainers: Dict[str, Any] = {}
    
    def load_shap_explainer(
        self, 
        model_path: str | Path, 
        dataset_path: str | Path,
        explainer_type: str = "kernel"
    ) -> Any:
        """
        Load a SHAP explainer for the given model and dataset.
        
        Args:
            model_path: Path to the model file
            dataset_path: Path to the dataset file
            explainer_type: Type of SHAP explainer ("kernel", "tree", "linear")
            
        Returns:
            Loaded SHAP explainer
            
        Raises:
            ModelLoadException: If model loading fails
            DataLoadException: If dataset loading fails
        """
        model_path_str = str(model_path)
        dataset_path_str = str(dataset_path)
        cache_key = f"shap_{explainer_type}_{model_path_str}_{dataset_path_str}"
        
        # Check if already loaded
        if cache_key in self._loaded_explainers:
            logger.debug(f"SHAP explainer already loaded from cache: {cache_key}")
            return self._loaded_explainers[cache_key]
        
        try:
            # Load model and dataset
            model = self.model_loader.load_model(model_path)
            dataset = self.data_loader.load_dataset(dataset_path)
            
            # Create explainer using implementation
            explainer = self._load_shap_explainer_impl(
                model, dataset, explainer_type
            )
            
            self._loaded_explainers[cache_key] = explainer
            logger.info(f"SHAP explainer loaded successfully: {cache_key}")
            return explainer
            
        except Exception as e:
            logger.error(f"Failed to load SHAP explainer {cache_key}: {e}")
            raise ModelLoadException(f"Failed to load SHAP explainer: {e}")
    
    def _load_shap_explainer_impl(
        self, 
        model: Any, 
        dataset: Any, 
        explainer_type: str
    ) -> Any:
        """
        Internal implementation for creating a SHAP explainer.
        
        Args:
            model: Loaded model object
            dataset: Loaded dataset as pandas DataFrame
            explainer_type: Type of SHAP explainer
            
        Returns:
            SHAP explainer object
        """
        try:
            # Prepare explanation dataset (sample for efficiency)
            explanation_dataset = copy.deepcopy(dataset)
            explanation_dataset = explanation_dataset.to_numpy()
            explanation_dataset = shap.kmeans(explanation_dataset, 25)
            
            # Create appropriate explainer based on type
            if explainer_type == "kernel":
                return shap.KernelExplainer(
                    model.predict, 
                    explanation_dataset, 
                    link="identity"
                )
            elif explainer_type == "tree":
                return shap.TreeExplainer(model)
            elif explainer_type == "linear":
                return shap.LinearExplainer(model, explanation_dataset)
            else:
                raise ValueError(f"Unsupported explainer type: {explainer_type}")
                
        except Exception as e:
            raise ModelLoadException(f"Error creating SHAP explainer: {e}")
    
    def load_dice_explainer(
        self, 
        model_path: str | Path, 
        dataset_path: str | Path,
        continuous_features: List[str],
        categorical_features: List[str] = None,
        outcome_name: str = "prediction"
    ) -> Any:
        """
        Load a DiCE explainer for the given model and dataset.
        
        Args:
            model_path: Path to the model file
            dataset_path: Path to the dataset file
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
            outcome_name: Name of the outcome column
            
        Returns:
            Loaded DiCE explainer
        """
        import dice_ml
        
        model_path_str = str(model_path)
        dataset_path_str = str(dataset_path)
        cache_key = f"dice_{model_path_str}_{dataset_path_str}"
        
        # Check if already loaded
        if cache_key in self._loaded_explainers:
            logger.debug(f"DiCE explainer already loaded from cache: {cache_key}")
            return self._loaded_explainers[cache_key]
        
        try:
            # Load model and dataset
            model = self.model_loader.load_model(model_path)
            dataset = self.data_loader.load_dataset(dataset_path)
            
            # Prepare dataset for DiCE
            dice_dataset = copy.deepcopy(dataset)
            dice_dataset[outcome_name] = model.predict(dice_dataset.to_numpy())
            
            # Create DiCE data and model objects
            dice_data = dice_ml.Data(
                dataframe=dice_dataset,
                continuous_features=continuous_features,
                categorical_features=categorical_features or [],
                outcome_name=outcome_name
            )
            
            dice_model = dice_ml.Model(
                model=model, 
                backend="sklearn",
                model_type="regressor" if "regressor" in str(type(model)) else "classifier"
            )
            
            # Create DiCE explainer
            dice_explainer = dice_ml.Dice(dice_data, dice_model, method="random")
            
            self._loaded_explainers[cache_key] = dice_explainer
            logger.info(f"DiCE explainer loaded successfully: {cache_key}")
            return dice_explainer
            
        except Exception as e:
            logger.error(f"Failed to load DiCE explainer {cache_key}: {e}")
            raise ModelLoadException(f"Failed to load DiCE explainer: {e}")
    
    def is_explainer_loaded(self, cache_key: str) -> bool:
        """
        Check if an explainer is already loaded.
        
        Args:
            cache_key: Cache key for the explainer
            
        Returns:
            True if loaded, False otherwise
        """
        return cache_key in self._loaded_explainers
    
    def unload_explainer(self, cache_key: str) -> None:
        """
        Unload an explainer from memory.
        
        Args:
            cache_key: Cache key for the explainer
        """
        if cache_key in self._loaded_explainers:
            del self._loaded_explainers[cache_key]
            logger.info(f"Explainer unloaded from memory: {cache_key}")
    
    def get_loaded_explainers(self) -> List[str]:
        """
        Get list of currently loaded explainer cache keys.
        
        Returns:
            List of loaded explainer cache keys
        """
        return list(self._loaded_explainers.keys())
    
    def clear_cache(self) -> None:
        """Clear all loaded explainers from cache."""
        self._loaded_explainers.clear()
        logger.info("Explainer cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "loaded_explainers": len(self._loaded_explainers),
            "max_cache_size": self.max_cache_size,
            "cache_keys": list(self._loaded_explainers.keys())
        }
