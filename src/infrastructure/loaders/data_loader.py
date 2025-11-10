"""Data loader implementation with lazy loading and caching."""

import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any
from src.domain.interfaces.data_loader import DataLoader
from src.core.exceptions import DataLoadException
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CachedDataLoader(DataLoader):
    """Data loader with caching and lazy loading."""
    
    def __init__(self, max_cache_size: int = 10):
        """
        Initialize the data loader.
        
        Args:
            max_cache_size: Maximum number of datasets to cache
        """
        self.max_cache_size = max_cache_size
        self._loaded_datasets: Dict[str, pd.DataFrame] = {}
        self._load_dataset_cached = lru_cache(maxsize=max_cache_size)(self._load_dataset_impl)
    
    def load_dataset(self, dataset_path: str | Path) -> pd.DataFrame:
        """
        Load a dataset from the given path with caching.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded dataset as pandas DataFrame
            
        Raises:
            DataLoadException: If dataset loading fails
        """
        dataset_path_str = str(dataset_path)
        
        # Check if already loaded in memory
        if dataset_path_str in self._loaded_datasets:
            logger.debug(f"Dataset already loaded from cache: {dataset_path_str}")
            return self._loaded_datasets[dataset_path_str]
        
        try:
            # Load dataset using cached implementation
            dataset = self._load_dataset_cached(dataset_path_str)
            self._loaded_datasets[dataset_path_str] = dataset
            logger.info(f"Dataset loaded successfully: {dataset_path_str} ({len(dataset)} rows)")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path_str}: {e}")
            raise DataLoadException(f"Failed to load dataset from {dataset_path_str}: {e}")
    
    def _load_dataset_impl(self, dataset_path: str) -> pd.DataFrame:
        """
        Internal implementation for loading a dataset.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded dataset as pandas DataFrame
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise DataLoadException(f"Dataset file not found: {dataset_path}")
        
        if not path.is_file():
            raise DataLoadException(f"Path is not a file: {dataset_path}")
        
        # Try loading as CSV
        try:
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            else:
                # Try other formats
                return pd.read_csv(path)
        except Exception as e:
            raise DataLoadException(f"Error loading dataset file {dataset_path}: {e}")
    
    def is_dataset_loaded(self, dataset_path: str | Path) -> bool:
        """
        Check if a dataset is already loaded.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            True if loaded, False otherwise
        """
        return str(dataset_path) in self._loaded_datasets
    
    def unload_dataset(self, dataset_path: str | Path) -> None:
        """
        Unload a dataset from memory.
        
        Args:
            dataset_path: Path to the dataset file
        """
        dataset_path_str = str(dataset_path)
        if dataset_path_str in self._loaded_datasets:
            del self._loaded_datasets[dataset_path_str]
            logger.info(f"Dataset unloaded from memory: {dataset_path_str}")
    
    def get_loaded_datasets(self) -> List[str]:
        """
        Get list of currently loaded dataset paths.
        
        Returns:
            List of loaded dataset paths
        """
        return list(self._loaded_datasets.keys())
    
    def clear_cache(self) -> None:
        """Clear all loaded datasets from cache."""
        self._loaded_datasets.clear()
        self._load_dataset_cached.cache_clear()
        logger.info("Dataset cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = self._load_dataset_cached.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "loaded_datasets": len(self._loaded_datasets)
        }
    
    def get_dataset_info(self, dataset_path: str | Path) -> Dict[str, Any]:
        """
        Get information about a dataset without loading it.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dictionary with dataset information
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise DataLoadException(f"Dataset file not found: {dataset_path}")
        
        try:
            # Read just the first few rows to get info
            sample = pd.read_csv(path, nrows=5)
            return {
                "path": str(dataset_path),
                "columns": list(sample.columns),
                "dtypes": sample.dtypes.to_dict(),
                "sample_rows": len(sample)
            }
        except Exception as e:
            raise DataLoadException(f"Error reading dataset info from {dataset_path}: {e}")
