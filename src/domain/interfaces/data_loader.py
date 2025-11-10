"""Data loader interface for loading datasets."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pathlib import Path
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load_dataset(self, dataset_path: str | Path) -> pd.DataFrame:
        """
        Load a dataset from the given path.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded dataset as pandas DataFrame
            
        Raises:
            DataLoadException: If dataset loading fails
        """
        pass
    
    @abstractmethod
    def is_dataset_loaded(self, dataset_path: str | Path) -> bool:
        """
        Check if a dataset is already loaded.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            True if loaded, False otherwise
        """
        pass
    
    @abstractmethod
    def unload_dataset(self, dataset_path: str | Path) -> None:
        """
        Unload a dataset from memory.
        
        Args:
            dataset_path: Path to the dataset file
        """
        pass
    
    @abstractmethod
    def get_loaded_datasets(self) -> List[str]:
        """
        Get list of currently loaded dataset paths.
        
        Returns:
            List of loaded dataset paths
        """
        pass
