"""Base use case class with lazy loading and common functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Any, Optional
from pathlib import Path
from src.domain.interfaces.model_loader import ModelLoader
from src.domain.interfaces.data_loader import DataLoader
from src.infrastructure.loaders.explainer_loader import ExplainerLoader
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class BaseUseCase(ABC):
    """Base class for all use cases with lazy loading and common functionality."""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        data_loader: DataLoader,
        explainer_loader: Optional[ExplainerLoader] = None,
        config: Any = None,
    ):
        """
        Initialize the base use case.
        
        Args:
            model_loader: Model loader instance
            data_loader: Data loader instance
            explainer_loader: Explainer loader instance (optional)
            config: Use case specific configuration
        """
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.explainer_loader = explainer_loader
        self.config = config
        
        # Lazy-loaded resources
        self._model: Optional[Any] = None
        self._dataset: Optional[Any] = None
        self._y_values: Optional[Any] = None
        self._explainer: Optional[Any] = None
        self._dice_exp: Optional[Any] = None
        self._functions: Optional[Dict[str, Callable]] = None
    
    @property
    def model(self) -> Any:
        """Lazy load and return the model."""
        if self._model is None:
            self._model = self._load_model()
            logger.debug(f"Model loaded for {self.__class__.__name__}")
        return self._model
    
    @property
    def dataset(self) -> Any:
        """Lazy load and return the dataset."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
            logger.debug(f"Dataset loaded for {self.__class__.__name__}")
        return self._dataset
    
    @property
    def y_values(self) -> Any:
        """Lazy load and return y values (target variable)."""
        if self._y_values is None:
            self._y_values = self._load_y_values()
            logger.debug(f"Y values loaded for {self.__class__.__name__}")
        return self._y_values
    
    @property
    def explainer(self) -> Any:
        """Lazy load and return the SHAP explainer."""
        if self._explainer is None:
            self._explainer = self._load_explainer()
            logger.debug(f"Explainer loaded for {self.__class__.__name__}")
        return self._explainer
    
    @property
    def dice_exp(self) -> Any:
        """Lazy load and return the DiCE explainer."""
        if self._dice_exp is None:
            self._dice_exp = self._load_dice_explainer()
            logger.debug(f"DiCE explainer loaded for {self.__class__.__name__}")
        return self._dice_exp
    
    def get_functions(self) -> Dict[str, Callable]:
        """
        Get function registry for this use case.
        
        Returns:
            Dictionary mapping function names to callable functions
        """
        if self._functions is None:
            self._functions = self._create_functions()
            logger.info(f"Functions registered for {self.__class__.__name__}: {len(self._functions)} functions")
        return self._functions
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Load model for this use case. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_dataset(self) -> Any:
        """Load dataset for this use case. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_y_values(self) -> Any:
        """Load y values (target variable) for this use case. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_explainer(self) -> Any:
        """Load SHAP explainer for this use case. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_dice_explainer(self) -> Any:
        """Load DiCE explainer for this use case. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_functions(self) -> Dict[str, Callable]:
        """
        Create function registry for this use case. Must be implemented by subclasses.
        
        Returns:
            Dictionary mapping function names to callable functions
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate system prompt for this use case.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            System prompt string
        """
        pass

