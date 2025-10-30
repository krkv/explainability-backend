"""Heart use case implementation."""

import json
import copy
import shap
import dice_ml
import pandas as pd
from pathlib import Path
from typing import Dict, Callable, List, Any
from src.usecases.base.base_usecase import BaseUseCase
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_functions import HeartFunctions
from src.infrastructure.loaders.explainer_loader import ExplainerLoader
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class HeartUseCase(BaseUseCase):
    """Heart disease use case with lazy loading and refactored functions."""
    
    def __init__(
        self,
        model_loader,
        data_loader,
        explainer_loader: ExplainerLoader = None,
        config: HeartConfig = None,
    ):
        """
        Initialize heart use case.
        
        Args:
            model_loader: Model loader instance
            data_loader: Data loader instance
            explainer_loader: Explainer loader instance
            config: Heart configuration
        """
        if config is None:
            config = HeartConfig()
        
        super().__init__(model_loader, data_loader, explainer_loader, config)
        
        # Heart-specific lazy-loaded resources
        self._dataset_full: pd.DataFrame = None
        self._dice_dataset: pd.DataFrame = None
        self._model_metadata: Dict[str, Any] = None
        self._feature_metadata: Dict[str, Any] = None
        self._alias_lookup: Dict[str, str] = None
        self._global_explainer: Any = None
        self._global_shap_values: Any = None
        self._global_feature_importances: Dict[str, float] = None
    
    @property
    def dataset_full(self) -> pd.DataFrame:
        """Lazy load and return the full dataset (with target variable)."""
        if self._dataset_full is None:
            self._dataset_full = self.data_loader.load_dataset(self.config.dataset_path).copy()
            logger.debug("Full dataset loaded for HeartUseCase")
        return self._dataset_full
    
    @property
    def model_metadata(self) -> Dict[str, Any]:
        """Lazy load and return model metadata."""
        if self._model_metadata is None:
            with open(self.config.model_metadata_path, 'r') as f:
                self._model_metadata = json.load(f)
            logger.debug("Model metadata loaded for HeartUseCase")
        return self._model_metadata
    
    @property
    def feature_metadata(self) -> Dict[str, Any]:
        """Lazy load and return feature metadata."""
        if self._feature_metadata is None:
            with open(self.config.feature_metadata_path, 'r') as f:
                self._feature_metadata = json.load(f)
            logger.debug("Feature metadata loaded for HeartUseCase")
        return self._feature_metadata
    
    @property
    def alias_lookup(self) -> Dict[str, str]:
        """Lazy load and return alias lookup dictionary."""
        if self._alias_lookup is None:
            self._alias_lookup = {
                alias.lower(): feat
                for feat, meta in self.feature_metadata.items()
                for alias in ([feat] + meta.get("aliases", []))
            }
            logger.debug("Alias lookup created for HeartUseCase")
        return self._alias_lookup
    
    @property
    def global_feature_importances(self) -> Dict[str, float]:
        """Lazy load and return global feature importances."""
        if self._global_feature_importances is None:
            import numpy as np
            # Compute global SHAP values
            if self._global_explainer is None:
                self._global_explainer = shap.Explainer(self.model.predict, self.dataset)
            if self._global_shap_values is None:
                self._global_shap_values = self._global_explainer(self.dataset)
            
            importance = np.abs(self._global_shap_values.values).mean(axis=0)
            feature_names = self.dataset.columns.tolist()
            self._global_feature_importances = dict(
                sorted(
                    {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
            logger.debug("Global feature importances computed for HeartUseCase")
        return self._global_feature_importances
    
    def _load_model(self) -> Any:
        """Load model for heart use case."""
        return self.model_loader.load_model(self.config.model_path)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset for heart use case (without target variable)."""
        dataset = self.data_loader.load_dataset(self.config.dataset_path)
        # Return a copy so we can modify it (drop target) without affecting cached version
        dataset_copy = dataset.copy()
        # Remove target variable if it exists
        if self.config.target_variable in dataset_copy.columns:
            dataset_copy = dataset_copy.drop(columns=[self.config.target_variable])
        return dataset_copy
    
    def _load_y_values(self) -> pd.Series:
        """Load y values (target variable) from original dataset."""
        # Load full dataset to get y values
        dataset_full = self.data_loader.load_dataset(self.config.dataset_path)
        if self.config.target_variable in dataset_full.columns:
            y_values = dataset_full[self.config.target_variable].copy()
            logger.debug(f"Y values extracted from dataset: {len(y_values)} values")
            return y_values
        else:
            raise ValueError(f"Target variable '{self.config.target_variable}' not found in dataset")
    
    def _load_explainer(self) -> Any:
        """Load SHAP explainer for heart use case."""
        if self.explainer_loader:
            try:
                return self.explainer_loader.load_shap_explainer(
                    self.config.model_path,
                    self.config.dataset_path,
                    explainer_type=self.config.shap_explainer_type
                )
            except Exception as e:
                logger.warning(f"Failed to use explainer loader, creating directly: {e}")
        
        # Fallback: create explainer directly
        explanation_dataset = copy.deepcopy(self.dataset)
        explanation_dataset = explanation_dataset.to_numpy()
        explanation_dataset = shap.kmeans(explanation_dataset, self.config.shap_sample_size)
        return shap.KernelExplainer(self.model.predict, explanation_dataset)
    
    def _load_dice_explainer(self) -> Any:
        """Load DiCE explainer for heart use case."""
        # Prepare dice dataset (dataset with predictions)
        if self._dice_dataset is None:
            self._dice_dataset = copy.deepcopy(self.dataset)
            self._dice_dataset['prediction'] = self.model.predict(self._dice_dataset.to_numpy())
        
        dice_data = dice_ml.Data(
            dataframe=copy.deepcopy(self._dice_dataset),
            continuous_features=self.config.dice_continuous_features,
            categorical_features=self.config.dice_categorical_features,
            outcome_name=self.config.dice_outcome_name
        )
        
        dice_model = dice_ml.Model(
            model=self.model,
            backend="sklearn",
            model_type=self.config.dice_model_type
        )
        
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")
        return dice_exp
    
    def _get_dice_dataset(self) -> pd.DataFrame:
        """Get dice dataset (dataset without prediction column)."""
        if self._dice_dataset is None:
            # Trigger dice explainer creation which sets up dice_dataset
            _ = self.dice_exp
        # Return copy without prediction column
        dice_dataset_copy = copy.deepcopy(self._dice_dataset)
        if 'prediction' in dice_dataset_copy.columns:
            dice_dataset_copy.pop('prediction')
        return dice_dataset_copy
    
    def _create_functions(self) -> Dict[str, Callable]:
        """Create function registry for heart use case."""
        heart_funcs = HeartFunctions(
            model=self.model,
            dataset=self.dataset,
            dataset_full=self.dataset_full,
            y_values=self.y_values,
            explainer=self.explainer,
            dice_exp=self.dice_exp,
            dice_dataset=self._get_dice_dataset(),
            model_metadata=self.model_metadata,
            feature_metadata=self.feature_metadata,
            alias_lookup=self.alias_lookup,
            global_feature_importances=self.global_feature_importances,
            target_variable=self.config.target_variable,
            class_names=self.config.class_names,
            feature_names=self.dataset.columns.tolist(),
        )
        
        return {
            'get_model_parameters': heart_funcs.get_model_parameters,
            'get_model_description': heart_funcs.get_model_description,
            'predict': heart_funcs.predict,
            'feature_importance_patient': heart_funcs.feature_importance_patient,
            'feature_importance_global': heart_funcs.feature_importance_global,
            'dataset_summary': heart_funcs.dataset_summary,
            'performance_metrics': heart_funcs.performance_metrics,
            'confusion_matrix_stats': heart_funcs.confusion_matrix_stats,
            'what_if': heart_funcs.what_if,
            'counterfactual': heart_funcs.counterfactual,
            'misclassified_cases': heart_funcs.misclassified_cases,
            'age_group_performance': heart_funcs.age_group_performance,
            'feature_interactions': heart_funcs.feature_interactions,
        }
    
    def get_system_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Generate system prompt for heart use case."""
        # Load functions.json
        with open(self.config.functions_json_path, 'r') as f:
            functions = json.load(f)
        
        # Get dataset description
        dataset_json = self.dataset.describe().to_json()
        
        # JSON schema for structured responses
        response_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Response",
            "type": "object",
            "properties": {
                "function_calls": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of function calls in string format."
                },
                "freeform_response": {
                    "type": "string",
                    "description": "A free-form response that strictly follows the rules of the assistant."
                }
            },
            "required": ["function_calls", "freeform_response"],
            "additionalProperties": "false"
        }
        
        # Format conversation for the prompt
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        system_prompt = f"""
    Your name is Claire. You are a trustworthy data science assistant that helps user to understand the data, model and predictions for a machine learning model application use case in medical sector.
    Here is the description of the dataset:
    
    {dataset_json}

    The model and the dataset are not available to you directly, but you have access to a set of functions that can be invoked to help the user.
    Here is the list of functions that can be invoked. ONLY these functions can be called:

    {json.dumps(functions, indent=2)}
    
    You are an expert in composing function calls. You are given a user query and a set of possible functions that you can call. Based on the user query, you need to decide whether any functions can be called or not.
    You are a trustworthy assistant, which means you should not make up any information or provide any answers that are not supported by the functions given above.
    
    Respond ONLY in JSON format, following strictly this JSON schema for your response:
    
    {json.dumps(response_schema, indent=2)}
    
    Please use double quotes for the keys and values in the JSON response. Do not use single quotes.
   
    If you decide to invoke one or several of the available functions, you MUST include them in the JSON response field "function_calls" in format "[func_name1(params_name1=params_value1, params_name2=params_value2...),func_name1(params_name1=params_value1, params_name2=params_value2...)]".
    When adding param values, only use param values given by user. Do not use any other values or make up any values.
    If you decide that no function(s) can be called, you should return an empty list [] as "function_calls".
      
    Your free-form response in JSON field "freeform_response" is mandatory and it should be a short comment about what you are trying to achieve with chosen function calls. 
    If user asked a question about data/model/prediction and it can not be answered with the available functions, your freeform_response should not try to answer this question. Just say that you are not able to answer this question and ask if user wants to see the list of available functions.

    You are also given the full history of user's messages in this conversation.
    Use this history to understand the context of the user query, for example, infer an ID or group filtering from the previous user query.
    Use user's query history to understand the question better and guide your responses if needed.

    {conversation_text}
    """
        
        return system_prompt

