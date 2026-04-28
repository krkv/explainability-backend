"""Heart use case implementation."""

import json
import copy
import shap
import dice_ml
import pandas as pd
from pathlib import Path
from typing import Dict, Callable, List, Any, Optional
from src.domain.interfaces.llm_provider import (
    AgentRole,
    StructuredGenerationConfig,
    build_response_schema,
)
from src.usecases.base.base_usecase import BaseUseCase
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_functions import HeartFunctions
from src.infrastructure.loaders.explainer_loader import ExplainerLoader
from src.core.prompt_history import build_compact_conversation_history
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
                for alias in ([feat, meta.get("display_name", feat)] + meta.get("aliases", []))
            }
            logger.debug("Alias lookup created for HeartUseCase")
        return self._alias_lookup
    
    @property
    def global_feature_importances(self) -> Dict[str, float]:
        """Lazy load and return global feature importances."""
        if self._global_feature_importances is None:
            import numpy as np
            import os, pickle
            
            cache_path = self.config.global_fi_cache_path
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        self._global_feature_importances = pickle.load(f)
                        logger.debug("Global feature importances loaded from cache")
                        return self._global_feature_importances
                except Exception as e:
                    logger.warning(f"Failed to load global feature importances from cache: {e}")
            
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
            
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self._global_feature_importances, f)
                except Exception as e:
                    logger.warning(f"Failed to save global feature importances to cache: {e}")
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
                    explainer_type=self.config.shap_explainer_type,
                    target_variable=self.config.target_variable
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
        functions_catalog = self._load_functions_catalog()

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
            functions_catalog=functions_catalog,
            shap_cache_path=self.config.shap_cache_path,
            cf_cache_path=self.config.cf_cache_path,
        )
        
        return {
            'available_functions': heart_funcs.available_functions,
            'get_model_parameters': heart_funcs.get_model_parameters,
            'get_model_description': heart_funcs.get_model_description,
            'count_patients': heart_funcs.count_patients,
            'predict': heart_funcs.predict,
            'feature_importance_patient': heart_funcs.feature_importance_patient,
            'feature_importance_global': heart_funcs.feature_importance_global,
            'dataset_summary': heart_funcs.dataset_summary,
            'define_feature': heart_funcs.define_feature,
            'performance_metrics': heart_funcs.performance_metrics,
            'confusion_matrix_stats': heart_funcs.confusion_matrix_stats,
            'what_if': heart_funcs.what_if,
            'counterfactual': heart_funcs.counterfactual,
            'misclassified_cases': heart_funcs.misclassified_cases,
            'age_group_performance': heart_funcs.age_group_performance,
            'feature_interactions': heart_funcs.feature_interactions,
            'show_ids': heart_funcs.show_ids,
            'show_one': heart_funcs.show_one,
        }

    def _load_functions_catalog(self) -> List[Dict[str, Any]]:
        """Load the heart function catalog from disk."""
        with open(self.config.functions_json_path, 'r') as f:
            return json.load(f)

    def _get_latest_assistant_response(
        self,
        conversation: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the latest assistant response for suggester prompt context."""
        if context and context.get("latest_assistant_response"):
            return str(context["latest_assistant_response"])

        for message in reversed(conversation):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message["content"])

        return ""

    def _build_prompt_context(
        self,
        conversation: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Build the shared serialized prompt context for heart agents."""
        return {
            "dataset_json": self.dataset.describe().to_json(),
            "feature_metadata_json": json.dumps(self.feature_metadata, indent=2),
            "functions_json": json.dumps(self._load_functions_catalog(), indent=2),
            "compact_conversation_history_json": build_compact_conversation_history(conversation),
            "conversation_json": json.dumps(conversation),
            "latest_assistant_response": self._get_latest_assistant_response(conversation, context),
        }

    def _build_assistant_system_prompt(self, prompt_context: Dict[str, str]) -> str:
        """Build the main assistant prompt for the heart use case."""
        response_schema = build_response_schema(AgentRole.ASSISTANT)

        return f"""
    Your name is Claire. You are a trustworthy data science assistant that helps user to understand the data, model and predictions for a machine learning model application use case in medical sector.
    Here is the description of the dataset:
    
    {prompt_context["dataset_json"]}

    Here is the feature metadata with display names, aliases, descriptions, and categorical options:

    {prompt_context["feature_metadata_json"]}

    The model and the dataset are not available to you directly, but you have access to a set of functions that can be invoked to help the user.
    Here is the list of functions that can be invoked. ONLY these functions can be called:

    {prompt_context["functions_json"]}
    
    You are an expert in composing function calls. You are given a user query and a set of possible functions that you can call. Based on the user query, you need to decide whether any functions can be called or not.
    You are a trustworthy assistant, which means you should not make up any information or provide any answers that are not supported by the functions given above.
    
    Respond ONLY in JSON format, following strictly this JSON schema for your response:
    
    {json.dumps(response_schema, indent=2)}
    
    Please use double quotes for the keys and values in the JSON response. Do not use single quotes.
   
    If you decide to invoke one or several of the available functions, you MUST include them in the JSON response field "function_calls" in format "[func_name1(params_name1=params_value1, params_name2=params_value2...),func_name1(params_name1=params_value1, params_name2=params_value2...)]".
    When adding param values, preserve the user's intent and only use values supported by the user message, function catalog, or feature metadata. You may normalize a user's wording to the matching canonical feature name or categorical label/code shown in the metadata. Do not invent unsupported values.
    For patient count questions, use the patient-counting tool from the function catalog. For example, use the positive predicted count option when the user asks how many patients have heart disease, the negative predicted count option for how many do not, and the total option for the dataset size.
    Users may refer to features by display name, alias, or shorthand label, for example "chest pain", "blood pressure", or "typical". Convert those into the appropriate function arguments instead of refusing.
    For what_if on categorical features, pass the desired category label or code as a string, for example value_change="Typical angina". For numeric features, pass a numeric change.
    If you decide that no function(s) can be called, you should return an empty list [] as "function_calls".
    If the user asks to see the list of available functions or asks what functions you can use, call <code>available_functions()</code>.
      
    Your free-form response in JSON field "freeform_response" is mandatory and it should be a short comment about what you are trying to achieve with chosen function calls. 
    If user asked a question about data/model/prediction and it can not be answered with the available functions, your freeform_response should not try to answer this question. Just say that you are not able to answer this question and ask if user wants to see the list of available functions.

    You are also given recent prior conversation turns with invoked function calls.
    Use this compact history only to resolve references from the current user query, such as IDs, previously selected records, or prior filters.

    {prompt_context["compact_conversation_history_json"]}
    """

    def _build_suggester_system_prompt(self, prompt_context: Dict[str, str]) -> str:
        """Build the follow-up suggester prompt for the heart use case."""
        response_schema = build_response_schema(AgentRole.SUGGESTER)
        latest_assistant_response = (
            prompt_context["latest_assistant_response"]
            if prompt_context["latest_assistant_response"]
            else "No assistant response is available yet."
        )

        return f"""
    You are acting as a follow-up suggester for a heart disease explainability assistant used by a medical professional who is evaluating the usefulness of the assistant.
    Your job is to suggest the next 3 to 5 user prompts that would help this evaluator explore the dataset, the model, patient-level explanations, counterfactuals, what-if analysis, errors, and performance characteristics.

    You are not answering the user. You are proposing short next user messages only.

    Example suggestions:
    'What data is loaded?',
    'How many patients are there?',
    'What are the patient IDs?',
    'What kind of model is used?',
    'Show the features of patient 2',
    'Is this patient prediction positive?',
    'Why did the model predict that this patient has heart disease?',
    'If the patient\'s blood pressure was lower by 15, would the risk of heart disease decrease?',
    'How would it be possible to change this prediction?',
    'If this patient were 10 years younger, how would that affect the prediction?',
    'Which patients did the model misclassify most often?',
    'What is the false positive rate of the model?',
    'What is the precision and recall of the model?',
    'How well does the model generalize across different age groups?',
    'What is the AUC-ROC score of the model?',
    'How does the model generally decide whether a patient has heart disease?',
    'How do different risk factors interact with each other in the model\'s decision-making process?'.

    Here is the description of the dataset:

    {prompt_context["dataset_json"]}

    Here is the feature metadata with display names, aliases, descriptions, and categorical options:

    {prompt_context["feature_metadata_json"]}

    Here is the list of heart functions that the main assistant can use. Use this catalog to understand the real capabilities that your suggestions should showcase:

    {prompt_context["functions_json"]}

    Here is the full conversation history:

    {prompt_context["conversation_json"]}

    Here is the latest assistant response from this conversation:

    {latest_assistant_response}

    Respond ONLY in JSON format, following strictly this JSON schema for your response:

    {json.dumps(response_schema, indent=2)}

    Each suggestion must be a plain user utterance that could be shown as an example prompt in the UI.
    Keep suggestions specific, concise, and varied across different supported capabilities.
    Suggestions must stay within the supported heart-disease functionality. Do not invent capabilities.
    Do not mention function names, tools, JSON, schemas, or implementation details.
    Do not number the suggestions, do not include rationale, and do not repeat near-duplicates.
    Prefer prompts that demonstrate clinically meaningful exploration for a professional audience, such as reviewing patient risk, understanding which features matter, comparing groups, inspecting performance, or testing plausible feature changes.
    """

    def get_generation_config(
        self,
        conversation: List[Dict[str, str]],
        agent_role: AgentRole = AgentRole.ASSISTANT,
        context: Optional[Dict[str, Any]] = None,
    ) -> StructuredGenerationConfig:
        """Build the role-aware prompt and schema for the heart use case."""
        prompt_context = self._build_prompt_context(conversation, context)

        if agent_role == AgentRole.SUGGESTER:
            return StructuredGenerationConfig(
                system_prompt=self._build_suggester_system_prompt(prompt_context),
                response_schema=build_response_schema(AgentRole.SUGGESTER),
            )

        return StructuredGenerationConfig(
            system_prompt=self._build_assistant_system_prompt(prompt_context),
            response_schema=build_response_schema(AgentRole.ASSISTANT),
        )
