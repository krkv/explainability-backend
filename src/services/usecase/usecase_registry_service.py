"""Use case registry service for managing use cases, functions, and system prompts."""

from typing import Dict, Callable, Any, List
from src.core.constants import UseCase
from src.core.exceptions import FunctionExecutionException
from src.core.logging_config import get_logger
from src.domain.interfaces.usecase_registry import UseCaseRegistry
from src.domain.entities.message import Message

logger = get_logger(__name__)


class UseCaseRegistryService(UseCaseRegistry):
    """
    Manages the registration and retrieval of functions and system prompts
    for different use cases.
    """
    
    def __init__(self):
        self._registries: Dict[UseCase, Dict[str, Callable]] = {}
        self._system_prompts: Dict[UseCase, str] = {}
        self._initialize_default_prompts()
        logger.info("UseCaseRegistryService initialized")
    
    def register_usecase(self, usecase: UseCase, functions: Dict[str, Callable]) -> None:
        """
        Register functions for a use case.
        
        Args:
            usecase: The use case to register
            functions: Dictionary mapping function names to callable functions
        """
        if not isinstance(usecase, UseCase):
            raise ValueError("Usecase must be an instance of UseCase enum.")
        if not isinstance(functions, dict):
            raise ValueError("Functions must be a dictionary.")
        
        self._registries[usecase] = functions
        logger.info(f"Registered {len(functions)} functions for usecase '{usecase.value}'")
    
    def register_usecase_functions(self, usecase: UseCase, functions: Dict[str, Callable]):
        """
        Registers a dictionary of functions for a specific use case.
        (Alias for register_usecase for backward compatibility)
        
        Args:
            usecase: The name of the use case (e.g., UseCase.ENERGY).
            functions: A dictionary where keys are function names (str) and values are callable functions.
        """
        self.register_usecase(usecase, functions)
    
    def get_function(self, usecase: UseCase, function_name: str) -> Callable:
        """
        Retrieves a specific function for a given use case.
        
        Args:
            usecase: The name of the use case.
            function_name: The name of the function to retrieve.
            
        Returns:
            The callable function.
            
        Raises:
            FunctionExecutionException: If the use case or function is not found.
        """
        if usecase not in self._registries:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' not registered.")
        
        if function_name not in self._registries[usecase]:
            raise FunctionExecutionException(f"Function '{function_name}' not found in usecase '{usecase.value}'.")
        
        return self._registries[usecase][function_name]
    
    def get_functions(self, usecase: UseCase) -> Dict[str, Callable]:
        """
        Retrieves all registered functions for a specific use case.
        
        Args:
            usecase: The name of the use case.
            
        Returns:
            A dictionary of callable functions.
            
        Raises:
            FunctionExecutionException: If the use case is not found.
        """
        if usecase not in self._registries:
            raise FunctionExecutionException(f"Usecase '{usecase.value}' not registered.")
        return self._registries[usecase]
    
    def get_system_prompt(self, usecase: UseCase, conversation: List[Dict[str, str]]) -> str:
        """
        Retrieves the system prompt for a given use case with embedded data and functions.
        
        Args:
            usecase: The use case for which to retrieve the system prompt.
            conversation: The list of past messages in the conversation.
            
        Returns:
            The complete system prompt with embedded data and functions.
        """
        if usecase == UseCase.ENERGY:
            return self._get_energy_system_prompt(conversation)
        elif usecase == UseCase.HEART:
            return self._get_heart_system_prompt(conversation)
        else:
            return self._get_default_prompt()
    
    def _get_energy_system_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Generate the energy system prompt with embedded data and functions."""
        import pandas as pd
        import json
        
        # Load energy dataset and functions
        energy_dataset = pd.read_csv('instances/energy/data/summer_workday_test.csv')
        dataset_json = energy_dataset.describe().to_json()
        
        with open('instances/energy/functions.json') as f:
            functions = json.load(f)
        
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
    Your name is Claire. You are a trustworthy data science assistant that helps user to understand the data, model and predictions for a machine learning model application use case in energy sector.
    Here is the description of the dataset:
    
    {dataset_json}

    The model and the dataset are not available to you directly, but you have access to a set of functions that can be invoked to help the user.
    Here is the list of functions that can be invoked. ONLY these functions can be called:

    {functions}
    
    You are an expert in composing function calls. You are given a user query and a set of possible functions that you can call. Based on the user query, you need to decide whether any functions can be called or not.
    You are a trustworthy assistant, which means you should not make up any information or provide any answers that are not supported by the functions given above.
    
    Respond ONLY in JSON format, following strictly this JSON schema for your response:
    
    {response_schema}
    
    Please use double quotes for the keys and values in the JSON response. Do not use single quotes.
   
    If you decide to invoke one or several of the available functions, you MUST include them in the JSON response field "function_calls" in format "[func_name1(params_name1=params_value1, params_name2=params_value2...),func_name1(params_name1=params_value1, params_name2=params_value2...)]".
    When adding param values, only use param values given by user. Do not use any other values or make up any values.
    If you decide that no function(s) can be called, you should return an empty list [] as "function_calls".
      
    Your free-form response in JSON field "freeform_response" is mandatory and it should be a short comment about what you are trying to achieve with chosen function calls. 
    If user asked a question about data/model/prediction and it can not be answered with the available functions, your free-form response should not try to answer this question. Just say that you are not able to answer this question and ask if user wants to see the list of available functions.

    You are also given the full history of user's messages in this conversation.
    Use this history to understand the context of the user query, for example, infer an ID or group filtering from the previous user query.
    Use user's query history to understand the question better and guide your responses if needed.

    {conversation_text}
    """
        
        return system_prompt
    
    def _get_heart_system_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Generate the heart system prompt with embedded data and functions."""
        import pandas as pd
        import json
        
        # Load heart dataset and functions
        heart_dataset = pd.read_csv('instances/heart/data/test_set.csv')
        dataset_json = heart_dataset.describe().to_json()
        
        with open('instances/heart/functions.json') as f:
            functions = json.load(f)
        
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

    {functions}
    
    You are an expert in composing function calls. You are given a user query and a set of possible functions that you can call. Based on the user query, you need to decide whether any functions can be called or not.
    You are a trustworthy assistant, which means you should not make up any information or provide any answers that are not supported by the functions given above.
    
    Respond ONLY in JSON format, following strictly this JSON schema for your response:
    
    {response_schema}
    
    Please use double quotes for the keys and values in the JSON response. Do not use single quotes.
   
    If you decide to invoke one or several of the available functions, you MUST include them in the JSON response field "function_calls" in format "[func_name1(params_name1=params_value1, params_name2=params_value2...),func_name1(params_name1=params_value1, params_name2=params_value2...)]".
    When adding param values, only use param values given by user. Do not use any other values or make up any values.
    If you decide that no function(s) can be called, you should return an empty list [] as "function_calls".
      
    Your free-form response in JSON field "freeform_response" is mandatory and it should be a short comment about what you are trying to achieve with chosen function calls. 
    If user asked a question about data/model/prediction and it can not be answered with the available functions, your free-form response should not try to answer this question. Just say that you are not able to answer this question and ask if user wants to see the list of available functions.

    You are also given the full history of user's messages in this conversation.
    Use this history to understand the context of the user query, for example, infer an ID or group filtering from the previous user query.
    Use user's query history to understand the question better and guide your responses if needed.

    {conversation_text}
    """
        
        return system_prompt
    
    def is_usecase_registered(self, usecase: UseCase) -> bool:
        """Check if a use case is registered."""
        return usecase in self._registries
    
    def get_registered_usecases(self) -> List[UseCase]:
        """Get a list of all registered use cases."""
        return list(self._registries.keys())
    
    def clear_all_usecases(self) -> None:
        """Clear all registered use cases."""
        self._registries.clear()
        logger.info("Cleared all use cases")
    
    def _initialize_default_prompts(self) -> None:
        """
        Initialize default system prompts for each use case.
        Note: The actual system prompts are handled by the original functions
        in instances/energy/prompt.py and instances/heart/prompt.py
        """
        # These are placeholder prompts - the real ones are in the original files
        self._system_prompts[UseCase.ENERGY] = "Energy analysis assistant (original prompt in instances/energy/prompt.py)"
        self._system_prompts[UseCase.HEART] = "Heart disease analysis assistant (original prompt in instances/heart/prompt.py)"
    
    def _get_default_prompt(self) -> str:
        """
        Get default system prompt for unknown use cases.
        """
        return "You are a helpful AI assistant."
