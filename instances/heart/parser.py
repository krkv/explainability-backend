from instances.heart.executive import (
    get_model_parameters,
    get_model_description,
    predict,
    feature_importance_patient,
    feature_importance_global,
    dataset_summary,
    performance_metrics,
    confusion_matrix_stats,
    what_if,
    counterfactual,
    misclassified_cases,
    age_group_performance,
    feature_interactions,
)
from src.services.parser.function_parser import FunctionParser
from src.core.exceptions import FunctionExecutionException

# Create function registry for heart usecase
_heart_functions = {
    'get_model_parameters': get_model_parameters,
    'get_model_description': get_model_description,
    'predict': predict,
    'feature_importance_patient': feature_importance_patient,
    'feature_importance_global': feature_importance_global,
    'dataset_summary': dataset_summary,
    'performance_metrics': performance_metrics,
    'confusion_matrix_stats': confusion_matrix_stats,
    'what_if': what_if,
    'counterfactual': counterfactual,
    'misclassified_cases': misclassified_cases,
    'age_group_performance': age_group_performance,
    'feature_interactions': feature_interactions,
}

# Create parser instance with heart functions
_parser = FunctionParser(_heart_functions)


def parse_calls(calls):
    """
    Parse and execute function calls safely without using eval().
    
    Args:
        calls: List of function call strings (e.g., ["predict(patient_id=5)", "performance_metrics()"])
        
    Returns:
        String containing newline-separated results
        Note: Heart functions return dicts with "text" key, which are extracted automatically
        
    Raises:
        FunctionExecutionException: If parsing or execution fails
    """
    if len(calls) == 0:
        raise FunctionExecutionException("No function calls provided")
    
    try:
        return _parser.parse_calls(calls)
    except FunctionExecutionException:
        # Re-raise as-is
        raise
    except Exception as e:
        raise FunctionExecutionException(f"Error parsing calls: {e}")