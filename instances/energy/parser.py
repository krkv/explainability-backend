from instances.energy.executive import (
    available_functions,
    about_dataset,
    about_dataset_in_depth,
    about_model,
    model_accuracy,
    about_explainer,
    count_all,
    count_group,
    show_ids,
    show_one,
    show_group,
    predict_one,
    predict_group,
    predict_new,
    mistake_one,
    mistake_group,
    explain_one,
    explain_group,
    cfes_one,
    what_if_one,
)
from src.services.parser.function_parser import FunctionParser
from src.core.exceptions import FunctionExecutionException

# Create function registry for energy usecase
_energy_functions = {
    'available_functions': available_functions,
    'about_dataset': about_dataset,
    'about_dataset_in_depth': about_dataset_in_depth,
    'about_model': about_model,
    'model_accuracy': model_accuracy,
    'about_explainer': about_explainer,
    'count_all': count_all,
    'count_group': count_group,
    'show_ids': show_ids,
    'show_one': show_one,
    'show_group': show_group,
    'predict_one': predict_one,
    'predict_group': predict_group,
    'predict_new': predict_new,
    'mistake_one': mistake_one,
    'mistake_group': mistake_group,
    'explain_one': explain_one,
    'explain_group': explain_group,
    'cfes_one': cfes_one,
    'what_if_one': what_if_one,
}

# Create parser instance with energy functions
_parser = FunctionParser(_energy_functions)


def parse_calls(calls):
    """
    Parse and execute function calls safely without using eval().
    
    Args:
        calls: List of function call strings (e.g., ["count_all()", "show_one(id=5)"])
        
    Returns:
        String containing newline-separated results
        
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