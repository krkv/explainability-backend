import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv('HF_TOKEN')

client = InferenceClient(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
)

def_predict = """{
    "name": "predict",
    "description": "Returns a prediction for given data points. It runs an ML prediction function with the given samples to obtain a prediction and return it to the user.",
    "parameters": {
        "type": "dict",
        "properties":
            "id": {
                "type": "integer",
                "description": "The id of the single data sample requested."
            },
            "indoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of indoor temperature."
            },
            "indoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "outdoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of outdoor temperature."
            },
            "outdoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "past_electricity_min": {
                "type": "float",
                "description": "The minimum value of past electricity."
            },
            "past_electricity_max": {
                "type": "float",
                "description": "The maximum value of past electricity."
            }
        }
    }
}"""

def_show = """{
    "name": "show",
    "description": "Show the original data features. Presents the original data points only. Can de used to display a data sample from the dataset.",
    "parameters": {
        "type": "dict",
        "properties":
            "id": {
                "type": "integer",
                "description": "The id of the single data sample requested."
            },
            "indoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of indoor temperature."
            },
            "indoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "outdoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of outdoor temperature."
            },
            "outdoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "past_electricity_min": {
                "type": "float",
                "description": "The minimum value of past electricity."
            },
            "past_electricity_max": {
                "type": "float",
                "description": "The maximum value of past electricity."
            }
        }
    }
}"""

def_explain = """{
    "name": "explain",
    "description": "Generate an explanation of the prediction for given data samples. It runs explainability tools and explains why a prediction was given.",
    "parameters": {
        "type": "dict",
        "properties": {
            "properties":
            "id": {
                "type": "integer",
                "description": "The id of the single data sample requested."
            },
            "indoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of indoor temperature."
            },
            "indoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "outdoor_temperature_min": {
                "type": "float",
                "description": "The minimum value of outdoor temperature."
            },
            "outdoor_temperature_max": {
                "type": "float",
                "description": "The maximum value of outdoor temperature."
            },
            "past_electricity_min": {
                "type": "float",
                "description": "The minimum value of past electricity."
            },
            "past_electricity_max": {
                "type": "float",
                "description": "The maximum value of past electricity."
            }
        }
    }
}"""

def get_prompt(user_input):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert in composing functions. You are given a user question and a set of possible functions that you can invoke.
    Based on the user query, you will need to make one or more function calls to achieve the purpose.
    Also, you need to decide how the underlying dataset should be filtered based on user query, and provide correct parameters to the function to ensure filtering.
    You should only return the function call in tools call sections.

    Ignore the questions that are not related to the dataset, features, predictions, model etc, and just return [] then.
    If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
    You SHOULD NOT include any other text in the response.

    Here is a list of functions in JSON format that you can invoke.

    [{def_show},{def_predict},{def_explain}]<|eot_id|><|start_header_id|>user<|end_header_id|>

    The user query is: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    
def generate_assistant_response(user_input):
    prompt = get_prompt(user_input)
    return client.text_generation(prompt).strip()