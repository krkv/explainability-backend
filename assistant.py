import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv('HF_TOKEN')

client = InferenceClient(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
)

def_predict = """{
    "type": "function",
    "function": {
        "name": "predict",
        "description": "Get a prediction for given data point(s). It calls an ML prediction function on given samples to obtain a prediction and return it to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The id of the single data sample requested."
                },
                "indoor_temperature_min": {
                    "type": "float",
                    "description": "Filter by minimum value of indoor temperature."
                },
                "indoor_temperature_max": {
                    "type": "float",
                    "description": "Filter by maximum value of outdoor temperature."
                },
                "outdoor_temperature_min": {
                    "type": "float",
                    "description": "Filter by minimum value of outdoor temperature."
                },
                "outdoor_temperature_max": {
                    "type": "float",
                    "description": "Filter by maximum value of outdoor temperature."
                },
                "past_electricity_min": {
                    "type": "float",
                    "description": "Filter by minimum value of past electricity."
                },
                "past_electricity_max": {
                    "type": "float",
                    "description": "Filter by maximum value of past electricity."
                }
            }
        }
    }
}"""

def_show = """{
    "type": "function",
    "function": {
        "name": "show",
        "description": "Show the original data features. Presents the original data points only. Can de used to display a data sample from the dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The id of the single data sample requested."
                },
                "indoor_temperature_min": {
                    "type": "float",
                    "description": "Filter by minimum value of indoor temperature."
                },
                "indoor_temperature_max": {
                    "type": "float",
                    "description": "Filter by maximum value of outdoor temperature."
                },
                "outdoor_temperature_min": {
                    "type": "float",
                    "description": "Filter by minimum value of outdoor temperature."
                },
                "outdoor_temperature_max": {
                    "type": "float",
                    "description": "Filter by maximum value of outdoor temperature."
                },
                "past_electricity_min": {
                    "type": "float",
                    "description": "Filter by minimum value of past electricity."
                },
                "past_electricity_max": {
                    "type": "float",
                    "description": "Filter by maximum value of past electricity."
                }
            }
        }
    }
}"""

def_explain = """{
    "type": "function",
    "function": {
        "name": "explain",
        "description": "Generate an explanation of the prediction for given data sample(s). It runs explainability tools and explains why a prediction was given.",
        "parameters": {
            "type": "object",
            "properties": {
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
    }
}"""

def get_prompt_caller(conversation):
    user_input = conversation[0]
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert in composing functions. You are given a user query and a set of possible functions that you can invoke.
    Based on the user query, you will need to make one or more function calls to achieve the purpose.
    Typically, the functions can be called for a specific data sample by specifying id.
    Alternatively, it is possible to filter the data based on the features (outdoor temperature, indoor temperature, past electricity) and call the functions.
    Do not mix id and filtering in the same function call. You should only use one of them.
    You need to decide how the underlying dataset should be filtered based on user query, and provide correct parameters to the function to ensure filtering.
    You should only return the function call in tools call sections.

    If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
    You SHOULD NOT include any other text in the response.

    Here is a list of functions in JSON format that you can invoke:

    [{def_show},{def_predict},{def_explain}]<|eot_id|><|start_header_id|>user<|end_header_id|>

    The user query is: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
def get_prompt_doorman(conversation):
    user_input = conversation[0]
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a central logical unit of a chat interaction system. You are helping another assistant, who is an expert in composing functions. You need to help another assistant understand
    whether any function call is intended by the user query or not. You are given a user query and a set of possible functions that can be invoked.
    Typically, the functions can be called for a specific data sample by specifying id. Alternatively, it is possible to filter the data based on the features (outdoor temperature, indoor temperature, past electricity) and call the functions.
    Based on the user query, you need to decide whether any functions can be called or not. Return "CALL" if any functions need to be called. Otherwise, return "IGNORE".
    You can only return "CALL" or "CLARIFY" in the response, nothing else!

    Here is a list of functions in JSON format that can be invoked:

    [{def_show},{def_predict},{def_explain}]<|eot_id|><|start_header_id|>user<|end_header_id|>

    The user query is: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
def get_prompt_claire(conversation):
    user_input = conversation[0]
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Your name is Claire. You are a data science assistant that helps user to understand the data, model and predictions for a business use case in energy sector.
    You are helping another assistant, who is an expert in composing functions. It has been determined that the following user query does not result any available function calls.
    Your task is to respond to the user in free form, engaging then in a conversation and asking them to formulate the questions in such way that function calls would be possible.
    For example, if user query is too general to call a function, ask them to narrow down the query by providing a specific id or filtering.
    Typically, the functions can be called for a specific data sample by specifying id. Alternatively, it is possible to filter the data based on the features (outdoor temperature, indoor temperature, past electricity) and call the functions.
    You can't mention the name of the functions or any technical details, they are for your reference only! You can only ask the user to provide more information or clarify his question in free form.

    Here is a list of functions in JSON format that can be invoked:

    [{def_show},{def_predict},{def_explain}]<|eot_id|><|start_header_id|>user<|end_header_id|>

    The user query is: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    
def generate_assistant_response(user_input):
    prompt_doorman = get_prompt_doorman(user_input)
    response_doorman = client.text_generation(prompt_doorman).strip()
    print(response_doorman)
    if (response_doorman == "CALL"):
        prompt_caller = get_prompt_caller(user_input)
        response_caller = client.text_generation(prompt_caller).strip()
        print(response_caller)
        if (response_caller == "[]"):
            prompt_claire = get_prompt_claire(user_input)
            response_claire = client.text_generation(prompt_claire).strip()
            return response_claire
        return response_caller
    else:
        prompt_claire = get_prompt_claire(user_input)
        response_claire = client.text_generation(prompt_claire).strip()
        return response_claire
    