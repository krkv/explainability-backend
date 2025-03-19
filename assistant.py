import os
import json
from huggingface_hub import InferenceClient
from parser import parse_calls, is_list_of_calls

with open('functions.json') as f:
    functions = json.load(f)

HF_TOKEN = os.getenv('HF_TOKEN')

client = InferenceClient(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
)
    
def format_llm_response(response):
    return response.replace("\n", "<br>")

def get_prompt(conversation):
    user_input = conversation[len(conversation) - 1]['content']
    
    print("=== User Input ===")
    print(user_input)
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your name is Claire. You are a data science assistant that helps user to understand the data, model and predictions for a machine learning model application use case in energy sector.
The features in the dataset are: outdoor temperature (outdoor_temperature), indoor temperature (indoor_temperature), past electricity consumption (past_electricity).

The model and the dataset are not available to you directly, but you have access to a set of functions that can be invoked to help the user.
You are an expert in composing functions. You are given a user query and a set of possible functions that you can invoke. Based on the user query, you need to decide whether any functions can be called or not.

Here is a list of functions in JSON format that can be invoked:

{functions}

You have only two options:
1. If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...);func_name2(params)].
In this case your response must always be an array! If you return one function call, put it into an array! Use ; as separator between function calls. In this case you SHOULD NOT return anything else.
2. If you decide that no function can be called, you should return a general response to the user in free form, engaging them in a conversation and asking them to formulate the questions in such a way that function calls would be possible.
In this case you are not allowed to mention the names of the functions or any technical details, because the user does not know them. They are for your reference only! Also, your free-form response should be short and concise, not more than 3-4 sentences.

You are also given the full history of user's messages in this conversation.
Use this history to understand the context of the user query, for example, infer an ID or group filtering from the previous user query.
Use user's query history to understand the question better and guide your responses if needed.

{conversation}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    
def generate_assistant_response(conversation):
        prompt = get_prompt(conversation)
        
        # to log last prompt:
        with open("last_prompt.log", "w") as f:
            f.write(prompt)
            
        response = client.text_generation(prompt).strip()
        
        print("=== Assistant Response ===")
        print(response)
        
        if is_list_of_calls(response):
            return parse_calls(response)
        else:
            return format_llm_response(response)