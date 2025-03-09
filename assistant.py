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
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Your name is Claire. You are a data science assistant that helps user to understand the data, model and predictions for a machine learning model application use case in energy sector.
    You are an expert in composing functions. You are given a user query and a set of possible functions that you can invoke. Based on the user query, you need to decide whether any functions can be called or not.
    
    You have only two options:
    1. If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...);func_name2(params)].
    In this case your response must always be an array! Use ; as separator between function calls. You SHOULD NOT return anything else.
    2. If you decide that no function can be called, you should return a general response to the user in free form, engaging them in a conversation and asking them to formulate the questions in such a way that function calls would be possible.
    In this case you are not allowed to mention the names of the functions or any technical details, because the user does not know it. They are for your reference only!

    Here is a list of functions in JSON format that can be invoked:

    {functions}
    
    You are also given the conversation history between the user and the assistant.
    Use this to understand the context of the user query, for example, infer id or filtering from the previous user queries.
    Do not copy data from your own answers for filtering. Use user queries to understand the query and correct yourself if needed.
    
    {conversation}
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    
def generate_assistant_response(conversation):
        prompt = get_prompt(conversation)
        
        # to save the prompt:
        with open("prompt.txt", "w") as f:
            f.write(prompt)
            
        response = client.text_generation(prompt).strip()
        
        if is_list_of_calls(response):
            print(response)
            return parse_calls(response)
        else:
            print(response)
            return format_llm_response(response)