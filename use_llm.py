import os
import json
from huggingface_hub import InferenceClient
from executive import about_dataset, about_dataset_in_depth, about_model

with open('functions.json') as f:
    functions = json.load(f)

HF_TOKEN = os.getenv('HF_TOKEN')

client = InferenceClient(
    "meta-llama/Llama-3.1-70B-Instruct",
    token=HF_TOKEN,
)

def get_prompt():
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

We are building a natural language interface that helps user to understand the data, model and predictions for a machine learning model application use case in energy sector.

The model and the dataset are not available to you directly, but you have access to a set of functions that can be invoked to help the user.

Here is some information about the model:

{about_model()}

Here is some information about the dataset:

{about_dataset()}

Here is some information about the dataset in depth:

{about_dataset_in_depth()}

Here is a list of functions in JSON format that can be invoked:

{functions}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please generate a set of example user prompts that can be used to test the assistant. The prompts should be in the format of a conversation, where the user asks questions and the assistant responds with function calls or free-form responses. The prompts should cover a variety of topics related to the dataset and model, including data exploration, model evaluation, and prediction generation.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt = get_prompt()

response = client.text_generation(prompt).strip()
        
print("=== Assistant Response ===")
print(response)