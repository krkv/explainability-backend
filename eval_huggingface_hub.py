import json
import csv
import ast
import os
from huggingface_hub import InferenceClient
from instances.energy.prompt import get_system_prompt as get_system_prompt_energy
import pandas as pd

MODEL_NAME = 'Llama-3.3-70B-Instruct'
MODEL = f'meta-llama/{MODEL_NAME}'
EVAL_DATASET_NAME = 'eval_dataset_c'
EVAL_DATASET_PATH = f'eval_datasets/{EVAL_DATASET_NAME}.csv'
RESULT_FILE_PATH = f'eval_results/{EVAL_DATASET_NAME}_{MODEL_NAME}.csv'

with open(EVAL_DATASET_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
    next(reader)  # Skip header
    input_dataset = [[row[0], row[1], ast.literal_eval(row[2]), ast.literal_eval(row[3])] for row in reader]

client = InferenceClient(
    provider="auto",
    token=os.getenv('HF_TOKEN'),
)

def generate_hugging_face_response(user_input, conversation):
    system_prompt = get_system_prompt_energy(conversation)
  
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
        
    response = completion.choices[0].message
    return response

def log_result(id, user_input, conversation, expected_parse, generated_parse):
    with open(RESULT_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id, user_input, conversation, expected_parse, generated_parse])

if __name__ == "__main__":
    if not os.path.exists(RESULT_FILE_PATH):
        with open(RESULT_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'user_input', 'conversation_history', 'expected_parse', 'generated_parse'])

    print("Starting evaluation...")
    
    for i in input_dataset:
        id = i[0]
        user_input = i[1]
        conversation = i[2]
        expected_parse = i[3]

        try:
            generated_response = generate_hugging_face_response(user_input, conversation)
            if not generated_response or not generated_response.content:
                raise ValueError("Generated response is None or empty")
            generated_parse = json.loads(generated_response.content)['function_calls']
            log_result(id, user_input, conversation, expected_parse, generated_parse)
        except Exception as e:  
            raise RuntimeError(f"Error processing ID {id}: {e}")
        
    print("Evaluation completed.")