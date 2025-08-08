import json
from pydantic import BaseModel
import csv
import ast
import os
from huggingface_hub import InferenceClient
from instances.energy.prompt import get_system_prompt as get_system_prompt_energy
import pandas as pd

HF_TOKEN = os.getenv('HF_TOKEN')

with open('eval_dataset_a1.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
    next(reader)  # Skip header
    input_dataset = [[row[0], ast.literal_eval(row[1]), ast.literal_eval(row[2])] for row in reader]

client = InferenceClient(
    provider="auto",
    token=HF_TOKEN,
)

def generate_hugging_face_response(user_input, conversation):
    system_prompt = get_system_prompt_energy(conversation)
  
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
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

EVALUATION_LOG_FILE = 'llama_3.3_70B_Instruct_eval.csv'

def log_result(user_input, conversation, expected_parse, generated_parse):
    with open(EVALUATION_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, conversation, expected_parse, generated_parse])
        
def calculate_accuracy(log_file):
    eval_log = pd.read_csv(log_file)
    eval_log_size = len(eval_log)
    eval_log = eval_log.drop_duplicates()
    correct_parses = 0
    correct_percent = 0
    expected_parses = eval_log['expected_parse'].values
    generated_parses = eval_log['generated_parse'].values
    log_size = len(expected_parses)
    for i in range(log_size):
        if expected_parses[i] == generated_parses[i]:
            correct_parses += 1
        else:
            print("Expected: " + expected_parses[i], "- Generated: " + generated_parses[i])
    print()
   
    if correct_parses > 0:
        correct_percent = round((correct_parses / eval_log_size) * 100, 2)
        
    return str(correct_percent)

if __name__ == "__main__":
    # if not os.path.exists(EVALUATION_LOG_FILE):
    #     with open(EVALUATION_LOG_FILE, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['user_input', 'conversation_history', 'expected_parse', 'generated_parse'])

    # print("Starting evaluation...")
    
    # for i in input_dataset[0:3]:
    #     user_input = i[0]
    #     conversation = i[1]
    #     expected_parse = i[2]

    #     try:
    #         generated_response = generate_hugging_face_response(user_input, conversation)
    #         generated_parse = json.loads(generated_response.content)['function_calls']
    #         log_result(user_input, conversation, expected_parse, generated_parse)
    #     except Exception as e:
    #         print(f"Error processing input {user_input}: {e}")
    #         log_result(user_input, conversation, expected_parse, str(e))

    print(calculate_accuracy(EVALUATION_LOG_FILE))