import json
from pydantic import BaseModel
import csv
import ast
import os
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from instances.energy.prompt import get_system_prompt
import pandas as pd

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

with open('eval_dataset_a2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
    next(reader)  # Skip header
    input_dataset = [[row[0], row[1], ast.literal_eval(row[2]), ast.literal_eval(row[3])] for row in reader]

client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    vertexai=True,
    project="explainability-app",
    location="global"
)

def generate_google_cloud_response(user_input, conversation):    
  response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=user_input,
    config=GenerateContentConfig(
      system_instruction= get_system_prompt(conversation),
      response_mime_type='application/json',
      response_schema=Response
    )
  )
  return response.text

EVALUATION_LOG_FILE = 'gemini_2.0_flash_eval.csv'

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
    
    # for i in input_dataset[45:]:
    #     user_input = i[0]
    #     conversation = i[1]
    #     expected_parse = i[2]
        
    #     eval_log = pd.read_csv(EVALUATION_LOG_FILE)
    #     user_inputs = eval_log['user_input'].values

    #     if user_input not in user_inputs:
    #         try:
    #             generated_response = generate_google_cloud_response(user_input, conversation)
    #             generated_parse = json.loads(generated_response)['function_calls']
    #             log_result(user_input, conversation, expected_parse, generated_parse)
    #         except Exception as e:
    #             print(f"Error processing input {user_input}: {e}")
    #             log_result(user_input, conversation, expected_parse, str(e))

    print(calculate_accuracy(EVALUATION_LOG_FILE))