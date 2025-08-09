import os
import csv
import ast
import time
import json
import numpy as np
from pydantic import BaseModel
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from instances.energy.prompt import get_system_prompt

MODEL = 'gemini-2.5-flash'
EVAL_DATASET_NAME = 'eval_dataset_c'
EVAL_DATASET_PATH = f'eval_datasets/{EVAL_DATASET_NAME}.csv'
RESULT_FILE_PATH = f'eval_results/{EVAL_DATASET_NAME}_{MODEL}.csv'

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

with open(EVAL_DATASET_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
    next(reader)  # Skip header
    eval_dataset = [[row[0], row[1], ast.literal_eval(row[2]), ast.literal_eval(row[3])] for row in reader]

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

    retry_ids = np.arange(1, len(eval_dataset) + 1).tolist()

    while len(retry_ids) > 0:
        current_id = retry_ids[0]
        data = eval_dataset[current_id-1]
        id = data[0]
        user_input = data[1]
        conversation = data[2]
        expected_parse = data[3]
        
        try:
            client = genai.Client(
                http_options=HttpOptions(api_version="v1"),
                vertexai=True,
                project="explainability-app",
                location="global"
            )
            response = client.models.generate_content(
                model=MODEL,
                contents=user_input,
                config=GenerateContentConfig(
                    system_instruction= get_system_prompt(conversation),
                    response_mime_type='application/json',
                    response_schema=Response
                )
            )
            generated_response = response.text
            if generated_response is None:
                print("Generated response is None, skipping")
                continue
            generated_parse = json.loads(generated_response)['function_calls']
            log_result(id, user_input, conversation, expected_parse, generated_parse)
            print(f"Processed ID: {id}")
            retry_ids.remove(current_id)
            print("Retry IDs remaining:", retry_ids)
        except Exception as e:
            print(f"Error processing ID {id}: {e}")
            time.sleep(10) # To avoid rate limiting
            continue