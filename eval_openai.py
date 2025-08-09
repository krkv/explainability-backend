import os
import csv
import ast
import json
from instances.energy.prompt import get_system_prompt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = 'gpt-5-mini'
EVAL_DATASET_NAME = 'eval_dataset_c'
EVAL_DATASET_PATH = f'eval_datasets/{EVAL_DATASET_NAME}.csv'
RESULT_FILE_PATH = f'eval_results/{EVAL_DATASET_NAME}_{MODEL}.csv'

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

    retry_ids = []

    for i in retry_ids:
        data = eval_dataset[i-1]
        id = data[0]
        user_input = data[1]
        conversation = data[2]
        expected_parse = data[3]

        try:
            client = OpenAI(
                api_key=os.getenv('OPEN_AI_KEY'),
            )
    
            response = client.responses.create(
                model=MODEL,
                instructions=get_system_prompt(conversation),
                input=user_input,
            )

            if response is None:
                print("Generated response is None, skipping")
                continue
            generated_parse = json.loads(response.output_text)['function_calls']
            print(f"Generated parse for ID {id}: {generated_parse}")
            log_result(id, user_input, conversation, expected_parse, generated_parse)
        except Exception as e:
            print(f"Error processing ID {id}: {e}")
            retry_ids.append(id)
            continue
    
    print("Evaluation completed.")
    print(f"Retry IDs: {retry_ids}")
