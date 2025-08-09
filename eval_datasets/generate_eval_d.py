from huggingface_hub import InferenceClient
import json
import pandas as pd
import csv
import os

HF_TOKEN = os.getenv('HF_TOKEN')
INSTANCE_PATH = 'instances/energy/'
RESULT_FILE_PATH = 'eval_datasets/eval_dataset_c2.csv'

with open(INSTANCE_PATH + 'functions.json') as f:
    functions = json.load(f)
    
dataset = pd.read_csv(INSTANCE_PATH + 'data/summer_workday_test.csv')

dataset_json = dataset.describe().to_json()

examples_energy = [
    {
        "user_input": "What are the ids of the instances?", 
        "conversation_history": ["Tell me about the dataset"],
        "function_calls": ["show_ids()"]
    },
    {
        "user_input": "Show me sample 34 and its prediction", 
        "conversation_history": ["I want to understand the model", "Is the weather data relevant?"],
        "function_calls": ["show_one(id=34)", "predict_one(id=34)"]
    },
    {
        "user_input": "What is the prediction", 
        "conversation_history": ["Show me the data ids", "Show id 92"],
        "function_calls": ["predict_one(id=92)"]
    },
    {
        "user_input": "How can we change this prediction?", 
        "conversation_history": ["What is the prediction for id 33?", "How correct is it?"],
        "function_calls": ["cfes_one(id=33)"]
    },
    {
        "user_input": "What would it predict if the outdoor temperature was 15?", 
        "conversation_history": ["What kind of explainer is used?", "What is the prediction for id 33?"],
        "function_calls": ["what_if(id=33, outdoor_temperature=15)"]
    },
    {
        "user_input": "Show mistakes for samples with indoor temperature between 27 and 29", 
        "conversation_history": [],
        "function_calls": ["mistake_group(indoor_temperature_min=27, indoor_temperature_max=29)"]
    },
    {
        "user_input": "What do I need to know about the model and explanations?", 
        "conversation_history": [],
        "function_calls": ["about_model()", "about_explainer()"]
    },
    {
        "user_input": "Why they are predicted so?", 
        "conversation_history": ["Show data ids", "Show predictions for data where outdoor is less than 22"],
        "function_calls": ["explain_group(outdoor_temperature_max=22)"]
    },
    {
        "user_input": "Predict indoor 24, outdoor 19, past 6800", 
        "conversation_history": ["Show id 1", "Now prediction", "Explain it"],
        "function_calls": ["predict_new(indoor_temperature=24, outdoor_temperature=19, past_electricity=6800)"]
    },
    {
        "user_input": "How many points have past electricity consumption between 6000 and 8000", 
        "conversation_history": [],
        "function_calls": ["count_group(past_electricity_min=6000, past_electricity_max=8000)"]
    },
    {
        "user_input": "Please give me financial advice", 
        "conversation_history": ["What is this chat?"],
        "function_calls": []
    },
    {
        "user_input": "What can it do?", 
        "conversation_history": ["Is this system reliable?"],
        "function_calls": ["available_functions()"]
    },
    {
        "user_input": "Is this correct?", 
        "conversation_history": ["Show me id 45", "What is the prediction?", "Why?"],
        "function_calls": ["mistake_one(id=45)"]
    },
    {
        "user_input": "Now show me the ones with past electricity below 5000", 
        "conversation_history": ["What are the ids?", "Show sample 450"],
        "function_calls": ["show_group(past_electricity_max=5000)"]
    },
    {
        "user_input": "Is this a typical result?", 
        "conversation_history": ["Predict 100th sample"],
        "function_calls": []
    },
    {
        "user_input": "Predict where it is warmer than 30", 
        "conversation_history": ["Show samples where outdoor is colder than 20"],
        "function_calls": ["predict_group(outdoor_temperature_min=30)"]
    },
    {
        "user_input": "How do you explain stuff and why indoor above 25 are predicted this way", 
        "conversation_history": ["Tell me about the data", "Show the first id"],
        "function_calls": ["about_explainer()", "explain_group(indoor_temperature_min=25)"]
    },
    {
        "user_input": "I don't agree", 
        "conversation_history": ["Is prediction for id 130 correct?",],
        "function_calls": []
    },
    {
        "user_input": "Show more details", 
        "conversation_history": ["Tell about data"],
        "function_calls": ["about_dataset_in_depth()"]
    },
    {
        "user_input": "What is the prediction and how to change it?", 
        "conversation_history": ["Show me the data", "Show id 38"],
        "function_calls": ["predict_one(id=38)", "cfes_one(id=38)"]
    },
                        
]

client = InferenceClient(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
)

def get_system_prompt():
    system_prompt = f"""
        You are a helpful assistant that generates gold dataset samples for evaluation.

        In this use case, a data science assistant helps user to understand the data, model and predictions
        for a machine learning model application use case of energy consumption prediction.

        Here is the description of the use case dataset:

        {dataset_json}

        The functions available in the system are:

        {functions}
        
        Users can ask any questions related to the dataset, model, predictions, and explanations.
        Their questions are not limited to the functions provided, but they can only be answered with these functions.
        Sometimes the assistant may decide to call multiple functions to answer a single user query.
        Sometimes it may be necessary to infer the context from the conversation history to apply correct filtering for the functions.

        The samples that you generate will be used to evaluate the assistant's performance in understanding user queries.
        For each data sample you should generate:
        - a user input that is a question or request related to the dataset, model, or predictions,
        - a conversation history that can be empty or contain a list of previous user queries that are needed to test the context understanding,
        - a list of function calls that the user input should trigger.
        
        You must generate conversation history for most samples, and there should be 1-3 previous questions in conversation history.
        Sometimes conversation history may contain unanswerable queries, queries designed to mislead the assistant, or queries that are not relevant to the current user input.
        But in some cases the current user question must require context from the conversation history to be answered correctly, for example to infer ID or data filtering.

        The samples should be formatted as a JSON object.

        Please use double quotes for the keys and values in the JSON response. Do not use single quotes for this.

        The user input should be a string, the conversation history should be a list of strings, and the function calls should be a list of strings.
        The function calls should be relevant to the user input. If nothing is relevant, the function calls can be an empty list.
        Do not include any additional text or explanations, just return the JSON object.

        Here are some examples of user inputs and their corresponding function calls:

        {examples_energy}

        Generate as many samples as user asks.
        """
        
    with open('eval_prompt.log', 'w') as f:
        f.write(system_prompt)
        
    return system_prompt

user_input = "Generate 25 samples."

def generate_hugging_face_response():
    system_prompt = get_system_prompt()
  
    completion = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=[
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": user_input
                }]
            }
        ],
    )
        
    response = completion.choices[0].message
    return response

if __name__ == "__main__":
    print("Generating response...")
    
    if not os.path.exists(RESULT_FILE_PATH):
        with open(RESULT_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'user_input', 'conversation_history', 'expected_parse'])

    try:
        generated_response = generate_hugging_face_response()
        if generated_response is None or generated_response.content is None:
            raise ValueError("Response text is None")
        response_json = json.loads(generated_response.content)
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")

    with open(RESULT_FILE_PATH, 'a') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
        id = 58
        for sample in response_json:
            writer.writerow([id, sample["user_input"], sample["conversation_history"], sample["function_calls"]])
            id += 1
