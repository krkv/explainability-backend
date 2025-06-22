import json
from instances.energy.parser import parse_calls as parse_calls_energy
from instances.heart.parser import parse_calls as parse_calls_heart
from huggingface import generate_hugging_face_response
from googlecloud import generate_google_cloud_response

def format_llm_response(response):
    return " ".join(response.split())
    
def generate_assistant_response(conversation, model, usecase):
    print("=== Model ===")
    print(model)
    
    user_input = conversation[len(conversation) - 1]['content']
    print("=== User Input ===")
    print(user_input)
    
    if model == "Llama 3.3 70B Instruct":
        response = generate_hugging_face_response(conversation, usecase)
    elif model == "Gemini 2.0 Flash":
        response = generate_google_cloud_response(conversation, usecase)
    else:   
        raise ValueError("Unsupported model specified.")
        
    print("=== Assistant Response ===")
    print(response)
    
    res = json.loads(response)
    
    if "function_calls" not in res or "freeform_response" not in res:
        raise ValueError("Invalid response format. Expected keys 'function_calls' and 'freeform_response'.")
    function_calls = res["function_calls"]
    freeform_response = res["freeform_response"]
    print("=== Function Calls ===")
    print(function_calls)
    print("=== Freeform Response ===")
    print(freeform_response)
    
    if len(function_calls) > 0:
        if (usecase == "heart"):
             parse = parse_calls_heart(function_calls)
        elif (usecase == "energy"):
            parse = parse_calls_energy(function_calls)
        res["parse"] = parse

    res["freeform_response"] = format_llm_response(freeform_response)
    
    return res