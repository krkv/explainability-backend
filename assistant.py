from parser import parse_calls, is_list_of_calls
from huggingface import generate_hugging_face_response
from googlecloud import generate_google_cloud_response

def format_llm_response(response):
    return response.replace("\n", "<br>")
    
def generate_assistant_response(conversation, model):
    print("=== Model ===")
    print(model)
    
    user_input = conversation[len(conversation) - 1]['content']
    print("=== User Input ===")
    print(user_input)
    
    if model == "Llama 3.3 70B Instruct":
        response = generate_hugging_face_response(conversation)
    elif model == "Gemini 2.0 Flash":
        response = generate_google_cloud_response(conversation)
    else:   
        raise ValueError("Unsupported model specified.")
        
    print("=== Assistant Response ===")
    print(response)
    
    if is_list_of_calls(response):
        return parse_calls(response)
    else:
        return format_llm_response(response)