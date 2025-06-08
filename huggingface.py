import os
from huggingface_hub import InferenceClient
from instances.energy.prompt import get_system_prompt

HF_TOKEN = os.getenv('HF_TOKEN')

client = InferenceClient(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
)

def generate_hugging_face_response(conversation):
  system_prompt = get_system_prompt(conversation)
  user_input = conversation[len(conversation) - 1]['content']
  llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  
  {system_prompt}
  
  <|eot_id|><|start_header_id|>user<|end_header_id|>
  
  {user_input}
  
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
  response = client.text_generation(llama_prompt).strip()
  return response
  
