from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pydantic import BaseModel
from instances.energy.prompt import get_system_prompt as get_system_prompt_energy
from instances.heart.prompt import get_system_prompt as get_system_prompt_heart

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    vertexai=True,
    project="explainability-app",
    location="europe-north1",
)

def generate_google_cloud_response(conversation, usecase):
  if (usecase == "heart"):
    get_system_prompt = get_system_prompt_heart
  elif (usecase == "energy"):
    get_system_prompt = get_system_prompt_energy
    
  system_prompt = get_system_prompt(conversation)
  user_input = conversation[len(conversation) - 1]['content']
  response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=user_input,
    config=GenerateContentConfig(
      system_instruction=system_prompt,
      response_mime_type='application/json',
      response_schema=Response
    )
  )
  return response.text