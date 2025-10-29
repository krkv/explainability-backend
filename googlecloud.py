import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from instances.energy.prompt import get_system_prompt as get_system_prompt_energy
from instances.heart.prompt import get_system_prompt as get_system_prompt_heart

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

def generate_google_cloud_response(conversation, usecase):
  client = genai.Client(
    http_options=types.HttpOptions(api_version="v1"),
    vertexai=True,
    project="explainability-app",
    location="europe-north1",
  )

  model="gemini-2.0-flash-001"

  if (usecase == "heart"):
    get_system_prompt = get_system_prompt_heart
  elif (usecase == "energy"):
    get_system_prompt = get_system_prompt_energy

  system_prompt = get_system_prompt(conversation)

  generate_content_config = types.GenerateContentConfig(
    system_instruction = system_prompt,
    response_mime_type = 'application/json',
    response_schema = Response,
  )
  user_input = conversation[len(conversation) - 1]['content']
  response = client.models.generate_content(
    model=model,
    contents=user_input,
    config=generate_content_config
  )
  return response.text