from flask import Flask, request
import json
import os
from dotenv import load_dotenv
from assistant import generate_assistant_response

app = Flask(__name__)

if os.getenv("FLASK_ENV") == "development":
    app.logger.info("Loading the local env...")
    load_dotenv()
    
@app.route('/ready', methods=['GET'])
def ready():
    return("OK", 200)

@app.route('/getAssistantResponse', methods=['POST'])
def get_response():
    if request.method == "POST":
        app.logger.info("Generating a response...")
        try:
            data = json.loads(request.data)
            conversation = data["conversation"]
            model = data["model"]
            usecase = data["usecase"]
            if (usecase == "Heart Disease"):
                uc = "heart"
            elif (usecase == "Energy Consumption"):
                uc = "energy"
            response = generate_assistant_response(conversation, model, uc)
            return {
                "assistantResponse": response
            }
        except Exception as ext:
            app.logger.error(f"Failed to generate a response! {ext}")
            return("Bad request", 400)