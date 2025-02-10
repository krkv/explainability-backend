from flask import Flask, request
import json
from dotenv import load_dotenv
from assistant import generate_assistant_response

load_dotenv()

app = Flask(__name__)

@app.route('/getAssistantResponse', methods=['POST'])
def get_response():
    if request.method == "POST":
        app.logger.info("Generating a response...")
        try:
            data = json.loads(request.data)
            conversation = data["conversation"]
            response = generate_assistant_response(conversation)
            return {
                "assistantResponse": response
            }
        except Exception as ext:
            app.logger.error("Failed to generate a response!", ext)
            return("Bad request", 400)