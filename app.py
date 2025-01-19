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
            user_input = data["userInput"]
            app.logger.info(f"User input: {user_input}")
            response = generate_assistant_response(user_input)
            app.logger.info(f"Assistant response: {response}")
            return {
                "assistantResponse": response
            }
        except Exception as ext:
            app.logger.error("Failed to generate a response!")
            return("Bad request", 400)