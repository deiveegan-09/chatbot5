import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)
MODEL_NAME = "gemini-3.1-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Please keep your message under 4,000 characters."}), 400
    if client is None:
        return jsonify({"error": "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."}), 500

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.5},
        )
        answer = (response.text or "").strip()
        return jsonify({"reply": answer or "I couldn't create a response. Please try again."})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "The fashion assistant is temporarily unavailable. Please try again."}), 502


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
