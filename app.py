import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

gemini_pro = genai.GenerativeModel("gemini-2.0-flash")




app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    response = gemini_pro.generate_content(user_input)
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run()