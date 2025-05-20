import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

def chat_gemini(user_query):
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.2,
            "top_p": 0.7,
            "max_output_tokens": 4096,
            "response_mime_type": "text/plain"
        }
    )
    
    response = model.generate_content(user_query)
    return response.text

# respond = chat_palmyra('Hi') 
# print(respond)