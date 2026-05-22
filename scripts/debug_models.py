
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("API Key not found!")
else:
    genai.configure(api_key=api_key)
    print("Available Embedding Models:")
    try:
        models = genai.list_models()
        print(f"{'Model Name':<40}")
        print("-" * 40)
        for m in models:
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
