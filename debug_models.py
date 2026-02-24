
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("API Key not found!")
else:
    client = genai.Client(api_key=api_key)
    print("Available Embedding Models:")
    try:
        models = client.models.list()
        print(f"{'Model Name':<40}")
        print("-" * 40)
        for m in models:
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
    except Exception as e:
        print(f"Error listing models: {e}")
