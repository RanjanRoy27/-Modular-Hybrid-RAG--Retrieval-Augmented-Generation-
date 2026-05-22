import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_llm():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        res = llm.invoke("Say hello")
        print(f"Response: {res.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()
