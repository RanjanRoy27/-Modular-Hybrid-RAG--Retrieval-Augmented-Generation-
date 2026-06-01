import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
    )
