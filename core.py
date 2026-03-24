import os
import sys
import pickle
import json
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

def validate_env():
    """Ensures critical environment variables are set."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
        print("Please visit https://aistudio.google.com/ to get your key.")
        return False
    return True

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

q_client = None
def get_qdrant_client():
    global q_client
    if q_client is None:
        q_client = QdrantClient(path="qdrant_store")
    return q_client

def load_retriever(embeddings):
    """Loads the Qdrant retriever."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists("documents"):
            client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="documents",
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        return retriever, "Using QDRANT search."

    except Exception as e:
        return None, f"Error loading vector store: {e}"

def clean_ai_content(content):
    """Ensures AI response content is a plain string."""
    if isinstance(content, list):
        return "\n".join([part['text'] for part in content if isinstance(part, dict) and 'text' in part])
    return str(content)

def rephrase_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", "Rephrase the following follow-up question into a standalone search query based on the chat history. Output ONLY the rephrased query."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

def qa_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question using ONLY the provided context and the conversation history. If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
