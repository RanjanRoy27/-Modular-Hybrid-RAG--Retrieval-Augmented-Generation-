import os
import sys
import pickle
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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

def load_retriever(embeddings):
    """Loads the hybrid retriever (FAISS + BM25) or falls back to FAISS."""
    if not os.path.exists("vector_store"):
        return None, "Vector store not found. Run 'python ingest.py' first."

    try:
        # Load FAISS vector retriever (semantic search)
        vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Load BM25 retriever (keyword search) from saved chunks
        chunks_path = os.path.join("vector_store", "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 3

            # Hybrid: 50% BM25 (keyword) + 50% FAISS (semantic)
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
            return retriever, "Using HYBRID search (BM25 + FAISS)."
        else:
            return faiss_retriever, "Using SEMANTIC search only (run ingest.py to enable hybrid)."

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
