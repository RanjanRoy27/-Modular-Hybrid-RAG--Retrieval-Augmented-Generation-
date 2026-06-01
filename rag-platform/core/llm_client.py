import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context and the conversation history. If the answer isn't in "
    "the context, say you don't know."
)


def validate_env():
    """Ensures critical environment variables are set."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
        print("Please visit https://aistudio.google.com/ to get your key.")
        return False
    return True


def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        temperature=0,
    )


def clean_ai_content(content):
    """Ensures AI response content is a plain string."""
    if isinstance(content, list):
        return "\n".join(
            [part["text"] for part in content if isinstance(part, dict) and "text" in part]
        )
    return str(content)


def rephrase_prompt_template():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Rephrase the following follow-up question into a standalone search query based on the chat history. Output ONLY the rephrased query.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def qa_prompt_template(system_prompt: str | None = None):
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return ChatPromptTemplate.from_messages([
        ("system", f"{prompt}\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
