import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
import core

# Force UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SESSION_FILE = "session.json"

def save_session(chat_history):
    """Serializes chat history to JSON."""
    data = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            data.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"role": "ai", "content": msg.content})
    import json
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session():
    """Loads saved chat history from JSON."""
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        import json
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = []
        for item in data:
            if item["role"] == "human":
                history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "ai":
                history.append(AIMessage(content=item["content"]))
        return history
    except Exception:
        return []

def run_rag():
    if not core.validate_env():
        return

    print("Initializing system...")
    embeddings = core.get_embeddings()
    llm = core.get_llm()
    retriever, msg = core.load_retriever(embeddings)
    
    if not retriever:
        print(f"Error: {msg}")
        return
    print(msg)

    rephrase_chain = core.rephrase_prompt_template() | llm
    qa_chain = core.qa_prompt_template() | llm

    chat_history = load_session()
    if chat_history:
        turns = len(chat_history) // 2
        print(f"Session restored: {turns} previous turn(s) loaded. Type 'clear' to start fresh.")
    else:
        print("New session started.")

    print("\n--- Modular RAG Chatbot Ready ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        
        if user_input.lower() in ["exit", "quit"]:
            save_session(chat_history)
            turns = len(chat_history) // 2
            print(f"Session saved ({turns} turn(s)). See you next time!")
            break
        if user_input.lower() == "clear":
            chat_history = []
            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)
            print("Session cleared. Starting fresh.")
            continue
        if not user_input:
            continue
            
        print("Processing...")
        
        try:
            search_query = user_input
            if chat_history:
                rephrase_res = rephrase_chain.invoke({
                    "chat_history": chat_history,
                    "question": user_input
                })
                search_query = core.clean_ai_content(rephrase_res.content)
                print(f"  [Debug] Search Query: {search_query}")

            docs = retriever.invoke(search_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            response = qa_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": user_input
            })
            
            answer = core.clean_ai_content(response.content)
            print(f"\nAI: {answer}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except Exception as e:
            print(f"Error during step processing: {e}")

if __name__ == "__main__":
    run_rag()
