import os
from typing import List
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import core
from reranker import CrossEncoderReranker

@tool
def semantic_search(query: str) -> str:
    """
    Search the company knowledge base for answers to conceptual questions, policies, guidelines, or summaries.
    Returns the most relevant text chunks along with their exact source document and page number.
    """
    embeddings = core.get_embeddings()
    retriever, msg = core.load_retriever(embeddings)
    if not retriever:
        return "Error: Database not initialized."
        
    initial_docs = retriever.invoke(query)
    if not initial_docs:
        return "No relevant information found in the knowledge base."
        
    reranker = CrossEncoderReranker()
    docs = reranker.rerank(query, initial_docs, top_k=5)
    
    # Format the output so the LLM perfectly sees the citations
    formatted_results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "N/A")
        content = doc.page_content.replace("\n", " ")
        formatted_results.append(f"[SOURCE: {source} | PAGE: {page}]\n{content}\n")
        
    return "\n\n".join(formatted_results)

@tool
def document_locator(filename_query: str) -> str:
    """
    Finds exactly which documents exist in the database matching a filename or keyword.
    Use this ONLY when the user asks "Where is document X?" or "Find the Excel file for Y".
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        return "No data directory found."
        
    files = os.listdir(data_dir)
    matches = [f for f in files if filename_query.lower() in f.lower()]
    
    if matches:
        return f"Found the following matching documents: {', '.join(matches)}. They are available in the system."
    else:
        return f"No documents found matching '{filename_query}'."

def build_agent():
    llm = core.get_llm()
    tools = [semantic_search, document_locator]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Smart Assistant with deep knowledge of the company's files. "
         "Use your tools to find accurate information. "
         "For EVERY factual claim or piece of data you return, you MUST provide an inline citation referencing the exact [SOURCE] and [PAGE] provided by the semantic_search tool. "
         "If the user is just looking for a file location, use the document_locator tool. "
         "Return your final answer in strict JSON format exactly like this:\n"
         "{{\"answer\": \"your detailed answer...\", \"citations\": [{{\"document\": \"doc name\", \"page\": \"page_num\"}}]}}"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
