import json
import os
import sys

# Define path to parent logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import core
from api import RAGState

def run_eval():
    print("Initializing RAG System for Evaluation...")
    state = RAGState()
    state.initialize()
    if not state.initialized:
        print("Failed to initialize RAG. Make sure GOOGLE_API_KEY is set in .env")
        return
        
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, "r") as f:
        golden_data = json.load(f)
        
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in golden_data:
        q = item["question"]
        print(f"Evaluating query: {q}")
        
        # 1. Retrieval
        try:
            initial_docs = state.retriever.invoke(q)
            docs = state.reranker.rerank(q, initial_docs, top_k=5)
            context_texts = [doc.page_content for doc in docs]
            context = "\n\n".join(context_texts)
            
            # 2. Generation
            response = state.qa_chain.invoke({
                "context": context, "chat_history": [], "question": q
            })
            ans = core.clean_ai_content(response.content)
            
            questions.append(q)
            answers.append(ans)
            contexts.append(context_texts)
            ground_truths.append(item["ground_truth"])
            
        except Exception as e:
            print(f"Error evaluating query '{q}': {e}")
            continue
        
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print("\nRunning RAGAS evaluation metrics...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision
    ]
    
    # Run evaluation with our core LLM and Embeddings
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=state.llm,
            embeddings=state.embeddings
        )
        print("\n--- Evaluation Results ---")
        print(result)
        
        res_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
        with open(res_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved results to {res_path}")
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")

if __name__ == "__main__":
    run_eval()
