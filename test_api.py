import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_ingest():
    print("Testing /rag/ingest...")
    response = requests.post(f"{BASE_URL}/rag/ingest")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_answer(question, history=[]):
    print(f"\nTesting /rag/answer for: '{question}'...")
    payload = {
        "question": question,
        "history": history
    }
    start = time.time()
    response = requests.post(f"{BASE_URL}/rag/answer", json=payload)
    end = time.time()
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        res_data = response.json()
        print(f"Answer: {res_data['answer']}")
        print(f"Latency: {res_data['latency_ms']} ms")
        print(f"Context Length: {res_data['context_length']} chars")
        return res_data
    else:
        print(f"Error: {response.text}")
        return None

if __name__ == "__main__":
    # Wait a moment for server
    time.sleep(2)
    
    # 1. Test Ingestion
    if test_ingest():
        # 2. Test Direct Question
        res1 = test_answer("What are the five layers of the RAG architecture?")
        
        if res1:
            # 3. Test Follow-up (Multi-turn)
            history = [
                {"role": "human", "content": "What are the five layers of the RAG architecture?"},
                {"role": "ai", "content": res1["answer"]}
            ]
            test_answer("What does the second one do?", history=history)
