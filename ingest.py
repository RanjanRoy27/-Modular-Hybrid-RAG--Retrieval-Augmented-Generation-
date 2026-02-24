import os
import sys
import pickle
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import core

# Force UTF-8 for terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"

def load_documents():
    """
    Scans the data/ directory and loads all .txt, .pdf, and .docx files.
    Returns a list of LangChain Document objects.
    """
    all_docs = []
    counts = {"txt": 0, "pdf": 0, "docx": 0, "skipped": 0}

    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}/' directory not found.")
        return []

    files = os.listdir(DATA_DIR)
    if not files:
        print(f"Error: No files found in '{DATA_DIR}/'.")
        return []

    print(f"Found {len(files)} file(s) in '{DATA_DIR}/'. Loading...")

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        ext = filename.lower().split(".")[-1]

        try:
            if ext == "txt":
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                counts["txt"] += 1
            elif ext == "pdf":
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                counts["pdf"] += 1
            elif ext == "docx":
                loader = Docx2txtLoader(filepath)
                docs = loader.load()
                counts["docx"] += 1
            else:
                print(f"  [Skip] '{filename}' — unsupported format.")
                counts["skipped"] += 1
                continue

            for doc in docs:
                doc.metadata["source"] = filename

            all_docs.extend(docs)
            print(f"  [OK]   '{filename}' ({ext.upper()}, {len(docs)} page(s)/section(s))")

        except Exception as e:
            print(f"  [Fail] '{filename}' — {e}")
            counts["skipped"] += 1

    print(f"\nLoaded: {counts['txt']} TXT | {counts['pdf']} PDF | {counts['docx']} DOCX | {counts['skipped']} skipped")
    return all_docs


def ingest_docs():
    """
    Full ingestion pipeline:
    1. Validate environment.
    2. Load all supported documents from data/.
    3. Split into chunks.
    4. Embed and store in FAISS.
    5. Save raw text chunks for BM25 hybrid search.
    """
    if not core.validate_env():
        return {"success": False, "files_loaded": 0, "chunks_created": 0, "error": "Environment validation failed."}

    results = {"success": False, "files_loaded": 0, "chunks_created": 0, "error": None}

    try:
        # 1. Load
        documents = load_documents()
        if not documents:
            results["error"] = "No documents found to ingest."
            print(results["error"])
            return results

        # 2. Split
        print("\nSplitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"  -> {len(texts)} chunks total.")
        results["chunks_created"] = len(texts)

        # 3. Embed & Store (FAISS)
        print("\nGenerating embeddings and saving vector store...")
        embeddings = core.get_embeddings()

        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local("vector_store")
        print("  -> FAISS vector store saved to 'vector_store/'.")

        # 4. Save raw chunks for BM25 (hybrid search)
        chunks_path = os.path.join("vector_store", "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(texts, f)
        print(f"  -> Text chunks saved to '{chunks_path}' (for hybrid search).")

        print("\nDone! Ingestion complete.")
        results["success"] = True
        results["files_loaded"] = len(set(doc.metadata.get("source") for doc in documents))
        return results

    except Exception as e:
        error_msg = f"Error during ingestion: {e}"
        print(error_msg)
        results["error"] = error_msg
        return results


if __name__ == "__main__":
    ingest_docs()
