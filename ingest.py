import os
import sys
import hashlib
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
import core

# Force UTF-8 for terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"

def load_documents():
    """
    Scans the data/ directory and loads all .txt, .pdf, .docx, and .xlsx files.
    Returns a list of LangChain Document objects.
    """
    all_docs = []
    counts = {"txt": 0, "pdf": 0, "docx": 0, "xlsx": 0, "skipped": 0}

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
            elif ext in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(filepath, mode="elements")
                docs = loader.load()
                counts["xlsx"] += 1
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

    print(f"\nLoaded: {counts['txt']} TXT | {counts['pdf']} PDF | {counts['docx']} DOCX | {counts['xlsx']} XLSX | {counts['skipped']} skipped")
    return all_docs


def ingest_docs():
    """
    Full ingestion pipeline:
    1. Validate environment.
    2. Load all supported documents from data/.
    3. Split into semantic chunks.
    4. Embed and store in Qdrant.
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
        embeddings = core.get_embeddings()
        text_splitter = SemanticChunker(embeddings)
        texts = text_splitter.split_documents(documents)
        
        # Enrich metadata
        ingestion_timestamp = datetime.now().isoformat()
        for i, chunk in enumerate(texts):
            chunk.metadata["source_file"] = chunk.metadata.get("source", "unknown")
            chunk.metadata["page_number"] = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_index"] = i
            chunk.metadata["ingestion_timestamp"] = ingestion_timestamp
            
            # Simple heuristic for section heading (first line)
            first_line = chunk.page_content.strip().split('\n')[0][:50]
            chunk.metadata["section_heading"] = first_line if first_line else "Unknown"
            
            chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            chunk.metadata["content_hash"] = chunk_hash

        print(f"  -> {len(texts)} chunks total.")
        results["chunks_created"] = len(texts)

        # 3. Embed & Store (Qdrant)
        print("\nGenerating embeddings and saving to Qdrant vector store...")
        client = core.get_qdrant_client()
        
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
        vector_store.add_documents(texts)
        print("  -> Qdrant vector store updated.")

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
