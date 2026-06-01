import os
import sys

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

import core
from core.chunker import split_documents
from domains.accounting.preprocessor import AccountingDomain

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data"


def load_documents(domain=None):
    """
    Scans the data/ directory and loads all .txt, .pdf, .docx, and .xlsx files.
    Returns a list of LangChain Document objects.
    """
    domain = domain or AccountingDomain()
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
                print(f"  [Skip] '{filename}' - unsupported format.")
                counts["skipped"] += 1
                continue

            for doc in docs:
                doc.metadata["source"] = filename
                doc.page_content = domain.preprocess_document(doc.page_content)

            all_docs.extend(docs)
            print(f"  [OK]   '{filename}' ({ext.upper()}, {len(docs)} page(s)/section(s))")

        except Exception as e:
            print(f"  [Fail] '{filename}' - {e}")
            counts["skipped"] += 1

    print(f"\nLoaded: {counts['txt']} TXT | {counts['pdf']} PDF | {counts['docx']} DOCX | {counts['xlsx']} XLSX | {counts['skipped']} skipped")
    return all_docs


def ingest_docs(domain=None):
    """
    Full ingestion pipeline:
    1. Validate environment.
    2. Load all supported documents from data/.
    3. Split into semantic chunks.
    4. Embed and store in Qdrant.
    """
    domain = domain or AccountingDomain()
    config = domain.get_config()
    if not core.validate_env():
        return {"success": False, "files_loaded": 0, "chunks_created": 0, "error": "Environment validation failed."}

    results = {"success": False, "files_loaded": 0, "chunks_created": 0, "error": None}

    try:
        documents = load_documents(domain)
        if not documents:
            results["error"] = "No documents found to ingest."
            print(results["error"])
            return results

        print("\nSplitting text into chunks...")
        embeddings = core.get_embeddings()
        texts = split_documents(
            documents,
            embeddings,
            chunk_size=config.get("chunk_size"),
            chunk_overlap=config.get("chunk_overlap"),
        )
        print(f"  -> {len(texts)} chunks total.")
        results["chunks_created"] = len(texts)

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
