"""
ingest.py -- Load chunked pages into a persistent Chroma vector store.

Embeds the English text of each chunk using Ollama (mxbai-embed-large by default).
Stores author, section, Latin text, and page references as metadata.

Usage:
    python scripts/ingest.py [--reset]
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
_chroma = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_DIR = Path(_chroma) if os.path.isabs(_chroma) else Path(__file__).parent.parent / _chroma
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "aigino")
_chunks = os.getenv("CHUNKS_FILE", "data/chunks/chunked_pages.jsonl")
CHUNKS_FILE = Path(_chunks) if os.path.isabs(_chunks) else Path(__file__).parent.parent / _chunks


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    reset = "--reset" in sys.argv

    if not CHUNKS_FILE.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_FILE}", file=sys.stderr)
        print("Run extract_pdf.py and chunk_pages.py first.", file=sys.stderr)
        sys.exit(1)

    # Import here so missing deps give a clear error
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    print(f"Embedding model: {EMBEDDING_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"Chroma dir:      {CHROMA_DIR}")
    print(f"Collection:      {CHROMA_COLLECTION}")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # If reset, delete existing collection
    if reset and CHROMA_DIR.exists():
        import shutil
        print("Resetting: deleting existing Chroma DB...")
        shutil.rmtree(CHROMA_DIR)

    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")

    # Build LangChain Documents
    documents = []
    for chunk in chunks:
        # Metadata: everything except the English text (which is page_content)
        metadata = {
            "chunkId": chunk["chunkId"],
            "author_id": chunk["author_id"],
            "author": chunk["author"],
            "section": chunk.get("section", ""),
            "pdf_page_en": chunk.get("pdf_page_en", -1),
            "pdf_page_la": chunk.get("pdf_page_la", -1),
            # Chroma metadata values must be str, int, float, or bool
            # Latin text stored as metadata so we can display it with results
            "latin": chunk.get("latin", ""),
        }
        doc = Document(page_content=chunk["english"], metadata=metadata)
        documents.append(doc)

    # Ingest into Chroma in batches
    BATCH_SIZE = 50
    vectorstore = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Ingesting batch {batch_num}/{total_batches} ({len(batch)} docs)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)

    print(f"\nDone. {len(documents)} chunks ingested into Chroma.")
    print(f"Persistent DB at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
