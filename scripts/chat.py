"""
chat.py -- Simple RAG chat for the Corpus Agrimensorum Romanorum.

Retrieves relevant English passages from Chroma, displays them with
the paired Latin text and author attribution, then generates an answer
using Ollama.

Usage:
    python scripts/chat.py
    python scripts/chat.py --top-k 3
    python scripts/chat.py --model qwen3.5:27b
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:27b")
_chroma = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_DIR = Path(_chroma) if os.path.isabs(_chroma) else Path(__file__).parent.parent / _chroma
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "aigino")
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a specialist in Roman land surveying (agrimensura) and the Corpus \
Agrimensorum Romanorum. You answer scholarly questions based ONLY on the \
provided source passages from the ancient Roman land surveyors' writings \
(Campbell 2000 edition).

Rules:
1. Base your answer strictly on the provided context passages. Cite each claim \
   with the author name and passage number, e.g. (Hyginus 2, [3]).
2. When multiple authors address the same topic, COMPARE their perspectives \
   and note where they agree, diverge, or contradict each other.
3. If the evidence supports multiple interpretations, present them as \
   alternatives with the textual evidence for each. This is critical for \
   scholarly use.
4. Use precise technical terminology (limes, decumanus, kardo, centuria, \
   controversia, subsecivum, etc.) and explain terms on first use.
5. Structure your answer with a brief synthesis first, then detailed evidence.
6. If the passages do not contain enough information, say so explicitly and \
   suggest what additional sources might be relevant.
7. Answer directly. Do not repeat the question or add unnecessary preamble.
"""


def format_context(results) -> str:
    """Format retrieved documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        author = meta.get("author", "Unknown")
        section = meta.get("section", "")
        page_en = meta.get("pdf_page_en", "?")

        header = f"[{i}] {author}"
        if section:
            header += f" - {section}"
        header += f" (p.{page_en})"

        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def display_results(results):
    """Display retrieved passages with English + Latin side by side."""
    print("\n" + "=" * 70)
    print("RETRIEVED PASSAGES")
    print("=" * 70)

    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        author = meta.get("author", "Unknown")
        section = meta.get("section", "")
        page_en = meta.get("pdf_page_en", "?")
        page_la = meta.get("pdf_page_la", "?")
        latin = meta.get("latin", "")

        print(f"\n--- [{i}] {author}", end="")
        if section:
            print(f" | {section}", end="")
        print(f" | EN p.{page_en}, LA p.{page_la} ---")

        # English
        print("\n  ENGLISH:")
        for line in doc.page_content.split("\n")[:12]:  # first ~12 lines
            print(f"    {line}")
        if len(doc.page_content.split("\n")) > 12:
            print("    [...]")

        # Latin
        if latin:
            print("\n  LATIN:")
            latin_lines = latin.split("\n")
            for line in latin_lines[:10]:
                print(f"    {line}")
            if len(latin_lines) > 10:
                print("    [...]")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AIgino RAG Chat")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", type=str, default=CHAT_MODEL)
    parser.add_argument("--no-generate", action="store_true",
                        help="Only retrieve, don't generate an answer")
    args = parser.parse_args()

    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_chroma import Chroma
    from langchain_core.messages import SystemMessage, HumanMessage

    if not CHROMA_DIR.exists():
        print("ERROR: Chroma DB not found. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    llm = None
    if not args.no_generate:
        llm = ChatOllama(
            model=args.model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )

    print(f"AIgino RAG Chat")
    print(f"  Embedding: {EMBEDDING_MODEL}")
    print(f"  Chat model: {args.model}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        # Retrieve (fetch extra, then deduplicate by Latin page)
        raw_results = vectorstore.similarity_search(query, k=args.top_k * 2)

        if not raw_results:
            print("\nNo relevant passages found.\n")
            continue

        # Deduplicate: keep one chunk per (author, pdf_page_la) pair
        seen = set()
        results = []
        for doc in raw_results:
            key = (doc.metadata.get("author_id", ""), doc.metadata.get("pdf_page_la", -1))
            if key not in seen:
                seen.add(key)
                results.append(doc)
            if len(results) >= args.top_k:
                break

        # Display retrieved passages
        display_results(results)

        # Generate answer
        if llm:
            context = format_context(results)
            prompt = (
                f"Based on the following passages from the Roman land surveyors' "
                f"writings, answer this question:\n\n"
                f"Question: {query}\n\n"
                f"Passages:\n{context}"
            )

            print("\nANSWER:")
            print("-" * 70)

            # Stream the response
            for chunk in llm.stream([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]):
                print(chunk.content, end="", flush=True)

            print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
