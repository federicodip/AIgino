"""
run_eval.py -- Batch-run evaluation questions through the RAG pipeline.

Reads questions from eval/demo_questions.jsonl, retrieves passages,
generates answers (with thinking enabled), and saves full results
including retrieved chunks, answer text, and timings.

Supports --resume to skip already-completed questions.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --resume
    python scripts/run_eval.py --top-k 5 --model qwen3.5:27b
"""

import argparse
import json
import os
import sys
import time
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

QUESTIONS_FILE = Path(__file__).parent.parent / "eval" / "demo_questions.jsonl"
RESULTS_FILE = Path(__file__).parent.parent / "eval" / "eval_results.jsonl"

# Same system prompt as chat.py
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


def serialize_result(doc) -> dict:
    """Convert a LangChain Document to a serializable dict."""
    meta = dict(doc.metadata)
    # Latin text can be very long — truncate for readability in results
    if "latin" in meta and len(meta["latin"]) > 500:
        meta["latin_truncated"] = meta["latin"][:500] + "..."
        meta["latin_full_length"] = len(meta["latin"])
        del meta["latin"]
    return {
        "english": doc.page_content,
        "metadata": meta,
    }


def main():
    parser = argparse.ArgumentParser(description="AIgino Batch Evaluation")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", type=str, default=CHAT_MODEL)
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in results file")
    args = parser.parse_args()

    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_chroma import Chroma
    from langchain_core.messages import SystemMessage, HumanMessage

    if not CHROMA_DIR.exists():
        print("ERROR: Chroma DB not found. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)

    if not QUESTIONS_FILE.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_FILE}", file=sys.stderr)
        sys.exit(1)

    # Load questions
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Load existing results for resume
    completed_ids = set()
    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed_ids.add(r["id"])
        print(f"Resuming: {len(completed_ids)} questions already completed")

    remaining = [q for q in questions if q["id"] not in completed_ids]
    if not remaining:
        print("All questions already completed.")
        return

    print(f"Questions to process: {len(remaining)}/{len(questions)}")
    print(f"Model: {args.model} | Embedding: {EMBEDDING_MODEL} | Top-K: {args.top_k}")
    print()

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    llm = ChatOllama(model=args.model, base_url=OLLAMA_BASE_URL, temperature=0.3)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    for i, q in enumerate(remaining, 1):
        qid = q["id"]
        question = q["question"]
        category = q.get("category", "")

        print(f"[{i}/{len(remaining)}] ({category}) {question[:70]}...")

        # --- Retrieve ---
        t0 = time.time()
        raw_results = vectorstore.similarity_search(question, k=args.top_k * 2)

        # Deduplicate by (author, latin page)
        seen = set()
        results = []
        for doc in raw_results:
            key = (doc.metadata.get("author_id", ""), doc.metadata.get("pdf_page_la", -1))
            if key not in seen:
                seen.add(key)
                results.append(doc)
            if len(results) >= args.top_k:
                break

        retrieval_time = time.time() - t0

        # --- Generate ---
        context = format_context(results)
        prompt = (
            f"Based on the following passages from the Roman land surveyors' "
            f"writings, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Passages:\n{context}"
        )

        t1 = time.time()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        generation_time = time.time() - t1

        answer = response.content

        # --- Save ---
        result = {
            "id": qid,
            "category": category,
            "question": question,
            "model": args.model,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": args.top_k,
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(generation_time, 2),
            "num_retrieved": len(results),
            "retrieved_authors": list(set(
                doc.metadata.get("author", "Unknown") for doc in results
            )),
            "retrieved_chunks": [serialize_result(doc) for doc in results],
            "answer": answer,
        }

        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        authors = ", ".join(result["retrieved_authors"])
        print(f"  Retrieved: {len(results)} chunks from [{authors}]")
        print(f"  Timings: retrieval {retrieval_time:.1f}s, generation {generation_time:.1f}s")
        print(f"  Answer preview: {answer[:120]}...")
        print()

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
