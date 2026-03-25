"""
generate_testset.py -- Generate synthetic evaluation QA pairs from corpus chunks.

Samples chunks from the corpus, prompts the LLM to generate scholarly questions
and reference answers for each. Produces eval_questions.jsonl for use with
hybrid_eval.py.

Usage:
    python scripts/generate_testset.py
    python scripts/generate_testset.py --num-questions 50
    python scripts/generate_testset.py --model qwen3.5:27b --resume
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:27b")
_chunks = os.getenv("CHUNKS_FILE", "data/chunks/chunked_pages.jsonl")
CHUNKS_FILE = Path(_chunks) if os.path.isabs(_chunks) else Path(__file__).parent.parent / _chunks

OUTPUT_FILE = Path(__file__).parent.parent / "eval" / "eval_questions.jsonl"

# ---------------------------------------------------------------------------
# QA generation prompt
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
/no_think
You are an expert in Roman land surveying and the Corpus Agrimensorum Romanorum. \
Your task is to generate scholarly evaluation questions from passages of the \
ancient Roman land surveyors' writings.
"""

QA_PROMPT_TEMPLATE = """\
Below is a passage from {author}'s writings in the Corpus Agrimensorum Romanorum \
(Campbell 2000 edition).

PASSAGE:
{text}

Generate exactly {num_qa} question-answer pair(s) based on this passage. Each \
question should:
- Require understanding of the passage content to answer (not just keyword matching)
- Be the kind of question a scholar studying Roman land surveying would ask
- Range from factual ("What does X say about Y?") to interpretive ("How does X's \
  description of Y relate to Z?")

For each pair, the answer should be 2-4 sentences, grounded strictly in the passage.

Return your response as a JSON array. Each element must have exactly these fields:
- "question": the scholarly question
- "answer": the reference answer based on the passage
- "category": one of "factual", "terminological", "procedural", "comparative", "interpretive"

Return ONLY the JSON array, no other text. Example format:
[
  {{"question": "...", "answer": "...", "category": "factual"}},
  {{"question": "...", "answer": "...", "category": "procedural"}}
]
"""


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_chunks(chunks: list[dict], num_questions: int) -> list[dict]:
    """Sample chunks ensuring author diversity.

    Strategy: sample proportionally to author representation but ensure
    every author with >2 chunks gets at least one question.
    """
    from collections import defaultdict

    by_author = defaultdict(list)
    for c in chunks:
        # Skip very short chunks
        if len(c["english"]) > 200:
            by_author[c["author_id"]].append(c)

    authors = list(by_author.keys())
    sampled = []

    # Phase 1: one chunk per author (for authors with enough content)
    for author in authors:
        if len(by_author[author]) >= 2:
            sampled.append(random.choice(by_author[author]))

    # Phase 2: fill remaining quota proportionally
    remaining = num_questions - len(sampled)
    if remaining > 0:
        all_eligible = [c for c in chunks if len(c["english"]) > 200]
        # Avoid duplicates
        sampled_ids = {c["chunkId"] for c in sampled}
        pool = [c for c in all_eligible if c["chunkId"] not in sampled_ids]
        random.shuffle(pool)
        sampled.extend(pool[:remaining])

    return sampled[:num_questions]


def generate_qa_for_chunk(llm, chunk: dict, num_qa: int = 2) -> list[dict]:
    """Generate QA pairs from a single chunk using the LLM."""
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = QA_PROMPT_TEMPLATE.format(
        author=chunk["author"],
        text=chunk["english"][:2000],  # cap length
        num_qa=num_qa,
    )

    response = llm.invoke([
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    # Parse JSON from response
    text = response.content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        qa_pairs = json.loads(text)
        if not isinstance(qa_pairs, list):
            qa_pairs = [qa_pairs]
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                qa_pairs = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    # Validate and enrich
    valid = []
    for qa in qa_pairs:
        if "question" in qa and "answer" in qa:
            qa.setdefault("category", "factual")
            qa["source_author_id"] = chunk["author_id"]
            qa["source_author"] = chunk["author"]
            qa["source_chunk_id"] = chunk["chunkId"]
            qa["source_section"] = chunk.get("section", "")
            qa["source_pdf_page"] = chunk.get("pdf_page_en", -1)
            valid.append(qa)

    return valid


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval QA")
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Target number of QA pairs to generate")
    parser.add_argument("--qa-per-chunk", type=int, default=2,
                        help="QA pairs to generate per sampled chunk")
    parser.add_argument("--model", type=str, default=CHAT_MODEL)
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing file, skip already-processed chunks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not CHUNKS_FILE.exists():
        print(f"ERROR: Chunks not found: {CHUNKS_FILE}", file=sys.stderr)
        sys.exit(1)

    from langchain_ollama import ChatOllama

    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")

    # Calculate how many chunks to sample
    chunks_needed = (args.num_questions + args.qa_per_chunk - 1) // args.qa_per_chunk
    sampled = sample_chunks(chunks, chunks_needed)
    print(f"Sampled {len(sampled)} chunks (target: {args.num_questions} QA pairs)")

    # Resume: load existing questions and skip processed chunks
    existing_chunk_ids = set()
    existing_questions = []
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    existing_questions.append(q)
                    existing_chunk_ids.add(q.get("source_chunk_id", ""))
        print(f"Resume: {len(existing_questions)} existing questions, "
              f"skipping {len(existing_chunk_ids)} chunks")
        sampled = [c for c in sampled if c["chunkId"] not in existing_chunk_ids]

    if not sampled:
        print("No new chunks to process.")
        return

    llm = ChatOllama(
        model=args.model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,  # some creativity for diverse questions
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total_generated = len(existing_questions)
    qa_id = total_generated + 1

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for i, chunk in enumerate(sampled, 1):
            print(f"[{i}/{len(sampled)}] {chunk['author']} - {chunk.get('section', '')[:40]}...",
                  end=" ", flush=True)

            t0 = time.time()
            try:
                qa_pairs = generate_qa_for_chunk(llm, chunk, args.qa_per_chunk)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            elapsed = time.time() - t0

            for qa in qa_pairs:
                qa["id"] = qa_id
                qa_id += 1
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                f.flush()

            total_generated += len(qa_pairs)
            print(f"-> {len(qa_pairs)} QA pairs ({elapsed:.1f}s)")

            if total_generated >= args.num_questions + len(existing_questions):
                print(f"Reached target of {args.num_questions} new questions.")
                break

    print(f"\nDone. Total QA pairs: {total_generated} -> {OUTPUT_FILE}")

    # Print category distribution
    all_qs = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_qs.append(json.loads(line))

    from collections import Counter
    cats = Counter(q.get("category", "unknown") for q in all_qs)
    authors = Counter(q.get("source_author", "unknown") for q in all_qs)
    print(f"\nBy category: {dict(cats)}")
    print(f"By author: {dict(authors)}")


if __name__ == "__main__":
    main()
