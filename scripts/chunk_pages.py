"""
chunk_pages.py -- Split extracted pages into ~512-token chunks.

Takes extracted_pages.jsonl (one record per page) and splits the English
text into smaller chunks at sentence boundaries. Each chunk inherits the
author metadata and carries the FULL Latin page text as metadata.

Usage:
    python scripts/chunk_pages.py
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = Path(__file__).parent.parent / "data" / "chunks" / "extracted_pages.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "chunks" / "chunked_pages.jsonl"

TARGET_TOKENS = 512
OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Simple tokenizer (word-based approximation)
# Good enough for chunking; real token counts vary by model.
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Approximate token count. ~1.3 words per token for English."""
    return max(1, int(len(text.split()) / 0.75))


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Split on sentence-ending punctuation followed by space + uppercase,
# or on paragraph breaks. Handles common abbreviations.
SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z(])'  # period/!/? followed by space + capital
    r'|(?<=\n)\s*\n'             # paragraph break
)

ABBREVS = {"e.g.", "i.e.", "cf.", "viz.", "vs.", "etc.", "no.", "vol.",
           "Ill.", "ill.", "fig.", "Fig.", "pp.", "p.", "C.", "A.D.", "B.C."}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations."""
    # First split on the regex
    raw_parts = SENTENCE_RE.split(text)

    sentences = []
    buffer = ""

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if buffer:
            candidate = buffer
            # Check if the buffer ends with a known abbreviation
            ends_with_abbrev = any(candidate.rstrip().endswith(a) for a in ABBREVS)
            if ends_with_abbrev:
                buffer = buffer + " " + part
                continue
            else:
                sentences.append(buffer.strip())
                buffer = part
        else:
            buffer = part

    if buffer:
        sentences.append(buffer.strip())

    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_sentences(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Group sentences into chunks of ~target tokens with overlap."""
    if not sentences:
        return []

    chunks = []
    current_sents = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = approx_tokens(sent)

        # If adding this sentence would exceed target and we have content,
        # finalize current chunk
        if current_tokens + sent_tokens > target and current_sents:
            chunks.append(" ".join(current_sents))

            # Build overlap from the tail of current sentences
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current_sents):
                t = approx_tokens(s)
                if overlap_count + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_count += t

            current_sents = overlap_sents
            current_tokens = overlap_count

        current_sents.append(sent)
        current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_sents:
        last_chunk = " ".join(current_sents)
        # If it's very short and we have a previous chunk, merge it
        if chunks and approx_tokens(last_chunk) < 100:
            chunks[-1] = chunks[-1] + " " + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks


def chunk_page(record: dict) -> list[dict]:
    """Split a single page record into multiple chunk records."""
    english = record["english"]
    tokens = approx_tokens(english)

    # If already small enough, return as-is
    if tokens <= TARGET_TOKENS + 50:  # small margin
        return [record]

    sentences = split_sentences(english)
    text_chunks = chunk_sentences(sentences, TARGET_TOKENS, OVERLAP_TOKENS)

    results = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_record = {
            "chunkId": f"{record['chunkId']}_{chr(97 + i)}",  # _a, _b, _c...
            "seq": record["seq"],
            "chunk_index": i,
            "chunks_total": len(text_chunks),
            "author_id": record["author_id"],
            "author": record["author"],
            "section": record["section"],
            "english": chunk_text,
            "latin": record["latin"],  # full Latin page
            "pdf_page_en": record["pdf_page_en"],
            "pdf_page_la": record["pdf_page_la"],
        }
        results.append(chunk_record)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input not found at {INPUT_FILE}", file=sys.stderr)
        print("Run extract_pdf.py first.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]

    all_chunks = []
    for page in pages:
        chunks = chunk_page(page)
        all_chunks.extend(chunks)

    # Reassign sequential IDs
    for i, chunk in enumerate(all_chunks):
        chunk["global_seq"] = i + 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Stats
    en_tokens = [approx_tokens(c["english"]) for c in all_chunks]
    en_tokens.sort()

    print(f"Input:  {len(pages)} pages")
    print(f"Output: {len(all_chunks)} chunks -> {OUTPUT_FILE}")
    print(f"\nToken distribution:")
    print(f"  Min:    {en_tokens[0]}")
    print(f"  Max:    {en_tokens[-1]}")
    print(f"  Median: {en_tokens[len(en_tokens)//2]}")
    print(f"  Mean:   {sum(en_tokens)//len(en_tokens)}")

    from collections import Counter
    author_counts = Counter(c["author"] for c in all_chunks)
    print(f"\nChunks per author:")
    for author, count in author_counts.most_common():
        print(f"  {author}: {count}")


if __name__ == "__main__":
    main()
