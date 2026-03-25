"""
extract_pdf.py — Extract paired English+Latin text from Campbell 2000
"The Writings of the Roman Land Surveyors"

Outputs a JSONL file with one record per English page, paired with its
facing Latin page, tagged by author and section.

Usage:
    python scripts/extract_pdf.py
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_PATH = Path(__file__).parent.parent / "data" / (
    "Campbell 2000 The Writings of the Roman Land Surveyors_"
    " Introduction, Text, Translation and Commentary.pdf"
)
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "extracted_pages.jsonl"

# ---------------------------------------------------------------------------
# Author / section map
#
# Each entry: (first_english_pdf_page, last_english_pdf_page, author, default_section)
#
# PDF pages are 0-indexed. In Part II the layout alternates:
#   even index → English translation
#   odd  index → Latin text
# So the Latin page for English page N is typically N-1 or N+1.
# From inspection: Latin pages are at ODD indices, English at EVEN indices.
# ---------------------------------------------------------------------------

AUTHOR_SECTIONS = [
    # (start_en, end_en, author_id, author_display)
    (66,  78,  "frontinus",        "Julius Frontinus"),
    (80,  138, "agennius_urbicus", "Agennius Urbicus"),
    (114, 138, "commentum",        "Commentum"),  # overlaps Agennius — see note below
    (140, 164, "hyginus1",         "Hyginus 1"),
    (166, 196, "siculus_flaccus",  "Siculus Flaccus"),
    (198, 226, "hyginus2",         "Hyginus 2"),
    (228, 266, "liber_coloniarum", "Liber Coloniarum"),
    (268, 280, "balbus",           "Balbus"),
    (282, 282, "lex_mamilia",      "Lex Mamilia"),
    (284, 284, "tombs",            "Tombs"),
    (286, 288, "dolabella",        "Dolabella"),
    (290, 292, "latinus",          "Latinus"),
    (294, 302, "casae_litterarum", "Casae Litterarum"),
    (304, 306, "names_surveyors",  "Names of Land Surveyors"),
    (308, 336, "various_authors",  "Various Authors: Boundaries and Lands"),
]

# The Commentum section (pdf 114-138) overlaps with Agennius Urbicus.
# In the book, the Commentum is a commentary appended within the Agennius
# section. We handle this by sorting sections so the more specific match
# (Commentum at 114) overrides the broader Agennius range.


def get_author_for_page(pdf_page: int) -> tuple[str, str]:
    """Return (author_id, author_display) for a given PDF page index."""
    # Check from most specific (latest start) to least specific
    best = None
    for start, end, aid, display in AUTHOR_SECTIONS:
        if start <= pdf_page <= end:
            if best is None or start > best[0]:
                best = (start, end, aid, display)
    if best:
        return best[2], best[3]
    return "unknown", "Unknown"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Running headers that appear at the top of English pages
RUNNING_HEADERS = [
    "TYPES OF LAND", "LAND DISPUTES", "LIMITES", "CATEGORIES OF LAND",
    "THE SCIENCE OF LAND MEASUREMENT", "ESTABLISHMENT OF LIMITES",
    "ESTABLISHMENT OF (LIMITES", "BOOK OF COLONIES", "(BOOK OF COLONIES)",
    "(BOOK OF COLONIES II)", "DESCRIPTION OF FIGURES",
    "BOUNDARY MARKING", "BOUNDARY MARKERS", "ARRANGEMENTS FOR MARKING BOUNDARIES",
    "DEFINITIONS OF LIMITES", "DEFINITIONS OF BOUNDARY STONES",
    "CASAE LITTERARUM", "USAGE OF MEASUREMENTS", "LANDS",
    "(TYPES OF DISPUTE)", "(CATEGORIES OF LAND)", "(COMMENTARY ON TYPES OF LAND)",
    "(LIMITES)", "(THE SCIENCE OF LAND MEASUREMENT)",
]

# Latin page markers — lines that are just manuscript sigla or reference numbers
LATIN_SIGLA_RE = re.compile(
    r"^[\s]*[ABCDEFGHJKLMNPQRSTUVWXYZ()]+[\s]*$"
)
LATIN_REF_RE = re.compile(
    r"^T\s*\d+[\.\d]*\s*=\s*L\s*\d+"
)
PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


def clean_english_text(text: str) -> str:
    """Clean an English translation page."""
    lines = text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines at start
        if not cleaned and not stripped:
            continue

        # Skip page numbers (standalone 1-3 digit numbers)
        if PAGE_NUM_RE.match(stripped):
            continue

        # Skip running headers (first ~3 lines)
        if i < 4 and stripped.upper().rstrip("0123456789 .,'") in [
            h.rstrip("0123456789 .,'") for h in RUNNING_HEADERS
        ]:
            continue

        # Skip lines that are just a number followed by running header
        if i < 4 and stripped and stripped[0].isdigit():
            rest = stripped.lstrip("0123456789 ")
            if rest.upper().rstrip("0123456789 .,'") in [
                h.rstrip("0123456789 .,'") for h in RUNNING_HEADERS
            ]:
                continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Remove footnote markers (superscript numbers) — they appear as isolated digits
    # within text, like "surveying.2 The" → keep as is, these are meaningful
    return result


def clean_latin_text(text: str) -> str:
    """Clean a Latin text page."""
    lines = text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        if not cleaned and not stripped:
            continue

        # Skip page numbers
        if PAGE_NUM_RE.match(stripped):
            continue

        # Skip manuscript sigla lines (e.g. "AP(F)", "B(Gp)", "ABEP")
        if LATIN_SIGLA_RE.match(stripped) and len(stripped) < 15:
            continue

        # Skip reference lines like "T 73.1 = L 112.22"
        if LATIN_REF_RE.match(stripped):
            continue

        # Skip lines that are just section headers in Latin caps
        # (like "DE LIMITIBVS", "DE CONTROVERSIIS AGRORVM")
        # We keep these as they're part of the text structure

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def extract_section_title(text: str) -> str:
    """Try to extract a section title from the first lines of an English page."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:3]:
        # Section titles are typically ALL CAPS or in parentheses
        upper = line.upper()
        if upper == line and len(line) > 3 and not PAGE_NUM_RE.match(line):
            return line.title()
    return ""


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def is_english_page(doc, pdf_page: int) -> bool:
    """Heuristic: English pages have mostly ASCII lowercase prose."""
    text = doc[pdf_page].get_text().strip()
    if len(text) < 50:
        return False

    # Latin pages typically start with a page number or sigla
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    # Check first meaningful line
    first = lines[0]
    if PAGE_NUM_RE.match(first):
        return False  # Latin pages start with their page number
    if LATIN_SIGLA_RE.match(first) and len(first) < 15:
        return False
    if LATIN_REF_RE.match(first):
        return False

    return True


def find_latin_pair(doc, en_page: int) -> int | None:
    """Find the Latin page paired with an English page.

    The layout is: Latin on odd indices, English on even indices (0-based).
    So for English page N, the Latin pair is typically N-1 or N+1.
    """
    # Try the adjacent odd page
    for candidate in [en_page - 1, en_page + 1]:
        if 0 <= candidate < doc.page_count:
            if not is_english_page(doc, candidate):
                text = doc[candidate].get_text().strip()
                if len(text) > 50:
                    return candidate
    return None


def extract_all(pdf_path: Path) -> list[dict]:
    """Extract all English+Latin page pairs from Part II."""
    doc = fitz.open(str(pdf_path))
    records = []
    seq = 0

    # Part II: PDF pages 64–336 (approximately)
    # We scan all pages in this range and pick English ones
    for pdf_page in range(64, 338):
        if pdf_page >= doc.page_count:
            break

        author_id, author_display = get_author_for_page(pdf_page)
        if author_id == "unknown":
            continue

        if not is_english_page(doc, pdf_page):
            continue

        en_text_raw = doc[pdf_page].get_text()
        en_text = clean_english_text(en_text_raw)

        if len(en_text.strip()) < 30:
            continue

        # Find paired Latin page
        la_page = find_latin_pair(doc, pdf_page)
        la_text = ""
        if la_page is not None:
            la_text = clean_latin_text(doc[la_page].get_text())

        section = extract_section_title(en_text_raw)

        seq += 1
        record = {
            "chunkId": f"{author_id}_{seq:03d}",
            "seq": seq,
            "author_id": author_id,
            "author": author_display,
            "section": section,
            "english": en_text,
            "latin": la_text,
            "pdf_page_en": pdf_page,
            "pdf_page_la": la_page,
        }
        records.append(record)

    doc.close()
    return records


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting from: {PDF_PATH.name}")
    records = extract_all(PDF_PATH)

    # Write JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} page pairs -> {OUTPUT_FILE}")

    # Print summary by author
    from collections import Counter
    author_counts = Counter(r["author"] for r in records)
    print("\nPages per author:")
    for author, count in author_counts.most_common():
        print(f"  {author}: {count}")


if __name__ == "__main__":
    main()
