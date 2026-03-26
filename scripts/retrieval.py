"""
retrieval.py -- Shared retrieval utilities for AIgino RAG.

Contains:
- Latin glossary for query expansion
- Author-diversity-aware retrieval
- Deduplication logic
"""

import re

# ---------------------------------------------------------------------------
# Latin glossary for query expansion
# Maps Latin technical terms to English definitions.
# When a query contains a Latin term, the definition is appended to
# improve embedding similarity with English chunks.
# ---------------------------------------------------------------------------

LATIN_GLOSSARY = {
    # Land types
    "subsecivum": "land left over from allocation, cut away from the centuriated grid",
    "subseciva": "leftover land parcels cut away from allocated centuriae",
    "arcifinius": "land with uncertain natural boundaries, not formally surveyed",
    "ager occupatorius": "land seized by occupation after driving away the enemy",
    "ager publicus": "public land owned by the Roman state",
    # Survey terms
    "limes": "boundary line or path between allocated land parcels",
    "limites": "boundary lines or paths in a land division grid",
    "decumanus": "east-west boundary line in Roman land division",
    "kardo": "north-south boundary line in Roman land division",
    "centuria": "standard unit of divided land, typically 200 iugera",
    "centuriae": "standard units of divided land in the survey grid",
    "centuriatio": "the system of dividing land into centuriae using a grid",
    "limitatio": "the process of establishing limites for land division",
    "iugera": "Roman unit of land area, approximately 0.25 hectares",
    "actus": "Roman unit of length used in land measurement, 120 Roman feet",
    "rigor": "straight line established by surveying instrument",
    # Survey instruments
    "ferramentum": "surveying instrument used to sight straight lines, likely the groma",
    "groma": "cross-shaped surveying instrument for establishing right angles",
    "gnomon": "vertical rod used to cast shadows for determining cardinal directions",
    "cultellatio": "method of measuring horizontal distance on sloping ground",
    # Land division features
    "quintarius": "every fifth limes, wider than ordinary limites",
    "actuarius": "the first laid-out limes and every fifth one after it",
    "linearius": "ordinary narrow limes between actuarii",
    "striga": "rectangular land plot where length exceeds breadth",
    "scamnum": "rectangular land plot where breadth exceeds length",
    # Dispute types
    "controversia": "legal dispute about land boundaries or ownership",
    "finis": "boundary or endpoint of a property",
    "locus": "site or specific location within land",
    "trifinium": "meeting point of three property boundaries",
    "quadrifinium": "meeting point of four property boundaries",
    # Administrative
    "praefectura": "administrative district attached to a colony",
    "colonia": "Roman colonial settlement with allocated land",
    "loca relicta": "places left unclaimed or unallocated in the survey",
    "forma": "official map or plan of the land division",
    # Boundary markers
    "terminus": "boundary stone or marker",
    "termini": "boundary stones or markers",
    "conportionales": "internal boundary markers within a landholding",
    # Specific classifications
    "ager divisus et assignatus": "land divided by limites and allocated to settlers",
    "mensura per extremitatem comprehensus": "land enclosed by survey along its outer boundary",
}


def expand_query(query: str) -> str:
    """Expand a query by appending English definitions for any Latin terms found."""
    query_lower = query.lower()
    expansions = []

    for latin_term, english_def in LATIN_GLOSSARY.items():
        if latin_term.lower() in query_lower:
            expansions.append(f"{latin_term} ({english_def})")

    if expansions:
        return query + "\n\nTerms: " + "; ".join(expansions)
    return query


def retrieve_diverse(vectorstore, query: str, top_k: int = 8,
                     expand: bool = True) -> list:
    """Retrieve chunks with author diversity and optional query expansion.

    Strategy:
    1. Optionally expand query with Latin glossary definitions
    2. Fetch top_k * 3 raw results
    3. Deduplicate by (author, latin_page)
    4. Author-diversity pass: pick best chunk per author first
    5. Fill remaining slots by score
    """
    search_query = expand_query(query) if expand else query

    raw_docs = vectorstore.similarity_search(search_query, k=top_k * 3)

    # Deduplicate by (author, latin_page)
    seen_pages = set()
    deduped = []
    for doc in raw_docs:
        key = (doc.metadata.get("author_id", ""), doc.metadata.get("pdf_page_la", -1))
        if key not in seen_pages:
            seen_pages.add(key)
            deduped.append(doc)

    # Author-diversity pass: best chunk per author first
    seen_authors = set()
    diverse = []
    remaining = []

    for doc in deduped:
        author_id = doc.metadata.get("author_id", "")
        if author_id not in seen_authors:
            seen_authors.add(author_id)
            diverse.append(doc)
        else:
            remaining.append(doc)

    # Fill remaining slots by original rank order
    docs = diverse[:top_k]
    if len(docs) < top_k:
        for doc in remaining:
            if len(docs) >= top_k:
                break
            docs.append(doc)

    return docs
