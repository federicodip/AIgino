# AIgino — RAG for the Corpus Agrimensorum Romanorum

A Retrieval-Augmented Generation system for scholarly research on the writings of the Roman land surveyors. Built on the Campbell 2000 edition (*The Writings of the Roman Land Surveyors: Introduction, Text, Translation and Commentary*).

Part of the **Agrimensor-LM** project (UZH / ENS Paris-CNRS / University of Naples Federico II).

## What it does

- Answers scholarly questions about Roman land surveying (*limitatio*, cadastral terminology, surveying procedures, boundary practices)
- Retrieves relevant passages and displays both **English translation** and **Latin original** for scholarly inspection
- Compares perspectives across authors (Frontinus, Hyginus, Siculus Flaccus, Agennius Urbicus, etc.)
- Identifies where textual evidence supports multiple interpretations

## Architecture

```
Campbell 2000 PDF (644 pages)
    → extract_pdf.py     (136 paired English+Latin pages, tagged by author)
    → chunk_pages.py     (259 chunks, ~400-500 tokens, sentence boundaries)
    → ingest.py          (Chroma vector DB, qwen3-embedding:8b)
    → chat.py            (RAG chat with qwen3.5:27b, deduplication, scholarly prompting)
```

**Models:**
- Embedding: `qwen3-embedding:8b` (#1 MTEB multilingual, 40K context)
- Chat: `qwen3.5:27b` (256K context, hybrid thinking mode)
- Eval QA generation: `gemma3:12b` (separate model family to avoid bias)

**Infrastructure:** Runs on UZH ScienceCluster (Apptainer + Ollama + GPU).

## Project structure

```
AIgino/
├── data/
│   ├── Campbell 2000 [...].pdf          # Source (not in repo)
│   └── chunks/
│       ├── extracted_pages.jsonl         # 136 page pairs
│       └── chunked_pages.jsonl           # 259 retrieval-ready chunks
├── scripts/
│   ├── extract_pdf.py                    # PDF → paired English+Latin sections
│   ├── chunk_pages.py                    # Pages → ~512-token chunks
│   ├── ingest.py                         # Chunks → Chroma vector DB
│   ├── chat.py                           # Interactive RAG chat
│   ├── run_eval.py                       # Batch eval (10 demo questions)
│   ├── generate_testset.py               # Synthetic QA generation
│   └── hybrid_eval.py                    # RAGAS + LLM-as-judge evaluation
├── eval/
│   └── demo_questions.jsonl              # 10 curated demo questions
├── jobs/                                 # Slurm scripts for HPC
│   ├── setup.sh                          # Pull models + ingest
│   ├── chat.sh                           # Interactive chat session
│   ├── eval.sh                           # Run demo questions
│   ├── generate_testset.sh               # Generate synthetic QA
│   └── hybrid_eval.sh                    # Full hybrid evaluation
├── .env.example
└── requirements.txt
```

## Quick start (HPC)

```bash
# 1. Clone
git clone https://github.com/federicodip/AIgino.git ~/aigino
cd ~/aigino && mkdir -p logs

# 2. Copy PDF into data/ (not in repo)
cp /path/to/Campbell_2000.pdf ~/aigino/data/

# 3. Copy .env
cp .env.example .env

# 4. Setup: pull embedding model + ingest chunks into Chroma
sbatch jobs/setup.sh

# 5. Interactive chat demo
srun --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB" \
     --partition=lowprio --mem=32G --time=02:00:00 --pty \
     bash jobs/chat.sh
```

## Evaluation pipeline

```bash
# Generate 50 synthetic QA pairs (gemma3:12b)
sbatch jobs/generate_testset.sh

# Run hybrid eval: RAGAS + LLM-as-judge (qwen3.5:27b)
sbatch jobs/hybrid_eval.sh
```

Metrics produced:
- **LLM-as-judge**: factual score (0-3), source hit rate
- **RAGAS**: Faithfulness, Answer Relevancy

## Roadmap

- [ ] Context-Oriented Translation (Iwata et al. 2024) — search-optimized paraphrases for better retrieval
- [ ] Commentary integration (Campbell's Part III) — for challenging established interpretations
- [ ] GraphRAG layer (Neo4j) — concept/place/technique relationships across authors
- [ ] Manuscript visual materials — illustration analysis with vision models
- [ ] Links to Pleiades gazetteer and archaeological datasets

## Authors covered

| Author | Chunks | Description |
|---|---|---|
| Liber Coloniarum | 39 | Colonial land registers |
| Agennius Urbicus | 34 | Land disputes |
| Siculus Flaccus | 32 | Categories of land |
| Hyginus 2 | 29 | Establishment of limites |
| Commentum | 26 | Commentary on types of land |
| Various Authors | 26 | Boundaries and lands |
| Hyginus 1 | 25 | Limites |
| Julius Frontinus | 14 | Types of land, disputes |
| Balbus | 13 | Description of figures |
| Casae Litterarum | 9 | Letter-shaped boundary markers |
| Latinus | 4 | Boundary stones |
| Dolabella | 3 | Works on surveying |
| Tombs | 2 | Tomb boundaries |
| Names of Surveyors | 2 | Surveyor lists |
| Lex Mamilia | 1 | Land law |
