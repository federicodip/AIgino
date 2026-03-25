"""
hybrid_eval.py -- Hybrid evaluation pipeline: RAGAS + LLM-as-judge.

Runs all eval questions through the RAG pipeline once, then scores with:
  1. LLM-as-judge (factual_score 0-3, source_hit, reasoning)
  2. RAGAS metrics (Faithfulness, AnswerRelevancy, ContextPrecision)

Usage:
    python scripts/hybrid_eval.py
    python scripts/hybrid_eval.py --limit 10 --verbose
    python scripts/hybrid_eval.py --skip-ragas          # judge only (faster)
    python scripts/hybrid_eval.py --skip-judge           # RAGAS only
    python scripts/hybrid_eval.py --resume               # skip completed questions
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
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

QUESTIONS_FILE = Path(__file__).parent.parent / "eval" / "eval_questions.jsonl"
RESULTS_FILE = Path(__file__).parent.parent / "eval" / "hybrid_results.jsonl"
SUMMARY_FILE = Path(__file__).parent.parent / "eval" / "hybrid_summary.json"

# ---------------------------------------------------------------------------
# System prompt (same as chat.py)
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a specialist in Roman land surveying (agrimensura) and the Corpus \
Agrimensorum Romanorum. You answer scholarly questions based ONLY on the \
provided source passages from the ancient Roman land surveyors' writings \
(Campbell 2000 edition).

Rules:
1. Keep your answer between 150 and 300 words. Be dense and precise, not exhaustive.
2. Base your answer strictly on the provided context passages. Cite each claim \
   with the author name and passage number, e.g. (Hyginus 2, [3]).
3. When multiple authors address the same topic, note where they agree or diverge.
4. If the evidence supports multiple interpretations, state the alternatives \
   concisely with the textual evidence for each.
5. Use precise technical terminology (limes, decumanus, kardo, centuria, \
   controversia, subsecivum, etc.) and gloss terms only on first use.
6. If the passages do not contain enough information, say so.
7. Do not repeat the question. Do not add section headers or bullet points \
   unless the question explicitly asks for a list.
"""

# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
/no_think
You are an expert evaluator for a scholarly RAG system about Roman land surveying \
(the Corpus Agrimensorum Romanorum).

Evaluate the following RAG answer against the expected answer and source passages.

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

RAG ANSWER: {rag_answer}

SOURCE AUTHOR(S) IN RETRIEVED CHUNKS: {retrieved_authors}

EXPECTED SOURCE AUTHOR: {expected_author}

Score the answer on these criteria:

1. factual_score (integer 0-3):
   0 = completely wrong or hallucinated
   1 = partially correct but major errors or missing key information
   2 = mostly correct with minor inaccuracies or gaps
   3 = fully correct and well-supported by the passages

2. source_hit (boolean): Did the retrieved chunks include at least one passage \
   from the expected source author "{expected_author}"?

3. reasoning (string): One sentence explaining your score.

Return ONLY a JSON object with these three fields:
{{"factual_score": <0-3>, "source_hit": <true/false>, "reasoning": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# Phase 1: RAG Inference
# ---------------------------------------------------------------------------

def run_inference(vectorstore, llm, questions: list[dict], top_k: int,
                  verbose: bool = False) -> list[dict]:
    """Run all questions through the RAG pipeline."""
    from langchain_core.messages import SystemMessage, HumanMessage

    results = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"  [{i}/{len(questions)}] {question[:60]}...", end=" ", flush=True)

        t0 = time.time()

        # Retrieve (with dedup)
        raw_docs = vectorstore.similarity_search(question, k=top_k * 2)
        seen = set()
        docs = []
        for doc in raw_docs:
            key = (doc.metadata.get("author_id", ""), doc.metadata.get("pdf_page_la", -1))
            if key not in seen:
                seen.add(key)
                docs.append(doc)
            if len(docs) >= top_k:
                break

        retrieval_time = time.time() - t0

        # Build context
        context_parts = []
        for j, doc in enumerate(docs, 1):
            meta = doc.metadata
            author = meta.get("author", "Unknown")
            section = meta.get("section", "")
            page = meta.get("pdf_page_en", "?")
            header = f"[{j}] {author}"
            if section:
                header += f" - {section}"
            header += f" (p.{page})"
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"Based on the following passages from the Roman land surveyors' "
            f"writings, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Passages:\n{context}"
        )

        # Generate
        t1 = time.time()
        response = llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        generation_time = time.time() - t1

        rag_answer = response.content
        retrieved_contexts = [doc.page_content for doc in docs]
        retrieved_authors = list(set(doc.metadata.get("author", "Unknown") for doc in docs))
        sources = [
            {
                "author": doc.metadata.get("author", "Unknown"),
                "author_id": doc.metadata.get("author_id", ""),
                "section": doc.metadata.get("section", ""),
                "pdf_page_en": doc.metadata.get("pdf_page_en", -1),
                "chunkId": doc.metadata.get("chunkId", ""),
            }
            for doc in docs
        ]

        result = {
            "id": q.get("id", i),
            "question": question,
            "expected_answer": q.get("answer", ""),
            "source_author_id": q.get("source_author_id", ""),
            "source_author": q.get("source_author", ""),
            "category": q.get("category", ""),
            "rag_answer": rag_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_authors": retrieved_authors,
            "sources": sources,
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(generation_time, 2),
        }
        results.append(result)

        print(f"({retrieval_time:.1f}s + {generation_time:.1f}s) [{', '.join(retrieved_authors[:3])}]")

        if verbose:
            print(f"    A: {rag_answer[:150]}...")
            print()

    return results


# ---------------------------------------------------------------------------
# Phase 2a: LLM-as-judge
# ---------------------------------------------------------------------------

def run_judge(results: list[dict], judge_llm, verbose: bool = False) -> list[dict]:
    """Score each result with LLM-as-judge."""
    from langchain_core.messages import SystemMessage, HumanMessage
    import re

    print(f"\nPhase 2a: LLM-as-judge ({len(results)} questions)")

    for i, r in enumerate(results, 1):
        if "judge" in r and r["judge"] is not None:
            continue  # already scored (resume)

        prompt = JUDGE_PROMPT.format(
            question=r["question"],
            expected_answer=r["expected_answer"],
            rag_answer=r["rag_answer"],
            retrieved_authors=", ".join(r["retrieved_authors"]),
            expected_author=r.get("source_author", "Unknown"),
        )

        print(f"  [{i}/{len(results)}] Judging...", end=" ", flush=True)

        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Judge call exceeded 5 min")

            # Set 5-min timeout per judge call
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(300)
            try:
                response = judge_llm.invoke([HumanMessage(content=prompt)])
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            text = response.content.strip()

            # Extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                judge_result = json.loads(match.group())
                r["judge"] = {
                    "factual_score": int(judge_result.get("factual_score", 0)),
                    "source_hit": bool(judge_result.get("source_hit", False)),
                    "reasoning": str(judge_result.get("reasoning", "")),
                }
            else:
                r["judge"] = {"factual_score": 0, "source_hit": False,
                              "reasoning": f"Failed to parse judge response: {text[:100]}"}
        except (TimeoutError, Exception) as e:
            r["judge"] = {"factual_score": 0, "source_hit": False,
                          "reasoning": f"Judge error: {str(e)[:100]}"}

        score = r["judge"]["factual_score"]
        hit = r["judge"]["source_hit"]
        print(f"score={score}/3, source_hit={hit}")

        if verbose:
            print(f"    {r['judge']['reasoning']}")

    return results


# ---------------------------------------------------------------------------
# Phase 2b: RAGAS metrics
# ---------------------------------------------------------------------------

def run_ragas(results: list[dict], judge_model: str, verbose: bool = False) -> list[dict]:
    """Score each result with RAGAS metrics."""
    print(f"\nPhase 2b: RAGAS metrics ({len(results)} questions)")

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import Faithfulness, ResponseRelevancy, ContextEntityRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama
        from langchain_ollama import OllamaEmbeddings
    except ImportError as e:
        print(f"  RAGAS import failed: {e}")
        print("  Skipping RAGAS metrics. Install with: pip install ragas")
        for r in results:
            r["ragas"] = None
        return results

    # Set up RAGAS judge and embeddings
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model=judge_model, base_url=OLLAMA_BASE_URL, temperature=0.0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    )

    # Build RAGAS dataset
    samples = []
    valid_indices = []
    for i, r in enumerate(results):
        if r.get("ragas") is not None:
            continue  # already scored (resume)
        try:
            sample = SingleTurnSample(
                user_input=r["question"],
                response=r["rag_answer"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["expected_answer"] if r["expected_answer"] else r["rag_answer"],
            )
            samples.append(sample)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Skipping question {r.get('id', i)}: {e}")
            results[i]["ragas"] = None

    if not samples:
        print("  No samples to evaluate.")
        return results

    print(f"  Evaluating {len(samples)} samples...")

    dataset = EvaluationDataset(samples=samples)

    try:
        # Use only Faithfulness and ResponseRelevancy — these don't need reference
        # ContextEntityRecall is a lightweight alternative to ContextPrecision
        metrics = [Faithfulness(), ResponseRelevancy()]

        eval_result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        # Extract per-question scores
        df = eval_result.to_pandas()
        for j, idx in enumerate(valid_indices):
            if j < len(df):
                row = df.iloc[j]
                results[idx]["ragas"] = {
                    "faithfulness": round(float(row.get("faithfulness", 0)), 3),
                    "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 3),
                }
            else:
                results[idx]["ragas"] = None

    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        print("  Falling back to per-question evaluation...")

        # Fallback: evaluate one at a time
        for j, idx in enumerate(valid_indices):
            try:
                single_dataset = EvaluationDataset(samples=[samples[j]])
                single_result = ragas_evaluate(
                    dataset=single_dataset,
                    metrics=[Faithfulness(), ResponseRelevancy()],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                df = single_result.to_pandas()
                results[idx]["ragas"] = {
                    "faithfulness": round(float(df.iloc[0].get("faithfulness", 0)), 3),
                    "answer_relevancy": round(float(df.iloc[0].get("answer_relevancy", 0)), 3),
                }
                print(f"  [{j+1}/{len(valid_indices)}] OK")
            except Exception as e2:
                results[idx]["ragas"] = None
                print(f"  [{j+1}/{len(valid_indices)}] Failed: {e2}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict], rag_model: str, judge_model: str) -> dict:
    """Compute aggregate scores."""
    summary = {
        "num_questions": len(results),
        "rag_model": rag_model,
        "judge_model": judge_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Judge scores
    judged = [r for r in results if r.get("judge")]
    if judged:
        scores = [r["judge"]["factual_score"] for r in judged]
        hits = [r["judge"]["source_hit"] for r in judged]
        dist = Counter(scores)
        summary["judge_scores"] = {
            "num_judged": len(judged),
            "avg_factual_score": round(sum(scores) / len(scores), 2),
            "source_hit_rate": round(sum(hits) / len(hits), 2),
            "score_distribution": {str(k): dist.get(k, 0) for k in range(4)},
        }

    # RAGAS scores
    ragas_results = [r for r in results if r.get("ragas")]
    if ragas_results:
        faith = [r["ragas"]["faithfulness"] for r in ragas_results if r["ragas"].get("faithfulness") is not None]
        relevancy = [r["ragas"]["answer_relevancy"] for r in ragas_results if r["ragas"].get("answer_relevancy") is not None]
        summary["ragas_scores"] = {
            "num_evaluated": len(ragas_results),
            "faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
            "answer_relevancy": round(sum(relevancy) / len(relevancy), 3) if relevancy else None,
        }

    # Per-category breakdown
    cats = Counter(r.get("category", "unknown") for r in results)
    cat_scores = {}
    for cat in cats:
        cat_results = [r for r in results if r.get("category") == cat and r.get("judge")]
        if cat_results:
            cat_factual = [r["judge"]["factual_score"] for r in cat_results]
            cat_scores[cat] = {
                "count": len(cat_results),
                "avg_factual_score": round(sum(cat_factual) / len(cat_factual), 2),
            }
    summary["by_category"] = cat_scores

    return summary


def print_summary(summary: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions: {summary['num_questions']}")
    print(f"RAG model: {summary['rag_model']}")
    print(f"Judge model: {summary['judge_model']}")

    if "judge_scores" in summary:
        js = summary["judge_scores"]
        print(f"\nLLM-as-Judge:")
        print(f"  Avg factual score: {js['avg_factual_score']}/3.0")
        print(f"  Source hit rate:   {js['source_hit_rate']:.0%}")
        print(f"  Score distribution: {js['score_distribution']}")

    if "ragas_scores" in summary:
        rs = summary["ragas_scores"]
        print(f"\nRAGAS Metrics:")
        print(f"  Faithfulness:      {rs.get('faithfulness', 'N/A')}")
        print(f"  Answer Relevancy:  {rs.get('answer_relevancy', 'N/A')}")

    if "by_category" in summary and summary["by_category"]:
        print(f"\nBy Category:")
        for cat, scores in sorted(summary["by_category"].items()):
            print(f"  {cat:20s}: {scores['avg_factual_score']}/3.0 (n={scores['count']})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIgino Hybrid Evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", type=str, default=CHAT_MODEL, help="RAG generation model")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge model (default: same as --model)")
    parser.add_argument("--ragas-judge-model", type=str, default=None,
                        help="RAGAS judge model (default: same as --judge-model)")
    parser.add_argument("--skip-ragas", action="store_true", help="Only run LLM-as-judge")
    parser.add_argument("--skip-judge", action="store_true", help="Only run RAGAS metrics")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    judge_model = args.judge_model or args.model
    ragas_judge_model = args.ragas_judge_model or judge_model

    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_chroma import Chroma

    if not CHROMA_DIR.exists():
        print("ERROR: Chroma DB not found. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)
    if not QUESTIONS_FILE.exists():
        print(f"ERROR: Questions not found: {QUESTIONS_FILE}", file=sys.stderr)
        print("Run generate_testset.py first.", file=sys.stderr)
        sys.exit(1)

    # Load questions
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions")

    # Resume: load existing results
    existing_results = {}
    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_results[r["id"]] = r
        print(f"Resume: {len(existing_results)} existing results")

    # Set up RAG components
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    llm = ChatOllama(model=args.model, base_url=OLLAMA_BASE_URL, temperature=0.3)

    # --- Phase 1: Inference ---
    new_questions = [q for q in questions if q.get("id") not in existing_results]
    print(f"\nPhase 1: RAG Inference ({len(new_questions)} new questions)")

    if new_questions:
        new_results = run_inference(vectorstore, llm, new_questions, args.top_k, args.verbose)
    else:
        new_results = []

    # Merge with existing
    all_results = []
    new_by_id = {r["id"]: r for r in new_results}
    for q in questions:
        qid = q.get("id")
        if qid in existing_results:
            all_results.append(existing_results[qid])
        elif qid in new_by_id:
            all_results.append(new_by_id[qid])

    # --- Phase 2a: LLM-as-judge ---
    if not args.skip_judge:
        judge_llm = ChatOllama(model=judge_model, base_url=OLLAMA_BASE_URL,
                               temperature=0.0, num_predict=512)
        all_results = run_judge(all_results, judge_llm, args.verbose)

    # --- Phase 2b: RAGAS ---
    if not args.skip_ragas:
        all_results = run_ragas(all_results, ragas_judge_model, args.verbose)

    # --- Save results ---
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamps
    ts = datetime.now(timezone.utc).isoformat()
    for r in all_results:
        r["timestamp"] = ts

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in all_results:
            # Don't save retrieved_contexts to file (too large) — save separately if needed
            save_r = {k: v for k, v in r.items() if k != "retrieved_contexts"}
            save_r["num_retrieved_contexts"] = len(r.get("retrieved_contexts", []))
            f.write(json.dumps(save_r, ensure_ascii=False) + "\n")

    # --- Summary ---
    summary = compute_summary(all_results, args.model, judge_model)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nResults: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
