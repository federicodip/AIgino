#!/usr/bin/bash -l
#SBATCH --job-name=aigino-eval
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# ---------------------------------------------------------------
# Batch evaluation: run all demo questions through RAG pipeline
# ---------------------------------------------------------------

set -euo pipefail

PROJECT_DIR=~/aigino
SCRATCH_DIR=/scratch/fdipas/aigino
OLLAMA_SIF=~/scratch/graphRAG/containers/ollama.sif
OLLAMA_MODELS=/scratch/fdipas/graphRAG/ollama
CONTAINER_SIF=~/scratch/graphRAG/containers/graphrag.sif

module load apptainer

# --- Start Ollama ---
echo "Starting Ollama server..."
HTTPS_PROXY=http://10.129.62.115:3128 HTTP_PROXY=http://10.129.62.115:3128 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    apptainer exec --nv --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama serve &
sleep 15

# Warm up models
echo "Loading models into GPU..."
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.5:27b", "prompt": "", "keep_alive": "60m"}' > /dev/null
curl -s http://localhost:11434/api/embed -d '{"model": "qwen3-embedding:8b", "input": "test"}' > /dev/null
echo "Models loaded."

# --- Run eval ---
echo ""
echo "Running batch evaluation..."
apptainer exec \
    --env OLLAMA_BASE_URL=http://localhost:11434 \
    --env EMBEDDING_MODEL=qwen3-embedding:8b \
    --env CHAT_MODEL=qwen3.5:27b \
    --env CHROMA_DIR=${SCRATCH_DIR}/chroma_db \
    --env CHROMA_COLLECTION=aigino \
    --env TOP_K=8 \
    ${CONTAINER_SIF} python ${PROJECT_DIR}/scripts/run_eval.py --resume

echo ""
echo "Evaluation complete. Results at: ${PROJECT_DIR}/eval/eval_results.jsonl"
