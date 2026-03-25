#!/usr/bin/bash -l
#SBATCH --job-name=aigino-chat
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --output=logs/chat-%j.out
#SBATCH --error=logs/chat-%j.err

# ---------------------------------------------------------------
# Interactive-ish RAG chat session
# Runs a single query passed as argument, or reads from stdin
#
# Usage:
#   sbatch jobs/chat.sh "What does Hyginus 2 say about limites?"
#   OR for interactive use via srun:
#   srun --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB" \
#        --partition=lowprio --mem=32G --time=02:00:00 --pty \
#        bash jobs/chat.sh
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
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.5:27b", "prompt": "", "keep_alive": "30m"}' > /dev/null
curl -s http://localhost:11434/api/embed -d '{"model": "qwen3-embedding:8b", "input": "test"}' > /dev/null
echo "Models loaded."

# --- Run chat ---
apptainer exec \
    --env OLLAMA_BASE_URL=http://localhost:11434 \
    --env EMBEDDING_MODEL=qwen3-embedding:8b \
    --env CHAT_MODEL=qwen3.5:27b \
    --env CHROMA_DIR=${SCRATCH_DIR}/chroma_db \
    --env CHROMA_COLLECTION=aigino \
    --env TOP_K=5 \
    ${CONTAINER_SIF} python ${PROJECT_DIR}/scripts/chat.py "$@"
