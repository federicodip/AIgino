#!/usr/bin/bash -l
#SBATCH --job-name=aigino-setup
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

# ---------------------------------------------------------------
# One-time setup: pull qwen3-embedding:8b model, run ingestion
# ---------------------------------------------------------------

set -euo pipefail

PROJECT_DIR=~/aigino
SCRATCH_DIR=/scratch/fdipas/aigino
OLLAMA_SIF=~/scratch/graphRAG/containers/ollama.sif
OLLAMA_MODELS=/scratch/fdipas/graphRAG/ollama
CONTAINER_SIF=~/scratch/graphRAG/containers/graphrag.sif

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${SCRATCH_DIR}

module load apptainer

# --- Start Ollama ---
echo "Starting Ollama server..."
HTTPS_PROXY=http://10.129.62.115:3128 HTTP_PROXY=http://10.129.62.115:3128 \
    apptainer exec --nv --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama serve &
sleep 15

# --- Pull embedding model (if not already present) ---
echo "Pulling qwen3-embedding:8b..."
curl -s http://localhost:11434/api/pull -d '{"name": "qwen3-embedding:8b"}' | tail -1
echo ""
echo "Model pull complete."

# Verify models
echo "Available models:"
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  {m['name']} ({m.get('size', 0) / 1e9:.1f}GB)\")
"

# --- Run ingestion ---
echo ""
echo "Running ingestion..."
apptainer exec \
    --env OLLAMA_BASE_URL=http://localhost:11434 \
    --env EMBEDDING_MODEL=qwen3-embedding:8b \
    --env CHROMA_DIR=${SCRATCH_DIR}/chroma_db \
    --env CHUNKS_FILE=data/chunks/chunked_pages.jsonl \
    --env CHROMA_COLLECTION=aigino \
    ${CONTAINER_SIF} python ${PROJECT_DIR}/scripts/ingest.py --reset

echo ""
echo "Setup complete. Chroma DB at: ${SCRATCH_DIR}/chroma_db"
