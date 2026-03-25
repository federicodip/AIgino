#!/usr/bin/bash -l
#SBATCH --job-name=aigino-testset
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --output=logs/testset-%j.out
#SBATCH --error=logs/testset-%j.err

# ---------------------------------------------------------------
# Generate synthetic eval QA pairs from corpus chunks
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
    apptainer exec --nv --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama serve &
sleep 15

# Warm up model
echo "Loading model..."
curl -s http://localhost:11434/api/generate -d '{"model": "gemma3:12b", "prompt": "", "keep_alive": "60m"}' > /dev/null
echo "Model loaded."

# --- Generate testset ---
echo ""
echo "Generating synthetic QA pairs..."
apptainer exec \
    --env OLLAMA_BASE_URL=http://localhost:11434 \
    --env CHAT_MODEL=gemma3:12b \
    --env CHUNKS_FILE=data/chunks/chunked_pages.jsonl \
    ${CONTAINER_SIF} python ${PROJECT_DIR}/scripts/generate_testset.py \
        --num-questions 50 \
        --qa-per-chunk 2 \
        --resume

echo ""
echo "Testset generation complete."
echo "Output: ${PROJECT_DIR}/eval/eval_questions.jsonl"
