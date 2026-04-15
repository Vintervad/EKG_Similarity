#!/bin/bash
#SBATCH --job-name=ekg_similarity_faiss
#SBATCH --output=logs/ekg_similarity_faiss_%j.out
#SBATCH --error=logs/ekg_similarity_faiss_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="/path/to/EKG_Similarity"
DATA_ROOT="/path/to/data_root"
CHECKPOINT="/path/to/checkpoints/best.pt"
OUTPUT_DIR="/path/to/embedding_output_dir"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Uncomment and adapt to your cluster environment.
# module load python/3.12 cuda/12.1
# source .venv/bin/activate
# or:
# conda activate your-env-name

# Optional sanity checks before building the FAISS GPU index.
python -c "import torch; print('torch.cuda.is_available =', torch.cuda.is_available())"
python -c "import faiss; print('faiss version =', faiss.__version__); print('gpu support =', hasattr(faiss, 'StandardGpuResources'))"

python embed_dataset.py \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --splits all \
  --batch-size 8 \
  --channels 12 \
  --device cuda \
  --embedding-type global \
  --output-dir "${OUTPUT_DIR}" \
  --backend faiss \
  --faiss-index-type ivf-flat \
  --faiss-use-gpu \
  --faiss-gpu-device 0 \
  --faiss-nlist 65536 \
  --faiss-nprobe 64 \
  --faiss-train-size 200000 \
  --faiss-add-batch-size 100000
