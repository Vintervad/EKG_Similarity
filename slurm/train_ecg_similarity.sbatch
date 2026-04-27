#!/bin/bash
#SBATCH --job-name=ekg_similarity_train
#SBATCH --output=logs/ekg_similarity_%j.out
#SBATCH --error=logs/ekg_similarity_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="/dnec/work/ST10/python/EKG_Sim"
DATA_ROOT="/dnec/data"
TRAIN_CSV="${DATA_ROOT}/metadata/train.csv"
VAL_CSV="${DATA_ROOT}/metadata/val.csv"
TEST_CSV="${DATA_ROOT}/metadata/test.csv"
OUTPUT_DIR="${REPO_DIR}/checkpoints"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Uncomment and adapt to your cluster environment.
# module load python/3.12 cuda/12.1
# source .venv/bin/activate
# or:
# conda activate your-env-name

python main.py \
  --data-root "${DATA_ROOT}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --test-csv "${TEST_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 16 \
  --channels 12 \
  --sequence-length 5000 \
  --epochs 50 \
  --num-workers 8 \
  --pin-memory \
  --early-stopping-patience 10 \
  --device cuda
