#!/usr/bin/env bash
# Run SOLIDER-REID benchmark evaluation on a custom dataset.
#
# Usage:
#   bash run_benchmark.sh
#
# Requirements:
#   - bounding_box_test/ and query/ inside DATASET_DIR
#   - Filename format: {pid}_c{camid}[...].jpg / .png

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/unified/swin_base.yml"
WEIGHT="${SCRIPT_DIR}/log/unified/swin_base/transformer_120.pth"
DATASET_DIR="/home/vnptai/duytv/projects/Person-ReID/PersonViT/datasets/Unified-ReID-Dataset/entireid_blured"
DEVICE="cuda"
BATCH_SIZE=64
NUM_WORKERS=4

# Set to "--reranking" to enable re-ranking post-processing, or leave empty.
RERANKING=""


if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -f "${WEIGHT}" ]; then
    echo "Checkpoint not found: ${WEIGHT}"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi



cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=0 python benchmark.py \
    --config_file "${CONFIG_FILE}" \
    --weight      "${WEIGHT}" \
    --dataset_dir "${DATASET_DIR}" \
    --device      "${DEVICE}" \
    --batch_size  "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    ${RERANKING}
