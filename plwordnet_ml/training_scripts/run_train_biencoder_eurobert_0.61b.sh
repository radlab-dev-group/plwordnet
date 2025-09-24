#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

OUT_DIR="/mnt/local/plwn-semantic-embeddings-pl-en"
BASE_MODEL="/mnt/data2/llms/models/community/EuroBERT/EuroBERT-610m"
BASE_NAME=$(basename "${BASE_MODEL}")

DATASET_SUBDIR="20250924/plwn_4_5_embedder_dataset/"
DATASET_PATH="/mnt/data2/data/datasets/radlab-semantic-embeddings/${DATASET_SUBDIR}"

DATASET_TRAIN="${DATASET_PATH}/train_deduplicated.json"
DATASET_EVAL="${DATASET_PATH}/test_deduplicated.json"

DATE_AS_STR=$(date +%Y%m%d_%H%M%S)
python3 ../embedder/trainer/train-bi-encoder.py \
  -m "${BASE_MODEL}" \
  --train-file "${DATASET_TRAIN}" \
  --valid-file "${DATASET_EVAL}" \
  -O "${OUT_DIR}/${BASE_NAME}/biencoder/${DATE_AS_STR}_${DATASET_SUBDIR}" \
  -WB
