#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

OUT_DIR="/mnt/local/plwn-semantic-embeddings"
BASE_MODEL="/mnt/data2/llms/models/community/EuroBERT/EuroBERT-2.1B"
BASE_NAME=$(basename "${BASE_MODEL}")

#DATASET_SUBDIR="test_dataset_ratio-2.0"
DATASET_SUBDIR="full_dataset_ratio-2.0"
DATASET_PATH="../../_resources_local/datasets/embedder/${DATASET_SUBDIR}"

DATASET_TRAIN="${DATASET_PATH}/train.json"
DATASET_EVAL="${DATASET_PATH}/test.json"

DATE_AS_STR=$(date +%Y%m%d_%H%M%S)
python3 train-bi-encoder.py \
  -m "${BASE_MODEL}" \
  --train-file "${DATASET_TRAIN}" \
  --valid-file "${DATASET_EVAL}" \
  -O "${OUT_DIR}/${BASE_NAME}/biencoder/${DATE_AS_STR}_${DATASET_SUBDIR}" \
  -WB
