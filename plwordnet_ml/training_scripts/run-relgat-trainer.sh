#!/bin/bash

DEVICE="cuda"

EPOCHS=25
BATCH_SIZE=256
TRAIN_RATIO="0.82"

DATASET_ROOT="/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers"
# Available datasets:
#  - FULL: dataset_20250824_full
#  - SAMPLE: dataset_20250824_limit_1000
DATASET_DIR="${DATASET_ROOT}/dataset_20250824_full"

LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

#CUDA_VISIBLE_DEVICES=2 python3 relgat_main.py \
CUDA_VISIBLE_DEVICES=2 python3 ../embedder/trainer/main/relgat.py \
  --epochs="${EPOCHS}" \
  --train-ratio="${TRAIN_RATIO}" \
  --batch-size="${BATCH_SIZE}" \
  --nodes-embeddings-path="${LU_EMBEDDING}" \
  --relations-mapping="${RELS_MAPPING}" \
  --relations-triplets="${RELS_TRIPLETS}" \
  --device="${DEVICE}"
