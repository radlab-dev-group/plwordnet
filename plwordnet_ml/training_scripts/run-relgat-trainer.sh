#!/bin/bash

# =============================================================================
# =============================================================================
# --------------------  GENERAL OPTIONS
# =============================================================================
# Device: {cuda, cpu, cuda:x}
DEVICE="cuda"
# Ratio of training data
TRAIN_RATIO="0.80"
#
# =============================================================================
# =============================================================================
# --------------------  TRAINING PARAMETERS
# =============================================================================
# Number of epochs
EPOCHS=5
# Tain/eval batch size
BATCH_SIZE=32
# Scorer, one of: {distmult, transe}
SCORER="distmult"
# Out RelGAT dimension (for each head)
GAT_OUT_DIM=300
# Number of heads (each with projection to GAT_OUT_DIM)
NUM_OF_HEADS=6
# Number of negative examples for each positive one
NUM_NEG_TO_POS=6
# Dropout used while training
DROPOUT=0.25
#
# =============================================================================
# =============================================================================
# --------------------  DATASET CONFIGURATION
# =============================================================================
DATASET_ROOT="/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers"
# Available datasets:
#  - FULL: dataset_20250824_full
#  - SAMPLE: dataset_20250824_limit_1000
DATASET_DIR="${DATASET_ROOT}/dataset_20250824_full"
LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

# =============================================================================
# =============================================================================
# --------------------  APPLICATION CALL
# =============================================================================
CUDA_VISIBLE_DEVICES=2 python3 ../embedder/trainer/main/relgat.py \
  --num-neg="${NUM_NEG_TO_POS}" \
  --heads="${NUM_OF_HEADS}" \
  --epochs="${EPOCHS}" \
  --scorer="${SCORER}" \
  --dropout="${DROPOUT}" \
  --gat-out-dim="${GAT_OUT_DIM}" \
  --train-ratio="${TRAIN_RATIO}" \
  --batch-size="${BATCH_SIZE}" \
  --nodes-embeddings-path="${LU_EMBEDDING}" \
  --relations-mapping="${RELS_MAPPING}" \
  --relations-triplets="${RELS_TRIPLETS}" \
  --device="${DEVICE}" --help
