#!/bin/bash

# =============================================================================
# =============================================================================
# --------------------  GENERAL OPTIONS
# =============================================================================
# Device: {cuda, cpu, cuda:x}
DEVICE="cuda"
CUDA_DEVICES="0"
MAX_CHECKPOINTS=5

# Ratio of training data
TRAIN_EVAL_DATASET_RATIO="0.80"

# Which architecture will be trained {small, medium}
ARCHITECTURE="small"
#
# =============================================================================
# =============================================================================
# --------------------  TRAINING PARAMETERS
# =============================================================================
if [[ "${ARCHITECTURE}" == "small" ]]
then
  # Tain/eval batch size
  BATCH_SIZE=64
  # Out RelGAT dimension (for each head)
  GAT_OUT_DIM=128
  # Number of layers
  NUM_OF_LAYERS=1
  # Number of heads (each with projection to GAT_OUT_DIM)
  NUM_OF_HEADS=6
elif [[ "${ARCHITECTURE}" == "medium" ]]
then
  # Tain/eval batch size
  BATCH_SIZE=32
  # Out RelGAT dimension (for each head)
  GAT_OUT_DIM=256
  # Number of layers
  NUM_OF_LAYERS=3
  # Number of heads (each with projection to GAT_OUT_DIM)
  NUM_OF_HEADS=12
else
  echo "Supported architectures: [small, medium]"
  exit 1
fi

# Number of epochs
EPOCHS=10
# Scorer, one of: [distmult, transe]
SCORER="distmult"
# Dropout used while training
DROPOUT=0.3
# Number of negative examples for each positive one
NUM_NEG_TO_POS=4
# Logging during training after each n steps
LOG_EVERY_N_STEPS=10
# Learning rate
LEARNING_RATE=0.00005
# Multiplicative factor applied to LR after warm‑up (default: 1.0 – no change)
LR_DECAY=1.0
# Learning rate scheduler, one of: [linear, cosine, constant]
LR_SCHEDULER="linear"
# Optional explicit warmup steps (comment out to auto-compute)
# WARMUP_STEPS=500
# weight decay (0.0 is default) - used in Adam optimizer
WEIGHT_DECAY=0.0
# If set, clips gradient norm to this value (default: None – no clipping)
#GRADIENT_CLIPPING=5
# Number of evaluation steps without improvement after which training stops
EARLY_STOP_PATIENCE_STEPS=10
# To enable Automatic Mixed Precision (AMP) training uncomment next line
#USE_AMP=1
# If set, the model will not mask edges by relation type - to use option
# uncomment next line
#DISABLE_EDGE_TYPES=1

# =============================================================================
# =============================================================================
# --------------------  STORING/EVALUATE MODEL WHILE TRAINING
# Output directory to store the model and checkpoints during training
OUT_MODEL_DIR="relgat-models/relgat-${ARCHITECTURE}_$(date +%Y%m%d_%H%M%S)"
# Save model every n steps
SAVE_N_STEPS=2000
# Optional explicit eval steps, if not given, then eval will be done after each epoch
EVAL_N_STEPS=1000
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
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
python3 ../embedder/trainer/main/relgat.py \
  --warmup-steps="${WARMUP_STEPS}" \
  --lr="${LEARNING_RATE}" \
  --lr-decay="${LR_DECAY}" \
  --early-stop-patience="${EARLY_STOP_PATIENCE_STEPS}" \
  --weight-decay="${WEIGHT_DECAY}" \
  --lr-scheduler="${LR_SCHEDULER}" \
  --num-neg="${NUM_NEG_TO_POS}" \
  --gat-num-layers="${NUM_OF_LAYERS}" \
  --heads="${NUM_OF_HEADS}" \
  --epochs="${EPOCHS}" \
  --scorer="${SCORER}" \
  --dropout="${DROPOUT}" \
  --gat-out-dim="${GAT_OUT_DIM}" \
  --train-ratio="${TRAIN_EVAL_DATASET_RATIO}" \
  --batch-size="${BATCH_SIZE}" \
  --nodes-embeddings-path="${LU_EMBEDDING}" \
  --relations-mapping="${RELS_MAPPING}" \
  --relations-triplets="${RELS_TRIPLETS}" \
  --device="${DEVICE}" \
  --log-every-n-steps="${LOG_EVERY_N_STEPS}" \
  --save-dir="${OUT_MODEL_DIR}" \
  --save-every-n-steps="${SAVE_N_STEPS}" \
  --max-checkpoints="${MAX_CHECKPOINTS}" \
  ${EVAL_N_STEPS:+--eval-every-n-steps="${EVAL_N_STEPS}"} \
  ${WARMUP_STEPS:+--warmup-steps="${WARMUP_STEPS}"} \
  ${GRADIENT_CLIPPING:+--grad-clip-norm="${GRADIENT_CLIPPING}"} \
  ${USE_AMP:+--use-amp="${USE_AMP}"} \
  ${DISABLE_EDGE_TYPES:+--disable-edge-type-mask="${DISABLE_EDGE_TYPES}"}
