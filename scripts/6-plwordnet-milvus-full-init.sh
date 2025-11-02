#!/bin/bash

# Workers count
WORKERS_COUNT=1

# Which device will be used to prepare embeddings
CUDA_DEVICE="cuda:1"

# path to Milvus database configuration
MILVUS_CONFIG="../resources/configs/milvus-config.json"

# path to embedder configuration
EMBEDDER_CONFIG="../resources/configs/embedder-config.json"


# ##################################################################################
# Prepare database
# ##################################################################################
plwordnet-milvus \
  --log-level=DEBUG \
  --milvus-config="${MILVUS_CONFIG}"on \
  --prepare-database


# ##################################################################################
# Base and fake embeddings
# ##################################################################################
plwordnet-milvus \
  --milvus-config="${MILVUS_CONFIG}" \
  --embedder-config="${EMBEDDER_CONFIG}" \
  --nx-graph-dir=../resources/plwordnet_4_5/full/graphs/full/nx/graphs/ \
  --device="${CUDA_DEVICE}" \
  --log-level=INFO \
  --workers-count=${WORKERS_COUNT} \
  --prepare-base-embeddings-lu \
  --prepare-base-embeddings-synset \
  --prepare-base-mean-empty-embeddings-lu
