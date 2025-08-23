#!/bin/bash

EPOCHS=12
BATCH_SIZE=1024
TRAIN_RATIO="0.9"

DATASET_ROOT="/mnt/data2/data/resources/plwordnet_handler/relgat"
DATASET_DIR="${DATASET_ROOT}/aligned-dataset-identifiers/dataset__limit_1000"
LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

python3 ../embedder/trainer/main/relgat.py \
  --epochs="${EPOCHS}" \
  --train-ratio="${TRAIN_RATIO}" \
  --batch-size="${BATCH_SIZE}" \
  --nodes-embeddings-path="${LU_EMBEDDING}" \
  --relations-mapping="${RELS_MAPPING}" \
  --relations-triplets="${RELS_TRIPLETS}"
