#!/bin/bash

OUT_DIR="resources/plwordnet_4_5/full"

TRAIN_FILE="${OUT_DIR}/embedder/plwn_4_5_embedder_dataset/train.json"
TEST_FILE="${OUT_DIR}/embedder/plwn_4_5_embedder_dataset/test.json"

python3 apps/utils/embedder/embedder-dataset-dedupliactor.py \
    --train="${TRAIN_FILE}" \
    --test="${TEST_FILE}"
