#!/bin/bash

OUT_DIR="resources/plwordnet_4_5/full"

INPUT_FILE="${OUT_DIR}/embedder/plwn_4_5_embedder_raw.jsonl"
OUTPUT_DIR="${OUT_DIR}/embedder/plwn_4_5_embedder_dataset"

python3 apps/utils/embedder/convert-raw-embedder-dump-to-dataset.py \
    --jsonl-path="${INPUT_FILE}" \
    --output-dir="${OUTPUT_DIR}" \
    --train-ratio=0.90 \
    --split-to-sentences \
    --n-workers=20 \
    --batch-size=500
