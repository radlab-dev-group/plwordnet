#!/bin/bash

NOT TESTED
exit 1

#
#DATASET_DIR="resources/plwordnet_4_5/embedder/"
#
#INPUT_FILE="${DATASET_DIR}/plwn_4_5_embedder_raw.jsonl"
#OUTPUT_DIR="${DATASET_DIR}/plwn_4_5_embedder_dataset"
#
#python3 apps/utils/embedder/convert-raw-embedder-dump-to-dataset.py \
#    --jsonl-path="${INPUT_FILE}" \
#    --output-dir="${OUTPUT_DIR}" \
#    --train-ratio=0.90 \
#    --split-to-sentences \
#    --n-workers=12 \
#    --batch-size=100 \
#    --correct-texts \
#    --prompts-dir=resources/prompts \
#    --prompt-name="ollama/correct_text" \
#    --openapi-config=resources/ollama-config.json

#python3 apps/utils/embedder/embedder-dataset-dedupliactor.py \
#    --train=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/train.json
#    --test=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/test.json