#!/bin/bash

OUT_DIR="resources/plwordnet_4_5/full"

EMBEDDER_FILE_RAW="${OUT_DIR}/embedder/plwn_4_5_embedder_raw.jsonl"
RELATIONS_FILE="${OUT_DIR}/relations.xlsx"
NX_GRAPH_DIR="${OUT_DIR}/graphs/full/nx/graphs"

# Convert networkx to raw embedder dataset
plwordnet-cli \
  --log-level=DEBUG \
  --show-progress-bar \
  --embedder-low-high-ratio=2.0 \
  --nx-graph-dir="${NX_GRAPH_DIR}" \
  --dump-embedder-dataset-to-file="${EMBEDDER_FILE_RAW}" \
  --xlsx-relations-weights="${RELATIONS_FILE}"
