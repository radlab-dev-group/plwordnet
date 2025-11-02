#!/bin/bash

# path to Milvus database configuration
MILVUS_CONFIG="../resources/configs/milvus-config.json"

# path to embedder configuration
EMBEDDER_CONFIG="../resources/configs/embedder-config.json"

# Path to NetworkX graphs (lu, synsets)
NX_GRAPHS_DIR="resources/plwordnet_4_5/full/graphs/full/nx/graphs/"

# Path to mappings (relations identifiers)
RELGAT_MAPPING_DIR="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm"

# Path to export RelGAT dataset
RELGAT_DATASET_DIR="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way"

# ##################################################################################
# Export dataset to RelGAT trainer
# ##################################################################################
plwordnet-milvus \
  --milvus-config="${MILVUS_CONFIG}" \
  --embedder-config="${EMBEDDER_CONFIG}" \
  --nx-graph-dir="${NX_GRAPHS_DIR}" \
  --relgat-mapping-directory="${RELGAT_MAPPING_DIR}" \
  --relgat-dataset-directory="${RELGAT_DATASET_DIR}" \
  --log-level=DEBUG \
  --export-relgat-dataset \
  --export-relgat-mapping