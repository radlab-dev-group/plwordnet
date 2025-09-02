#!/bin/bash


plwordnet-milvus \
  --milvus-config=resources/milvus-config-pk.json \
  --embedder-config=resources/embedder-config.json \
  --nx-graph-dir=/path/to/plwordnet/graphs \
  --relgat-mapping-directory=resources/aligned-dataset-identifiers/ \
  --relgat-dataset-directory=resources/aligned-dataset-identifiers/dataset \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-database \
  --prepare-base-embeddings-lu \
  --prepare-base-mean-empty-embeddings-lu \
  --prepare-base-embeddings-synset \
  --export-relgat-dataset \
  --export-relgat-mapping
