#!/bin/bash

# prepare database
plwordnet-milvus \
  --log-level=DEBUG \
  --milvus-config=../resources/configs/milvus-config-pk.json \
  --prepare-database


# Base and fake embeddings
plwordnet-milvus \
  --milvus-config=../resources/configs/milvus-config-pk.json \
  --embedder-config=../resources/configs/embedder-config.json \
  --nx-graph-dir=../resources/plwordnet_4_5/full/graphs/full/nx/graphs/ \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-base-embeddings-lu \
  --prepare-base-embeddings-synset \
  --prepare-base-mean-empty-embeddings-lu
