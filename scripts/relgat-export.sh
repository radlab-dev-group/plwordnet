#!/bin/bash


plwordnet-milvus \
  --milvus-config=../resources/milvus-config-pk.json \
  --nx-graph-dir=/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs \
  --relgat-mapping-directory=/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers/wtcsnxj9 \
  --relgat-dataset-directory=/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers/wtcsnxj9/dataset_syn_two_way \
  --log-level=DEBUG \
  --export-relgat-dataset
