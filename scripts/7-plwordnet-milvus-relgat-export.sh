#!/bin/bash

plwordnet-milvus \
  --milvus-config=resources/configs/milvus-config-pk.json \
  --nx-graph-dir="resources/plwordnet_4_5/full/graphs/full/nx/graphs/" \
  --relgat-mapping-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm" \
  --relgat-dataset-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way" \
  --log-level=DEBUG \
  --export-relgat-dataset \
  --export-relgat-mapping