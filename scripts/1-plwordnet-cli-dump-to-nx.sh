#!/bin/bash

OUT_DIR="resources/plwordnet_4_5/full"
NX_GRAPH_DIR="${OUT_DIR}/graphs/full"

MYSQL_DB_FILE=resources/configs/plwordnet-mysql-db-pk.json

# Convert db to networkx file (with wikipedia extraction)
plwordnet-cli \
  --use-database \
  --db-config="${MYSQL_DB_FILE}" \
  --log-level=DEBUG \
  --show-progress-bar \
  --nx-graph-dir="${NX_GRAPH_DIR}" \
  --convert-to-nx-graph \
  --extract-wikipedia-articles
