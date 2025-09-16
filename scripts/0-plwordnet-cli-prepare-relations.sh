#!/bin/bash

OUT_DIR="resources/plwordnet_4_5/full"
RELATIONS_FILE="${OUT_DIR}/relations.xlsx"

MYSQL_DB_FILE=resources/plwordnet-mysql-db.json

# Prepare relations.xlsx file
plwordnet-cli \
  --use-database \
  --db-config="${MYSQL_DB_FILE}" \
  --dump-relation-types-to-file="${RELATIONS_FILE}" \
  --log-level=DEBUG
