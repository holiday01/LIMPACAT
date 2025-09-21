#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=LIMPACAT.py

DATA_ROOT=""

declare -A json_name=(
  ["b"]="log_B_cell_svs.json"
  ["b34"]="Pro-B_cell_CD34+_svs.json"
  ["mo"]="Monocyte_svs.json"
  ["nk"]="NK_cell_svs.json"
)

for cell in "b" "b34" "mo" "nk"; do
  CKPT="./model/${cell}.pt"
  JSON_PATH="${json_name[$cell]}"

  ${PY} "${SCRIPT}" \
    --dataset_json "${JSON_PATH}" \
    --data_root "${DATA_ROOT}" \
    --checkpoint "${CKPT}" \
    --split_key validation \
    --tile_count 45 \
    --tile_size 224 \
    --workers 0 \
    --amp
done