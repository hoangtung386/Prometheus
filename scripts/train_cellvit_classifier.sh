#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/CellViT-plus-plus /path/to/cellvit-puma-track2.yaml" >&2
  exit 2
fi

CELLVIT_REPO="$1"
CONFIG="$2"

if [[ ! -f "$CELLVIT_REPO/cellvit/train_cell_classifier_head.py" ]]; then
  echo "CellViT++ source not found at: $CELLVIT_REPO" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Generated classifier config not found: $CONFIG" >&2
  exit 2
fi

cd "$CELLVIT_REPO"
python cellvit/train_cell_classifier_head.py --config "$CONFIG"
