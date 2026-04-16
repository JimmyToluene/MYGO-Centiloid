#!/bin/bash
# dev/train.sh — launcher for dev/train.py
# ─────────────────────────────────────────
# Usage:
#     bash dev/train.sh                               # uses config/default.toml values
#     bash dev/train.sh --epochs 50 --batch_size 8    # override flags are forwarded
#
# Defaults match dev/config/default.toml; edit that file for persistent changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_CSV="${TRAIN_CSV:-/projectnb/medaihack/ABPET/data/train.csv}"
VAL_CSV="${VAL_CSV:-/projectnb/medaihack/ABPET/data/val.csv}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"

python3 "$SCRIPT_DIR/train.py" \
    --train_csv      "$TRAIN_CSV" \
    --val_csv        "$VAL_CSV" \
    --checkpoint_dir "$CKPT_DIR" \
    "$@"
