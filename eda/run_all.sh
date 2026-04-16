#!/bin/bash
# eda/run_all.sh — Run the full EDA suite in one shot
# ==============================================================
# Usage:
#     bash eda/run_all.sh
#     bash eda/run_all.sh --pred_csv predictions.csv              # include error analysis
#     bash eda/run_all.sh --cohort_csv data/cohort.csv            # include cohort plots
#     bash eda/run_all.sh --pred_csv predictions.csv --n_samples 100
#
# All flags are optional. Scripts 01–03 always run; script 04
# runs in baseline mode unless --pred_csv is provided.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults (override via env or flags below) ────────────────
TRAIN_CSV="${TRAIN_CSV:-$REPO_ROOT/data/train.csv}"
VAL_CSV="${VAL_CSV:-$REPO_ROOT/data/val.csv}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/eda}"
N_SAMPLES="${N_SAMPLES:-50}"
N_SLICES="${N_SLICES:-3}"
PRED_CSV=""
COHORT_CSV=""

# ── Parse optional flags ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_csv)   TRAIN_CSV="$2";   shift 2 ;;
        --val_csv)     VAL_CSV="$2";     shift 2 ;;
        --out_root)    OUT_ROOT="$2";    shift 2 ;;
        --pred_csv)    PRED_CSV="$2";    shift 2 ;;
        --cohort_csv)  COHORT_CSV="$2";  shift 2 ;;
        --n_samples)   N_SAMPLES="$2";   shift 2 ;;
        --n_slices)    N_SLICES="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  MYGO — EDA Suite"
echo "============================================================"
echo "  train_csv  : $TRAIN_CSV"
echo "  val_csv    : $VAL_CSV"
echo "  out_root   : $OUT_ROOT"
echo "  pred_csv   : ${PRED_CSV:-<none — script 04 uses baseline>}"
echo "  cohort_csv : ${COHORT_CSV:-<none — cohort plots skipped>}"
echo "  n_samples  : $N_SAMPLES"
echo "  n_slices   : $N_SLICES"
echo "============================================================"
echo ""

FAILED=0

# ── 01: Centiloid distribution ────────────────────────────────
echo ">>> 01_centiloid_distribution.py"
python3 "$SCRIPT_DIR/01_centiloid_distribution.py" \
    --train_csv "$TRAIN_CSV" \
    --val_csv   "$VAL_CSV" \
    --out_dir   "$OUT_ROOT/01_centiloid_distribution" \
    || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
echo ""

# ── 02: Tracer comparison ─────────────────────────────────────
echo ">>> 02_tracer_comparison.py"
python3 "$SCRIPT_DIR/02_tracer_comparison.py" \
    --train_csv "$TRAIN_CSV" \
    --val_csv   "$VAL_CSV" \
    --out_dir   "$OUT_ROOT/02_tracer_comparison" \
    --n_slices  "$N_SLICES" \
    || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
echo ""

# ── 03: Calibration analysis ──────────────────────────────────
echo ">>> 03_calibration_analysis.py"
COHORT_ARG=""
if [[ -n "$COHORT_CSV" ]]; then
    COHORT_ARG="--cohort_csv $COHORT_CSV"
fi
python3 "$SCRIPT_DIR/03_calibration_analysis.py" \
    --train_csv  "$TRAIN_CSV" \
    --val_csv    "$VAL_CSV" \
    --out_dir    "$OUT_ROOT/03_calibration_analysis" \
    --n_samples  "$N_SAMPLES" \
    $COHORT_ARG \
    || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
echo ""

# ── 04: Model error analysis ──────────────────────────────────
echo ">>> 04_model_error_analysis.py"
PRED_ARG=""
if [[ -n "$PRED_CSV" ]]; then
    PRED_ARG="--pred_csv $PRED_CSV"
fi
python3 "$SCRIPT_DIR/04_model_error_analysis.py" \
    --val_csv  "$VAL_CSV" \
    --out_dir  "$OUT_ROOT/04_model_error_analysis" \
    $PRED_ARG \
    || { echo "  ✗ FAILED"; FAILED=$((FAILED+1)); }
echo ""

# ── Summary ───────────────────────────────────────────────────
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
    echo "  ✓ All 4 EDA scripts passed"
else
    echo "  ✗ $FAILED script(s) failed"
fi
echo "  Outputs → $OUT_ROOT"
echo "============================================================"

exit $FAILED
