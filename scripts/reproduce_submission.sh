#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/reproduce_submission.sh [ASSET_ROOT] [FEATURE_CACHE]

Reproduce the archived competition topology from downloaded task-head weights and a frozen test-pair feature cache.

Defaults:
  ASSET_ROOT    artifacts/competition_submission
  FEATURE_CACHE ASSET_ROOT/cache/test_pairs_s518

Environment overrides:
  PAIRUAV_SUBMISSION_ASSETS   Default asset root
  PAIRUAV_SUBMISSION_CACHE    Default feature-cache directory
  PAIRUAV_SUBMISSION_RUN_DIR  Exact output directory (must not already exist)
  PAIRUAV_EXPECTED_ROWS       Expected cache rows (default: 2773116)
  PAIRUAV_DEVICE              PyTorch device (default: cuda)
  PAIRUAV_BATCH_SIZE          Cached-head inference batch size (default: 2048)
  PAIRUAV_SKIP_MANIFEST       Set to 1 only for local smoke assets without MANIFEST.sha256
  PYTHON_BIN                  Python executable (default: python)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${1:-${PAIRUAV_SUBMISSION_ASSETS:-$ROOT/artifacts/competition_submission}}"
FEATURE_CACHE="${2:-${PAIRUAV_SUBMISSION_CACHE:-$ASSET_ROOT/cache/test_pairs_s518}}"
RUN_DIR="${PAIRUAV_SUBMISSION_RUN_DIR:-$ROOT/outputs/submission_$(date +%Y%m%d_%H%M%S)}"
EXPECTED_ROWS="${PAIRUAV_EXPECTED_ROWS:-2773116}"
DEVICE="${PAIRUAV_DEVICE:-cuda}"
BATCH_SIZE="${PAIRUAV_BATCH_SIZE:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Output directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/workflow.log") 2>&1

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

ANGLE_CKPT="$ASSET_ROOT/checkpoints/S0_rich_noc/head_best_angle.pt"
RANGE_CKPT="$ASSET_ROOT/checkpoints/C_rel_rich/range_head_best_distance.pt"
RANGE_B_CKPT="$ASSET_ROOT/checkpoints/B_mse_ab/range_head_best_distance.pt"
ANGLE_CONFIG="$ROOT/configs/submission/angle_s0.json"
RANGE_C_CONFIG="$ROOT/configs/submission/range_c_rel_rich.json"
RANGE_B_CONFIG="$ROOT/configs/submission/range_b_mse_ab.json"

require_file "$FEATURE_CACHE/features.npy"
require_file "$FEATURE_CACHE/meta.json"
require_file "$ANGLE_CKPT"
require_file "$RANGE_CKPT"
require_file "$RANGE_B_CKPT"
require_file "$ANGLE_CONFIG"
require_file "$RANGE_C_CONFIG"
require_file "$RANGE_B_CONFIG"

if [[ "${PAIRUAV_SKIP_MANIFEST:-0}" != "1" ]]; then
  require_file "$ASSET_ROOT/MANIFEST.sha256"
  (
    cd "$ASSET_ROOT"
    sha256sum -c MANIFEST.sha256
  )
fi

printf '%s\n' \
  "asset_root=$ASSET_ROOT" \
  "feature_cache=$FEATURE_CACHE" \
  "expected_rows=$EXPECTED_ROWS" \
  "device=$DEVICE" \
  "batch_size=$BATCH_SIZE" \
  > "$RUN_DIR/workflow.env"

cd "$ROOT"
"$PYTHON_BIN" -m pairuav.infer_cache \
  --feature-cache "$FEATURE_CACHE" \
  --angle-ckpt "$ANGLE_CKPT" \
  --angle-config "$ANGLE_CONFIG" \
  --range-ckpt "$RANGE_CKPT" \
  --range-config "$RANGE_C_CONFIG" \
  --range2-ckpt "$RANGE_B_CKPT" \
  --range2-config "$RANGE_B_CONFIG" \
  --gate-threshold 80 \
  --raw-heads-out "$RUN_DIR/raw_heads.txt" \
  --out "$RUN_DIR/result_continuous.txt" \
  --expected-rows "$EXPECTED_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --matmul-precision high

"$PYTHON_BIN" -m pairuav.postproc_maphard \
  --pred "$RUN_DIR/result_continuous.txt" \
  --out-dir "$RUN_DIR/maphard" \
  --expected-lines "$EXPECTED_ROWS"

echo "Competition topology reproduced under: $RUN_DIR"
echo "Continuous output: $RUN_DIR/result_continuous.txt"
echo "Decoded submission: $RUN_DIR/maphard/result.zip"
