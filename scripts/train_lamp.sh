#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_lamp.sh [TRAIN_INDEX] [VAL_INDEX] [RUN_DIR]

Train the paper LaMP method from fixed pair indexes. The default indexes are the released 32K training subset and
2,048-pair validation subset. Supply different index files to change the data scale.

Required environment when extracting features:
  PAIRUAV_ALL_TRAIN_JSON  Root containing all pair JSON files
  PAIRUAV_TRAIN_IMAGES    Root containing the training images
  VGGT_WEIGHT             Official VGGT checkpoint

Optional precomputed-cache mode (both variables are required together):
  PAIRUAV_TRAIN_CACHE     Cache matching TRAIN_INDEX
  PAIRUAV_VAL_CACHE       Cache matching VAL_INDEX

Other overrides:
  PAIRUAV_ENV_FILE, PAIRUAV_DEVICE, PAIRUAV_SEED, PAIRUAV_IMAGE_SIZE,
  PAIRUAV_EXTRACT_BATCH_SIZE, PAIRUAV_WORKERS, PAIRUAV_FEATURE_DEVICE, PYTHON_BIN
  PAIRUAV_ANGLE_CONFIG, PAIRUAV_RANGE_CONFIG (advanced recipe override)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PAIRUAV_ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$PAIRUAV_ENV_FILE"
fi

TRAIN_INDEX="${1:-$ROOT/data_index/train_balanced_32768.txt}"
VAL_INDEX="${2:-$ROOT/data_index/val_quick_2048.txt}"
SEED="${PAIRUAV_SEED:-2026}"
RUN_DIR="${3:-$ROOT/outputs/lamp_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${PAIRUAV_DEVICE:-cuda}"
IMAGE_SIZE="${PAIRUAV_IMAGE_SIZE:-518}"
EXTRACT_BATCH_SIZE="${PAIRUAV_EXTRACT_BATCH_SIZE:-16}"
WORKERS="${PAIRUAV_WORKERS:-8}"
FEATURE_DEVICE="${PAIRUAV_FEATURE_DEVICE:-auto}"
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

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: $name" >&2
    exit 2
  fi
}

require_file "$TRAIN_INDEX"
require_file "$VAL_INDEX"

ANGLE_CONFIG="${PAIRUAV_ANGLE_CONFIG:-$ROOT/configs/lamp/angle_s0.json}"
RANGE_CONFIG="${PAIRUAV_RANGE_CONFIG:-$ROOT/configs/lamp/range_ab_relsmooth.json}"
require_file "$ANGLE_CONFIG"
require_file "$RANGE_CONFIG"

config_name() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["name"])' "$1"
}
ANGLE_NAME="$(config_name "$ANGLE_CONFIG")"
RANGE_NAME="$(config_name "$RANGE_CONFIG")"

PRECOMPUTED_TRAIN="${PAIRUAV_TRAIN_CACHE:-}"
PRECOMPUTED_VAL="${PAIRUAV_VAL_CACHE:-}"
if [[ -n "$PRECOMPUTED_TRAIN" || -n "$PRECOMPUTED_VAL" ]]; then
  if [[ -z "$PRECOMPUTED_TRAIN" || -z "$PRECOMPUTED_VAL" ]]; then
    echo "PAIRUAV_TRAIN_CACHE and PAIRUAV_VAL_CACHE must be set together" >&2
    exit 2
  fi
  TRAIN_CACHE="$PRECOMPUTED_TRAIN"
  VAL_CACHE="$PRECOMPUTED_VAL"
else
  require_env PAIRUAV_ALL_TRAIN_JSON
  require_env PAIRUAV_TRAIN_IMAGES
  require_env VGGT_WEIGHT

  TRAIN_JSON="$RUN_DIR/splits/train"
  VAL_JSON="$RUN_DIR/splits/val"
  CACHE_ROOT="$RUN_DIR/cache"
  TRAIN_CACHE="$CACHE_ROOT/train_nfull_s${IMAGE_SIZE}"
  VAL_CACHE="$CACHE_ROOT/val_nfull_s${IMAGE_SIZE}"

  cd "$ROOT"
  "$PYTHON_BIN" -m pairuav.index materialize \
    --index "$TRAIN_INDEX" \
    --source-json-dir "$PAIRUAV_ALL_TRAIN_JSON" \
    --out-json-dir "$TRAIN_JSON" \
    --name train

  "$PYTHON_BIN" -m pairuav.index materialize \
    --index "$VAL_INDEX" \
    --source-json-dir "$PAIRUAV_ALL_TRAIN_JSON" \
    --out-json-dir "$VAL_JSON" \
    --name val

  "$PYTHON_BIN" -m pairuav.features \
    --train-json-dir "$TRAIN_JSON" \
    --val-json-dir "$VAL_JSON" \
    --image-dir "$PAIRUAV_TRAIN_IMAGES" \
    --vggt-weight "$VGGT_WEIGHT" \
    --cache-root "$CACHE_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --extract-batch-size "$EXTRACT_BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
fi

cd "$ROOT"
"$PYTHON_BIN" -m pairuav.index verify-cache --index "$TRAIN_INDEX" --cache-dir "$TRAIN_CACHE"
"$PYTHON_BIN" -m pairuav.index verify-cache --index "$VAL_INDEX" --cache-dir "$VAL_CACHE"

printf '%s\n' \
  "train_index=$TRAIN_INDEX" \
  "val_index=$VAL_INDEX" \
  "train_cache=$TRAIN_CACHE" \
  "val_cache=$VAL_CACHE" \
  "angle_config=$ANGLE_CONFIG" \
  "range_config=$RANGE_CONFIG" \
  "seed=$SEED" \
  "device=$DEVICE" \
  > "$RUN_DIR/workflow.env"

GEOMETRY_DIR="$RUN_DIR/geometry"
TRAIN_GEOM="$GEOMETRY_DIR/train_geometry.npz"
VAL_GEOM="$GEOMETRY_DIR/val_geometry.npz"
"$PYTHON_BIN" -m pairuav.geometry --cache-dir "$TRAIN_CACHE" --out "$TRAIN_GEOM"
"$PYTHON_BIN" -m pairuav.geometry --cache-dir "$VAL_CACHE" --out "$VAL_GEOM"

ANGLE_ROOT="$RUN_DIR/heads/angle"
RANGE_ROOT="$RUN_DIR/heads/range"
"$PYTHON_BIN" -m pairuav.train_angle \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --train-geom "$TRAIN_GEOM" \
  --val-geom "$VAL_GEOM" \
  --config "$ANGLE_CONFIG" \
  --run-root "$ANGLE_ROOT" \
  --no-timestamp-run-root \
  --device "$DEVICE" \
  --seed "$SEED"

"$PYTHON_BIN" -m pairuav.train_range \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --config "$RANGE_CONFIG" \
  --output-dir "$RANGE_ROOT" \
  --feature-device "$FEATURE_DEVICE" \
  --device "$DEVICE" \
  --seed "$SEED"

ANGLE_CKPT="$ANGLE_ROOT/$ANGLE_NAME/head_best_angle.pt"
RANGE_CKPT="$RANGE_ROOT/$RANGE_NAME/range_head_best_distance.pt"
METRICS="$RUN_DIR/metrics.json"
"$PYTHON_BIN" -m pairuav.eval_val \
  --val-cache "$VAL_CACHE" \
  --val-geom "$VAL_GEOM" \
  --angle-ckpt "$ANGLE_CKPT" \
  --range-ckpt "$RANGE_CKPT" \
  --out "$METRICS" \
  --device "$DEVICE" \
  --seed "$SEED"

printf '%s\n' \
  "angle_checkpoint=$ANGLE_CKPT" \
  "range_checkpoint=$RANGE_CKPT" \
  "metrics=$METRICS" \
  >> "$RUN_DIR/workflow.env"

echo "LaMP training workflow completed under: $RUN_DIR"
echo "Metrics: $METRICS"
