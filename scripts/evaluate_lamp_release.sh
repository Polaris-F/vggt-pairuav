#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_lamp_release.sh [ASSET_ROOT] [OUTPUT_DIR]

Evaluate the three released LaMP seeds on the released val_quick_2048
feature cache. This does not rerun VGGT or retrain a head.

Defaults:
  ASSET_ROOT  artifacts/lamp_ours_3seed
  OUTPUT_DIR  outputs/lamp_release_eval_<timestamp>

Environment overrides:
  PAIRUAV_DEVICE  PyTorch device (default: cuda)
  PYTHON_BIN      Python executable (default: python)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${1:-$ROOT/artifacts/lamp_ours_3seed}"
OUTPUT_DIR="${2:-$ROOT/outputs/lamp_release_eval_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${PAIRUAV_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VAL_CACHE="$ASSET_ROOT/validation/val_quick_2048"
VAL_GEOM="$VAL_CACHE/geometry.npz"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

require_file "$ASSET_ROOT/MANIFEST.sha256"
require_file "$VAL_CACHE/features.npy"
require_file "$VAL_CACHE/json_paths.json"
require_file "$VAL_CACHE/meta.json"
require_file "$VAL_GEOM"

for seed in 2026 2027 2028; do
  require_file "$ASSET_ROOT/weights/seed${seed}/angle/S0_rich_noc/head_best_angle.pt"
  require_file "$ASSET_ROOT/weights/seed${seed}/angle/S0_rich_noc/config.json"
  require_file "$ASSET_ROOT/weights/seed${seed}/range/R_ab_relsmooth/range_head_best_distance.pt"
  require_file "$ASSET_ROOT/weights/seed${seed}/range/R_ab_relsmooth/config.json"
done

(
  cd "$ASSET_ROOT"
  sha256sum -c MANIFEST.sha256
)

cd "$ROOT"
"$PYTHON_BIN" -m pairuav.index verify-cache \
  --index data_index/val_quick_2048.txt \
  --cache-dir "$VAL_CACHE"

mkdir -p "$OUTPUT_DIR"
printf '%s\n' \
  "asset_root=$ASSET_ROOT" \
  "validation_cache=$VAL_CACHE" \
  "validation_geometry=$VAL_GEOM" \
  "device=$DEVICE" \
  > "$OUTPUT_DIR/workflow.env"

result_files=()
for seed in 2026 2027 2028; do
  angle_dir="$ASSET_ROOT/weights/seed${seed}/angle/S0_rich_noc"
  range_dir="$ASSET_ROOT/weights/seed${seed}/range/R_ab_relsmooth"
  result="$OUTPUT_DIR/seed${seed}.json"
  "$PYTHON_BIN" -m pairuav.eval_val \
    --val-cache "$VAL_CACHE" \
    --val-geom "$VAL_GEOM" \
    --angle-ckpt "$angle_dir/head_best_angle.pt" \
    --range-ckpt "$range_dir/range_head_best_distance.pt" \
    --out "$result" \
    --device "$DEVICE" \
    --seed "$seed" \
    --angle-matmul-precision highest \
    --range-matmul-precision highest
  result_files+=("$result")
done

"$PYTHON_BIN" scripts/summarize_release_metrics.py \
  --results "${result_files[@]}" \
  --system range_C \
  --out-json "$OUTPUT_DIR/summary.json" \
  --out-md "$OUTPUT_DIR/summary.md"

echo "LaMP release evaluation completed: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/summary.md"
