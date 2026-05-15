#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_MODEL="/home/jerryzhu/workspace/train_workstation/projects/anti_dart_noP2_yolo26n_0207/runs/detect/runs/rm_dart/yolo26n_dart_4090_02072/weights/best.pt"
DEFAULT_OUTPUT="${SCRIPT_DIR}/yolo26n_dart_u8_nhwc_notopk_fp32.onnx"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT}"
IMGSZ="${IMGSZ:-384,640}"
NUM_CLASSES="${NUM_CLASSES:-2}"
TARGET_NODE="${TARGET_NODE:-/model.23/Transpose}"
INPUT_NAME="${INPUT_NAME:-images_u8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./run_export.sh [model.pt] [output.onnx]

Environment overrides:
  MODEL_PATH    input checkpoint, default is the latest known yolo26n best.pt
  OUTPUT_PATH   output ONNX path, default is ./yolo26n_dart_u8_nhwc_notopk_fp32.onnx
  IMGSZ         export image size as H,W, default 384,640
  NUM_CLASSES   class count, default 2
  TARGET_NODE   raw-head node to expose, default /model.23/Transpose
  INPUT_NAME    ONNX input name, default images_u8
  PYTHON_BIN    Python executable, default python3

Examples:
  ./run_export.sh
  ./run_export.sh /path/to/best.pt /tmp/anti_dart.onnx
  MODEL_PATH=/path/to/new/best.pt OUTPUT_PATH=./new.onnx ./run_export.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
fi

if [[ $# -ge 2 ]]; then
  OUTPUT_PATH="$2"
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/export_anti_dart_onnx.py" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH" \
  --imgsz "$IMGSZ" \
  --num-classes "$NUM_CLASSES" \
  --target-node "$TARGET_NODE" \
  --input-name "$INPUT_NAME"
