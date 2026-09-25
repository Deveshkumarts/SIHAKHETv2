#!/usr/bin/env bash
# Build TensorRT engines from the exported ONNX models ON the Jetson Orin NX (engines are device-specific).
#
#   1. On the dev machine:   python scripts/export_onnx.py --models ae lstm yolo
#   2. Copy weights/onnx/*.onnx (+ weights/conv_ae_normal.calibration.json) to the Jetson.
#   3. On the Jetson:        bash deploy/jetson/build_engines.sh [fp16|int8]
#
# STATUS: written from the TensorRT/trtexec documentation. It has NOT been executed - the development machine has
# no NVIDIA GPU or Jetson. Validate on the target device before relying on it.
set -euo pipefail

PRECISION="${1:-fp16}"                       # fp16 | int8
ONNX_DIR="${ONNX_DIR:-weights/onnx}"
OUT_DIR="${OUT_DIR:-weights/trt}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"      # JetPack default location
mkdir -p "$OUT_DIR"

case "$PRECISION" in
  fp16) PREC_FLAGS="--fp16" ;;
  int8) PREC_FLAGS="--int8 --fp16" ;;         # int8 with fp16 fallback for layers that cannot be quantised
  *) echo "precision must be fp16 or int8"; exit 2 ;;
esac

build() {  # name, extra flags
  local name="$1"; shift
  echo ">>> building $name ($PRECISION)"
  "$TRTEXEC" --onnx="$ONNX_DIR/$name.onnx" --saveEngine="$OUT_DIR/${name}_${PRECISION}.engine" \
             --memPoolSize=workspace:"${WORKSPACE_MB}" $PREC_FLAGS "$@" 2>&1 | tee "$OUT_DIR/${name}_${PRECISION}.log" | tail -n 25
}

# YOLO11-Seg: DYNAMIC height/width. Do not build this with a fixed 256x256 shape: measured on the test split, square-padded
# inference scores mAP@50:95 0.824 vs 0.896 for minimal-padding rectangular inference (these crops are non-square), and
# a static engine would silently run in the worse mode. The ONNX is exported with dynamic=True for this reason.
build yolo11n_seg --minShapes=images:1x3x32x32 --optShapes=images:1x3x224x256 --maxShapes=images:1x3x256x256
# Conv autoencoder: dynamic batch of ROI patches (1..64)
build conv_ae --minShapes=patch:1x1x64x64 --optShapes=patch:16x1x64x64 --maxShapes=patch:64x1x64x64
# LSTM tracker: dynamic batch of tracks (1..64), 8 steps x 9 features
build lstm_tracker --minShapes=history:1x8x9 --optShapes=history:16x8x9 --maxShapes=history:64x8x9

cat <<EOF

Engines written to $OUT_DIR. INT8 notes:
  * trtexec --int8 without a calibration cache uses dummy scales and is only good for latency measurement.
  * For accuracy-preserving INT8 build a calibration cache from REAL sonar frames (entropy calibrator) and pass
    --calib=<cache>; for YOLO the simplest route is:  yolo export model=<best.pt> format=engine int8=True data=<data.yaml>
  * Re-run  python -m models.yolo11_seg.evaluate  against the engine and compare mAP with the FP32 numbers before
    accepting INT8 (the autoencoder threshold must be re-validated too: reconstruction error shifts under INT8).
EOF
