# Anti-Dart ONNX Exporter

Standalone exporter for the deployed anti-dart YOLO ONNX format.

It reproduces the export chain used for the latest known anti-dart ONNX:

- source checkpoint: `best.pt`
- base export: Ultralytics ONNX, FP32, `imgsz=384,640`, `opset=17`, `nms=False`
- graph rewrite: expose raw head output at `/model.23/Transpose`
- input rewrite: `UINT8 NHWC [1,384,640,3]` named `images_u8`
- preprocessing inside ONNX: cast to FP32, transpose to NCHW, divide by 255
- final cleanup: `onnxsim`

## One-command export

```bash
./run_export.sh
```

By default this reads:

```text
/home/jerryzhu/workspace/train_workstation/projects/anti_dart_noP2_yolo26n_0207/runs/detect/runs/rm_dart/yolo26n_dart_4090_02072/weights/best.pt
```

and writes:

```text
./yolo26n_dart_u8_nhwc_notopk_fp32.onnx
```

## Export another checkpoint

```bash
./run_export.sh /path/to/best.pt /path/to/output.onnx
```

or:

```bash
MODEL_PATH=/path/to/best.pt OUTPUT_PATH=./anti_dart.onnx ./run_export.sh
```

## Environment

Use the same Python environment that was used for the original exporter if
available:

```bash
conda activate anti-dart-data
./run_export.sh
```

Required Python packages are listed in `requirements.txt`.
