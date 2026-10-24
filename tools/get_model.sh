#!/usr/bin/env bash
# Fetches the YOLO26n-pose weights from Ultralytics and exports them to ONNX for OpenCV 5 cv2.dnn.
# The weights are NOT included in this repository: they are licensed by Ultralytics under AGPL-3.0
# (or an Ultralytics Enterprise licence). Running this script means you accept Ultralytics' terms.
# Output: models/yolo26n-pose.onnx (imgsz 640), the file FallLoop loads at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
python3 -m venv .venv-export
. .venv-export/bin/activate
pip install --quiet "ultralytics==8.4.181" onnx onnxslim
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO("yolo26n-pose.pt")             # downloaded from Ultralytics on first use
p = m.export(format="onnx", imgsz=640, simplify=True)
import shutil; shutil.move(p, "models/yolo26n-pose.onnx"); print("wrote models/yolo26n-pose.onnx")
PY
deactivate
