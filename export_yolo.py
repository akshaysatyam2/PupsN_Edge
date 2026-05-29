import os
import shutil
from ultralytics import YOLO

print("Exporting YOLO26 Nano...")
model = YOLO('yolo26n.pt')
model.export(format='onnx', imgsz=640)

if not os.path.exists('models'):
    os.makedirs('models')

if os.path.exists('yolo26n.onnx'):
    shutil.move('yolo26n.onnx', 'models/yolo26nano.onnx')
    print("Successfully exported YOLO26 Nano to models/yolo26nano.onnx")
else:
    print("Export failed: yolo26n.onnx not found.")
