import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path, output_path):
    print(f"Quantizing {input_path} to {output_path} (INT16)...")
    try:
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt16
        )
        print(f"Successfully quantized {input_path}")
    except Exception as e:
        print(f"Failed to quantize {input_path}: {e}")

if __name__ == "__main__":
    quantize_model("models/yolo26nano.onnx", "models/yolo26nano_int16.onnx")
    quantize_model("models/osnet_x1_0.onnx", "models/osnet_x1_0_int16.onnx")
