import os
import cv2
import numpy as np
import onnxruntime as ort
from core.database import GLOBAL_PET_CACHE

class AIPipeline:
    """
    Day 1 MVP Pipeline AI Models (YOLO Nano & OSNet).
    Utilizes ONNX Runtime for high-speed, lightweight inference.
    """
    def __init__(self, yolo_path="models/yolo26nano.onnx", osnet_path="models/osnet_x1_0.onnx"):
        self.yolo_path = yolo_path
        self.osnet_path = osnet_path
        
        # Verify model files exist to prevent silent failures
        if not os.path.exists(self.yolo_path):
            print(f"WARNING: YOLO model not found at {self.yolo_path}. Inference will be bypassed.")
            self.yolo_session = None
        else:
            self.yolo_session = ort.InferenceSession(self.yolo_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.yolo_input_name = self.yolo_session.get_inputs()[0].name
            
        if not os.path.exists(self.osnet_path):
            print(f"WARNING: OSNet model not found at {self.osnet_path}. Inference will be bypassed.")
            self.osnet_session = None
        else:
            self.osnet_session = ort.InferenceSession(self.osnet_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.osnet_input_name = self.osnet_session.get_inputs()[0].name

    def preprocess_yolo(self, frame):
        """Prepares the frame for YOLOv8/11 (640x640, float32, normalized)."""
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0 # Normalize 0-1
        return img

    def postprocess_yolo(self, outputs, orig_w, orig_h):
        """
        Parses YOLO output, handling both standard [1, 84, 8400] and NMS-free [1, 300, 6] (YOLO26).
        Filters for COCO Class ID 16 (Dog) and applies scaling.
        """
        DOG_CLASS_ID = 16
        CONFIDENCE_THRESHOLD = 0.5
        
        boxes = []
        confidences = []
        
        # Check if the output is NMS-free format (YOLOv10 / YOLO26)
        # NMS-free models output shape: (1, 300, 6) -> [left, top, right, bottom, confidence, class]
        if outputs.shape[-1] == 6 and len(outputs.shape) == 3:
            predictions = np.squeeze(outputs) # shape: (300, 6)
            for row in predictions:
                x1, y1, x2, y2, confidence, class_id = row
                
                if int(class_id) == DOG_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    # Scale back to original image dimensions
                    x_scale = orig_w / 640.0
                    y_scale = orig_h / 640.0
                    
                    x_min = int(x1 * x_scale)
                    y_min = int(y1 * y_scale)
                    x_max = int(x2 * x_scale)
                    y_max = int(y2 * y_scale)
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence))
                    
            # For NMS-free models, we don't need to run cv2.dnn.NMSBoxes, but we can structure the output directly
            final_detections = []
            for i in range(len(boxes)):
                final_detections.append({
                    "bbox": boxes[i],
                    "confidence": confidences[i]
                })
            return final_detections
            
        else:
            # Standard YOLO output shape: (1, 84, 8400)
            predictions = np.squeeze(outputs).T # shape: (8400, 84)
            
            # Parse output grid
            for row in predictions:
                classes_scores = row[4:]
                class_id = np.argmax(classes_scores)
                confidence = classes_scores[class_id]
                
                if class_id == DOG_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    # YOLO outputs center_x, center_y, width, height
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    
                    # Scale back to original image dimensions
                    x_scale = orig_w / 640.0
                    y_scale = orig_h / 640.0
                    
                    cx *= x_scale
                    cy *= y_scale
                    w *= x_scale
                    h *= y_scale
                    
                    # Convert to [x_min, y_min, x_max, y_max]
                    x_min = int(cx - (w / 2))
                    y_min = int(cy - (h / 2))
                    x_max = int(cx + (w / 2))
                    y_max = int(cy + (h / 2))
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence))
                    
            # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
            
            final_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append({
                        "bbox": boxes[i],
                        "confidence": confidences[i]
                    })
                    
            return final_detections

    def preprocess_osnet(self, cropped_img):
        """
        Prepares the cropped dog image for OSNet (256x128).
        Applies standard ImageNet normalization.
        """
        img = cv2.resize(cropped_img, (128, 256)) # W=128, H=256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0)  # BCHW
        return img

    def run_inference(self, frame):
        """
        Executes the 4-step processing pipeline.
        """
        if self.yolo_session is None or self.osnet_session is None:
            # Fallback to prevent crash if models are not loaded
            return []
            
        h, w = frame.shape[:2]
        
        # Step 1: Image Downscaling & Preprocessing
        yolo_input = self.preprocess_yolo(frame)
        
        # Step 2: Detection and Tracking (YOLO Nano)
        yolo_outputs = self.yolo_session.run(None, {self.yolo_input_name: yolo_input})[0]
        detections = self.postprocess_yolo(yolo_outputs, w, h)
        
        results = []
        tracking_id_counter = 1 # Simple mock ID incrementer for MVP
        
        # Step 3: Dynamic Cropping & Aspect-Ratio Normalization
        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            yolo_conf = det["confidence"]
            
            # Boundary checks
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
                
            cropped_dog_patch = frame[y_min:y_max, x_min:x_max]
            osnet_input = self.preprocess_osnet(cropped_dog_patch)
            
            # Step 4: Metric Feature Extraction (OSNet)
            osnet_outputs = self.osnet_session.run(None, {self.osnet_input_name: osnet_input})[0]
            live_vector = osnet_outputs.flatten() # 512-dim vector
            
            # Normalize vector for Cosine Similarity
            live_vector_norm = live_vector / np.linalg.norm(live_vector)
            
            # Deep Metric Learning Math & Thresholding
            best_match = "Unknown Dog"
            best_score = 0.0
            
            for pet_name, db_vector in GLOBAL_PET_CACHE.items():
                db_vector_norm = db_vector / np.linalg.norm(db_vector)
                similarity = np.dot(live_vector_norm, db_vector_norm)
                
                if similarity > best_score:
                    best_score = similarity
                    if similarity >= 0.85: # Threshold from specification
                        best_match = pet_name
                        
            results.append({
                "tracking_id": tracking_id_counter,
                "name": best_match,
                "confidence": float(yolo_conf), # Return YOLO confidence for the bbox
                "bbox": [x_min, y_min, x_max, y_max]
            })
            tracking_id_counter += 1
            
            # Memory Management Guardrails
            del cropped_dog_patch
            del osnet_input
            
        del yolo_input
        return results
