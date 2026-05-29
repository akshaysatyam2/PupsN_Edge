import cv2
import numpy as np
from database import GLOBAL_PET_CACHE

class MockAIPipeline:
    """
    Robust stub for the Day 1 MVP Pipeline AI Models (YOLO Nano & OSNet).
    In a real scenario, this loads ONNX models. Here we simulate the logic.
    """
    def __init__(self):
        # Would init ONNX runtime sessions here
        pass

    def run_inference(self, frame):
        """
        Runs the 4-step processing pipeline.
        Returns a list of detections: {"tracking_id": int, "name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
        """
        # Step 1: Image Downscaling
        downscaled_frame = cv2.resize(frame, (640, 640))
        
        # Step 2: Detection and Tracking (Mock YOLOv8)
        # We'll just fake a detection in the center of the frame for demonstration.
        h, w = frame.shape[:2]
        bbox = [int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)]
        tracking_id = 1
        
        # Step 3: Dynamic Cropping & Aspect-Ratio Normalization
        x_min, y_min, x_max, y_max = bbox
        # Avoid out of bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        if x_max > x_min and y_max > y_min:
            cropped_dog_patch = frame[y_min:y_max, x_min:x_max]
            cropped_dog_patch = cv2.resize(cropped_dog_patch, (128, 256)) # 256x128 aspect ratio (W=128, H=256) per spec
            
            # Step 4: Metric Feature Extraction (Mock OSNet)
            # Produces a 512-dim vector
            live_vector = np.random.rand(512).astype(np.float32)
            # Normalize to match Cosine Similarity formula
            live_vector = live_vector / np.linalg.norm(live_vector)
            
            # Match against DB Cache
            best_match = "Unknown Dog"
            best_score = 0.0
            
            for pet_name, db_vector in GLOBAL_PET_CACHE.items():
                db_vector_norm = db_vector / np.linalg.norm(db_vector)
                similarity = np.dot(live_vector, db_vector_norm)
                
                if similarity > best_score:
                    best_score = similarity
                    if similarity >= 0.85:
                        best_match = pet_name
            
            # If our mock DB is empty, let's just make it Unknown Dog with mock conf
            if not GLOBAL_PET_CACHE:
                best_score = 0.54 # Random mock confidence
                
            # Cleanup step 1 arrays to prevent memory leaks as per spec
            del downscaled_frame
            del cropped_dog_patch
            
            return [{
                "tracking_id": tracking_id,
                "name": best_match,
                "confidence": float(best_score),
                "bbox": bbox
            }]
            
        del downscaled_frame
        return []
