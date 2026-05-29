import os
import time
import base64
import gc
import threading
import sqlite3
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_PATH = 'pupsn_core.db'
# GLOBAL_SCOPED_CACHE = { camera_id: { "Pet_Name": np.ndarray([v1...v512]) } }
GLOBAL_SCOPED_CACHE = {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 1. camera_config
    c.execute('''
        CREATE TABLE IF NOT EXISTS camera_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_name TEXT NOT NULL,
            stream_url TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 2. pet_profiles
    c.execute('''
        CREATE TABLE IF NOT EXISTS pet_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            pet_name TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(camera_id) REFERENCES camera_config(id) ON DELETE CASCADE
        )
    ''')
    # 3. detection_logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            camera_id INTEGER NOT NULL,
            tracking_id INTEGER NOT NULL,
            identified_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            inference_latency_ms REAL NOT NULL
        )
    ''')
    
    # Insert a default webcam if table is empty for easy testing
    c.execute("SELECT COUNT(*) FROM camera_config")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO camera_config (stream_name, stream_url) VALUES (?, ?)", ("Default Webcam", "0"))
        
    conn.commit()
    conn.close()

def load_cache():
    global GLOBAL_SCOPED_CACHE
    GLOBAL_SCOPED_CACHE.clear()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM camera_config")
    cameras = c.fetchall()
    
    for cam in cameras:
        cam_id = cam[0]
        GLOBAL_SCOPED_CACHE[cam_id] = {}
        c.execute("SELECT pet_name, vector_blob FROM pet_profiles WHERE camera_id = ?", (cam_id,))
        profiles = c.fetchall()
        for p in profiles:
            pet_name = p[0]
            vector_blob = p[1]
            # Deserialize FP32 array
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            GLOBAL_SCOPED_CACHE[cam_id][pet_name] = vector
            
    conn.close()

# Mock AI Models
def mock_yolo_count_dogs(image_bgr):
    """
    Mock YOLO Nano. Simulates counting dogs. 
    Returns exactly 1 dog to satisfy the validation step, plus a bounding box.
    """
    # Using 10% padding for a mock bounding box
    h, w = image_bgr.shape[:2]
    return 1, (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))

def mock_osnet_extract(cropped_img):
    """
    Mock OSNet. Extracts a 512-dimension FP32 embedding.
    """
    vec = np.random.rand(512).astype(np.float32)
    # Normalize the vector
    vec /= np.linalg.norm(vec)
    return vec

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cameras', methods=['GET'])
def get_cameras():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, stream_name, stream_url, is_active FROM camera_config")
    cams = [{"id": row[0], "name": row[1], "url": row[2], "active": row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(cams)


@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.json
    name = data.get('stream_name')
    url = data.get('stream_url')
    if not name or not url:
        return jsonify({"error": "Missing name or url"}), 400
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO camera_config (stream_name, stream_url) VALUES (?, ?)", (name, url))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    
    global GLOBAL_SCOPED_CACHE
    if new_id not in GLOBAL_SCOPED_CACHE:
        GLOBAL_SCOPED_CACHE[new_id] = {}
        
    # Start thread dynamically (for continuous scaling)
    worker_manager.start_camera_thread(new_id, url)
        
    return jsonify({"success": True, "camera_id": new_id})


@app.route('/register_pet', methods=['POST'])
def register_pet():
    """ 1-DOG PHOTO REGISTRATION GATEKEEPER """
    camera_id = request.form.get('camera_id', type=int)
    pet_name = request.form.get('pet_name')
    file = request.files.get('image')
    
    if not camera_id or not pet_name or not file:
        return jsonify({"error": "Missing required fields"}), 400
        
    # Read the image file directly from memory into OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format."}), 400
        
    # 1. Class-Count Verification
    dog_count, bbox = mock_yolo_count_dogs(img)
    
    # 2. The 1-Dog Rule
    if dog_count != 1:
        return jsonify({"error": "Registration failed: The uploaded image must contain exactly one dog."}), 400
        
    # 3. Crop & Extract
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    # Resize it to 256x128 (width=128, height=256)
    crop_resized = cv2.resize(crop, (128, 256))
    
    # Extract 512-dimension vector
    vector = mock_osnet_extract(crop_resized)
    vector_blob = vector.tobytes()
    
    # Commit binary blob to DB under selected camera_id
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO pet_profiles (camera_id, pet_name, vector_blob) VALUES (?, ?, ?)",
              (camera_id, pet_name, vector_blob))
    conn.commit()
    conn.close()
    
    # 4. Cache Sync without restarting
    global GLOBAL_SCOPED_CACHE
    if camera_id not in GLOBAL_SCOPED_CACHE:
        GLOBAL_SCOPED_CACHE[camera_id] = {}
    GLOBAL_SCOPED_CACHE[camera_id][pet_name] = vector
    
    # Memory Management
    del img
    del crop
    del crop_resized
    gc.collect()
    
    return jsonify({"success": True, "message": f"{pet_name} registered successfully."})


class BackgroundWorkerManager:
    """ Persistent Scope-Isolated Background Stream Engine """
    def __init__(self):
        self.running = True
        self.threads = {}

    def start_camera_thread(self, camera_id, stream_url):
        if camera_id in self.threads:
            return # Already running
            
        def worker():
            backoff_intervals = [5, 10, 15, 30]
            backoff_idx = 0
            
            # Cast stream_url to int if it's purely digits (USB cams)
            target = int(stream_url) if stream_url.isdigit() else stream_url
            
            while self.running:
                cap = cv2.VideoCapture(target)
                
                if not cap.isOpened():
                    socketio.emit('stream_update', {"camera_id": camera_id, "status": "inactive"})
                    sleep_time = backoff_intervals[backoff_idx]
                    time.sleep(sleep_time)
                    if backoff_idx < len(backoff_intervals) - 1:
                        backoff_idx += 1
                    continue
                
                # Successful connection, reset backoff
                backoff_idx = 0
                last_ai_time = 0.0
                ai_interval = 1.0 # 1,000 milliseconds clock gate
                inference_latency_ms = 0.0
                
                # Mock Tracking state across frames
                trackers = {1: {"name": "Unknown Dog", "bbox": [150, 90, 380, 520]}}
                
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        # Dropped stream, break to hit reconnection logic
                        break 
                        
                    # Hard-resize to 640x640 to conserve memory
                    frame = cv2.resize(frame, (640, 640))
                    
                    current_time = time.time()
                    
                    # 1-FPS Scoped Inference
                    if current_time - last_ai_time >= ai_interval:
                        start_inference = time.time()
                        
                        cache_for_cam = GLOBAL_SCOPED_CACHE.get(camera_id, {})
                        
                        for tid, obj in list(trackers.items()):
                            if obj["name"] == "Unknown Dog":
                                # Mock extracting embedding for this crop
                                crop_vec = mock_osnet_extract(frame)
                                
                                best_match = None
                                highest_sim = 0.0
                                
                                # Isolate matching against only THIS camera's cache
                                for p_name, p_vec in cache_for_cam.items():
                                    sim = cosine_similarity(crop_vec, p_vec)
                                    if sim > highest_sim:
                                        highest_sim = sim
                                        best_match = p_name
                                        
                                if highest_sim >= 0.85 and best_match:
                                    obj["name"] = best_match
                                elif cache_for_cam and np.random.rand() > 0.8:
                                    # MOCK SIMULATION: Forcefully resolve sometimes so UI can be demonstrated 
                                    # since purely random vectors almost never hit 0.85 cosine similarity.
                                    best_match = list(cache_for_cam.keys())[0]
                                    obj["name"] = best_match
                                    highest_sim = 0.86
                                    
                                # Log detection to SQLite
                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                c.execute('''
                                    INSERT INTO detection_logs (camera_id, tracking_id, identified_name, confidence, inference_latency_ms)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (camera_id, tid, obj["name"], float(highest_sim), inference_latency_ms))
                                conn.commit()
                                conn.close()
                                
                        last_ai_time = time.time()
                        inference_latency_ms = (last_ai_time - start_inference) * 1000
                    
                    # Telemetry Payload Preparation
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    detections = []
                    for tid, obj in trackers.items():
                        detections.append({
                            "tracking_id": tid,
                            "name": obj["name"],
                            "confidence": 0.92,
                            "bbox": obj["bbox"]
                        })
                        
                    payload = {
                        "camera_id": camera_id,
                        "status": "active",
                        "image": f"data:image/jpeg;base64,{img_str}",
                        "metrics": {
                            "inference_latency_ms": round(inference_latency_ms, 2),
                            "system_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        },
                        "detections": detections
                    }
                    socketio.emit('stream_update', payload)
                    
                    # Explicit Memory Management
                    del frame
                    del buffer
                    gc.collect()
                    
                    # Restrict frame pull rate slightly to avoid massive CPU pegging
                    time.sleep(0.03) 
                    
                cap.release()
                
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.threads[camera_id] = t

worker_manager = BackgroundWorkerManager()

def start_background_workers():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, stream_url FROM camera_config WHERE is_active=1")
    cams = c.fetchall()
    conn.close()
    
    for cam_id, stream_url in cams:
        worker_manager.start_camera_thread(cam_id, stream_url)

if __name__ == '__main__':
    init_db()
    load_cache()
    # Boot persistent engine with Flask
    start_background_workers()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
