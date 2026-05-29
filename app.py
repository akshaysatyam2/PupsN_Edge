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
import logging

logging.basicConfig(level=logging.INFO)

from core.inference import AIPipeline
from core.camera import VideoStream

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_PATH = 'pupsn_core.db'
# GLOBAL_SCOPED_CACHE = { camera_id: { "Pet_Name": np.ndarray([v1...v512]) } }
GLOBAL_SCOPED_CACHE = {}

# Global AI Pipeline instance
ai_pipeline = AIPipeline()

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
            if pet_name not in GLOBAL_SCOPED_CACHE[cam_id]:
                GLOBAL_SCOPED_CACHE[cam_id][pet_name] = []
            GLOBAL_SCOPED_CACHE[cam_id][pet_name].append(vector)
            
    conn.close()


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


@app.route('/delete_camera/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    # Completely stop stream and inference thread before db operation
    worker_manager.stop_camera_thread(camera_id)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM camera_config WHERE id=?", (camera_id,))
    c.execute("DELETE FROM pet_profiles WHERE camera_id=?", (camera_id,))
    conn.commit()
    conn.close()
    
    global GLOBAL_SCOPED_CACHE
    if camera_id in GLOBAL_SCOPED_CACHE:
        del GLOBAL_SCOPED_CACHE[camera_id]
        
    return jsonify({"success": True})


@app.route('/pets', methods=['GET'])
def get_pets():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT p.camera_id, c.stream_name, p.pet_name, COUNT(p.id) as photo_count
        FROM pet_profiles p
        JOIN camera_config c ON p.camera_id = c.id
        GROUP BY p.camera_id, p.pet_name
    ''')
    pets = [{"camera_id": row[0], "camera_name": row[1], "pet_name": row[2], "photo_count": row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(pets)


@app.route('/delete_pet/<int:camera_id>/<pet_name>', methods=['DELETE'])
def delete_pet(camera_id, pet_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM pet_profiles WHERE camera_id=? AND pet_name=?", (camera_id, pet_name))
    conn.commit()
    conn.close()
    
    global GLOBAL_SCOPED_CACHE
    if camera_id in GLOBAL_SCOPED_CACHE and pet_name in GLOBAL_SCOPED_CACHE[camera_id]:
        del GLOBAL_SCOPED_CACHE[camera_id][pet_name]
        
    return jsonify({"success": True})


@app.route('/register_pet', methods=['POST'])
def register_pet():
    """ MULTI-PHOTO REGISTRATION GATEKEEPER """
    camera_id = request.form.get('camera_id', type=int)
    pet_name = request.form.get('pet_name')
    files = request.files.getlist('images')
    
    if not camera_id or not pet_name or not files or len(files) == 0:
        return jsonify({"error": "Missing required fields or images"}), 400
        
    if len(files) > 5:
        return jsonify({"error": "Maximum of 5 images allowed per pet."}), 400
        
    if ai_pipeline.yolo_session is None or ai_pipeline.osnet_session is None:
        return jsonify({"error": "Models are not loaded on backend."}), 500

    extracted_vectors = []
    failed_count = 0
    
    for file in files:
        if file.filename == '':
            continue
            
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            failed_count += 1
            continue
            
        h, w = img.shape[:2]
        yolo_input = ai_pipeline.preprocess_yolo(img)
        yolo_outputs = ai_pipeline.yolo_session.run(None, {ai_pipeline.yolo_input_name: yolo_input})[0]
        detections = ai_pipeline.postprocess_yolo(yolo_outputs, w, h)
        
        # 2. The 1-Dog Rule per uploaded image
        if len(detections) != 1:
            failed_count += 1
            continue
            
        # 3. Crop & Extract using real OSNet model
        x_min, y_min, x_max, y_max = detections[0]["bbox"]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        crop = img[y_min:y_max, x_min:x_max]
        osnet_input = ai_pipeline.preprocess_osnet(crop)
        
        osnet_outputs = ai_pipeline.osnet_session.run(None, {ai_pipeline.osnet_input_name: osnet_input})[0]
        vector = osnet_outputs.flatten()
        
        # Normalize the vector
        vector /= np.linalg.norm(vector)
        extracted_vectors.append(vector)
        
        # Cleanup loop memory
        del img
        del crop
        del yolo_input
        del osnet_input
    
    if len(extracted_vectors) == 0:
        return jsonify({"error": f"Failed to register. All {len(files)} uploaded images were invalid (e.g., no dog found, or multiple dogs found)."}), 400
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 4. Cache Sync without restarting
    global GLOBAL_SCOPED_CACHE
    if camera_id not in GLOBAL_SCOPED_CACHE:
        GLOBAL_SCOPED_CACHE[camera_id] = {}
    if pet_name not in GLOBAL_SCOPED_CACHE[camera_id]:
        GLOBAL_SCOPED_CACHE[camera_id][pet_name] = []
        
    # Delete existing profiles for this pet on this camera to ensure exactly max 5
    c.execute("DELETE FROM pet_profiles WHERE camera_id=? AND pet_name=?", (camera_id, pet_name))
    GLOBAL_SCOPED_CACHE[camera_id][pet_name] = []
    
    for vector in extracted_vectors:
        vector /= np.linalg.norm(vector)
        vector_blob = vector.tobytes()
        c.execute("INSERT INTO pet_profiles (camera_id, pet_name, vector_blob) VALUES (?, ?, ?)",
                  (camera_id, pet_name, vector_blob))
        GLOBAL_SCOPED_CACHE[camera_id][pet_name].append(vector)
        
    conn.commit()
    conn.close()
    
    gc.collect()
    
    msg = f"{pet_name} registered successfully with {len(extracted_vectors)} photos."
    if failed_count > 0:
        msg += f" (Note: {failed_count} photos were skipped due to invalid dog count)."
        
    return jsonify({"success": True, "message": msg})


class BackgroundWorkerManager:
    """ Persistent Scope-Isolated Background Stream Engine """
    def __init__(self):
        self.threads = {}
        self.running_flags = {}
        self.video_streams = {}

    def stop_camera_thread(self, camera_id):
        if camera_id in self.running_flags:
            self.running_flags[camera_id] = False
            del self.running_flags[camera_id]
        if camera_id in self.video_streams:
            self.video_streams[camera_id].stop()
            del self.video_streams[camera_id]
        if camera_id in self.threads:
            del self.threads[camera_id]
        gc.collect()

    def start_camera_thread(self, camera_id, stream_url):
        if camera_id in self.threads:
            return # Already running
            
        self.running_flags[camera_id] = True
        
        # Cast stream_url to int if it's purely digits (USB cams)
        target = int(stream_url) if stream_url.isdigit() else stream_url
        
        # Instantiate continuous backlog-free stream reader
        self.video_streams[camera_id] = VideoStream(target, reconnect_interval=10)
            
        def worker():
            last_ai_time = 0.0
            ai_interval = 1.0 # 1,000 milliseconds clock gate
            inference_latency_ms = 0.0
            last_detections = []
            
            while self.running_flags.get(camera_id, False):
                stream = self.video_streams.get(camera_id)
                if not stream:
                    break
                    
                frame = stream.read()
                
                if frame is None:
                    socketio.emit('stream_update', {"camera_id": camera_id, "status": "inactive"})
                    time.sleep(1.0)
                    continue
                    
                # Hard-resize to 640x640 to conserve memory and align with AI
                try:
                    frame = cv2.resize(frame, (640, 640))
                except Exception as e:
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # 1-FPS Scoped Inference
                if current_time - last_ai_time >= ai_interval:
                    start_inference = time.time()
                    
                    cache_for_cam = GLOBAL_SCOPED_CACHE.get(camera_id, {})
                    
                    # Real AI Pipeline Inference
                    last_detections = ai_pipeline.run_inference(frame, cache_for_cam)
                    
                    last_ai_time = time.time()
                    inference_latency_ms = (last_ai_time - start_inference) * 1000
                    
                    # Log detections to SQLite
                    if len(last_detections) > 0:
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            for det in last_detections:
                                c.execute('''
                                    INSERT INTO detection_logs (camera_id, tracking_id, identified_name, confidence, inference_latency_ms)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (camera_id, det["tracking_id"], det["name"], det["confidence"], inference_latency_ms))
                            conn.commit()
                            conn.close()
                        except Exception as e:
                            logging.error(f"Failed to log to db: {e}")

                # Backend Overlay Drawing
                for det in last_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    name = det["name"]
                    conf = det["confidence"]
                    tid = det["tracking_id"]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 204), 2)
                    
                    label = f"#{tid} {name} ({int(conf*100)}%)"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_width, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)
                
                # Telemetry Payload Preparation
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                img_str = base64.b64encode(buffer).decode('utf-8')
                    
                payload = {
                    "camera_id": camera_id,
                    "status": "active",
                    "image": f"data:image/jpeg;base64,{img_str}",
                    "metrics": {
                        "inference_latency_ms": round(inference_latency_ms, 2),
                        "system_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                socketio.emit('stream_update', payload)
                
                del frame
                del buffer
                
                # Restrict frame pull rate slightly
                time.sleep(0.03) 
                
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
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)