import eventlet
eventlet.monkey_patch()

import time
import cv2
import base64
import gc
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from camera import RealTimeVideoStream
from models import AIPipeline
from database import get_camera_url, log_detection_event

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state for the persistent background thread
ai_pipeline = AIPipeline()
video_stream = None
background_thread = None

def persistent_video_engine():
    """
    The Background Worker Thread (Persistent Engine).
    Runs independently of client connections.
    """
    global video_stream
    
    camera_url = get_camera_url()
    video_stream = RealTimeVideoStream(camera_url).start()
    
    last_ai_time = 0.0
    ai_interval = 1.0 # 1,000 milliseconds delta-time check
    
    while True:
        ret, frame = video_stream.read()
        
        if not ret or frame is None:
            socketio.emit('stream_update', {"status": "inactive"})
            time.sleep(0.1) # Prevent CPU pegging when inactive
            continue
            
        current_time = time.time()
        detections = []
        
        # AI Gating (1-FPS Strategic Throttling)
        if current_time - last_ai_time >= ai_interval:
            last_ai_time = current_time
            detections = ai_pipeline.run_inference(frame)
            
            # Log detections to SQLite
            for det in detections:
                log_detection_event(det['tracking_id'], det['name'], det['confidence'])

        # Base64 Frame Streaming Protocol
        # Compress to JPEG at 80% quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "status": "active",
            "image": f"data:image/jpeg;base64,{img_str}",
            "detections": detections
        }
        
        socketio.emit('stream_update', payload)
        
        # Extreme Memory Management Guardrails
        del frame
        del buffer
        gc.collect()
        
        # Small sleep to allow thread switching, yielding to eventlet/socketio
        eventlet.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Spawning the background worker daemon thread
    background_thread = threading.Thread(target=persistent_video_engine, daemon=True)
    background_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
