# PupsN Vision System - Day 1 MVP Pipeline

Welcome to the **PupsN Vision System**. This repository contains the definitive, low-level implementation of the Day 1 MVP Pipeline for an ultra-lightweight, dual-threaded Edge AI Camera System.

The software is built specifically to decouple the Web Server State from the Video Processing State, allowing high-performance neural network inference (YOLOv8/11 Nano + OSNet) on low-power architectures without choking the hardware.

---

## 🏗️ System Architecture & Topology

To prevent frame drops and thread exhaustion, the system is separated into two primary execution spaces running concurrently.

```text
+-------------------------------------------------------------------------------+
|                           CORE FLASK BACKEND ARCHITECTURE                     |
|                                                                               |
|    +------------------------+             +------------------------------+    |
|    |  MAIN THREAD           |             |  BACKGROUND WORKER THREAD    |    |
|    |  (Web Server)          |             |  (Persistent Engine)         |    |
|    |                        |             |                              |    |
|    |  - Serves UI Pages     |             |  - Persistent OpenCV Loop    |    |
|    |  - Handles User Form   |             |  - Exponential Backoff Engine|    |
|    |    Submissions         |             |  - 1-FPS AI Inference Throttler|  |
|    |  - SQLite I/O Ops      |             |  - Memory Flushing Engine    |    |
|    +-----------+------------+             +--------------+---------------+    |
|                |                                         |                    |
+----------------|-----------------------------------------|--------------------+
                 |                                         |
                 |             WebSocket Channel           |  Raw Frame + 
                 |            (Flask-SocketIO)             |  JSON Metadata
                 |                                         v
                 |                            +----------------------------+
                 |                            |    CLIENT-SIDE UI          |
                 +--------------------------->|    - HTML5 Canvas Render   |
                                              |    - Bounding Box Overlay  |
                                              +----------------------------+
```

---

## 🧠 AI Pipeline (ONNX Runtime)

The application utilizes `onnxruntime` to perform Deep Metric Learning matching.

1. **Step 1: Downscaling & Preprocessing:** The frame is downscaled to 640x640, normalized, and converted to the NCHW format expected by YOLO ONNX exports.
2. **Step 2: Detection (YOLOv8/11 Nano INT8):** The frame passes through the quantized YOLO network. Non-Maximum Suppression (NMS) isolates class ID 16 (`dog`).
3. **Step 3: Dynamic Cropping:** The localized bounding box is extracted from the original, unscaled frame and hard-resized to the 256x128 aspect ratio.
4. **Step 4: Metric Feature Extraction (OSNet `osnet_x1_0`):** Standard ImageNet normalization is applied, and OSNet extracts a 512-dimension mathematical vector.
5. **Step 5: Math & Thresholding:** The extracted vector is mathematically compared (using Cosine Similarity) against all pre-calculated pet vectors loaded in the SQLite RAM Map cache. A match is confirmed if `Similarity >= 0.85`.

---

## 🚀 Installation & Hardware Setup

### Prerequisites
- Python 3.9+
- Linux/macOS/Windows (Tested on Ubuntu/Raspberry Pi OS)
- A working Web Camera (USB or Built-in) or a valid RTSP stream URL.

### 1. Environment Setup

```bash
# Clone the repository
git clone git@github.com:akshaysatyam2/PupsN_Edge.git
cd PupsN_Edge

# Initialize a Virtual Environment to avoid system pollution
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the strict dependencies
pip install -r requirements.txt
```

### 2. Loading the AI Models (.onnx)
Because `.onnx` files are extremely large binary weights, they are **strictly ignored** by Git. **The application will safely bypass AI inference if the files are not present, but you will only see the raw stream.**

You must supply your own YOLO Nano and OSNet ONNX exports and place them in the `models/` directory:

1. Place your YOLO ONNX export at: `models/yolo_nano.onnx`
2. Place your OSNet ONNX export at: `models/osnet_x1_0.onnx`

*Note: The YOLO model must be exported with standard output formatting `[1, 84, 8400]` or adjusted in `models.py`. OSNet must output a standard `[1, 512]` vector.*

---

## 💻 Running the Application

Once dependencies and models are loaded, execute the main thread:

```bash
python app.py
```

### Accessing the Feed Locally & Over the Network
The Flask server binds strictly to `0.0.0.0:5000` via Eventlet/SocketIO. 

* **On the Host Machine:** Open a browser to `http://localhost:5000`
* **Over the Local Network:** Open a browser to `http://<YOUR_LOCAL_IP>:5000` (e.g. `http://192.168.1.3:5000`)

---

## 🛡️ Key Edge Optimizations Built-In

1. **Explicit Memory De-allocation:** Python's native `gc.collect()` is triggered manually at the end of the streaming while-loop, preventing memory fragmentation over days of continuous operation.
2. **RAM Map Caching:** Bypasses SQLite disk lookup latency by serializing `BLOB` weights back into numpy `float32` arrays globally on application boot.
3. **HTML5 Canvas:** Bypasses DOM memory leaks caused by rapid `<img>` `src` replacements by pushing Base64 JPEGs directly into a Javascript 2D Context render layer.
4. **Hardware Queue Polling:** A secondary background thread isolates `cv2.VideoCapture.read()`, absorbing camera buffer buildup at native FPS while standardizing downstream AI checks to exactly 1.0 seconds (1 FPS) delta-time.