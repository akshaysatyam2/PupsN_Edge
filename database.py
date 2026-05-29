import sqlite3
import numpy as np

DB_PATH = 'pupsn_edge.db'
GLOBAL_PET_CACHE = {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Profile Store
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pet_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pet_name TEXT NOT NULL UNIQUE,
        vector_blob BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Hardware Config
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS camera_config (
        id INTEGER PRIMARY KEY CHECK (id = 1), 
        stream_url TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Analytical Event Logger
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tracking_id INTEGER NOT NULL,
        identified_name TEXT NOT NULL,
        confidence REAL NOT NULL
    )
    ''')
    
    # Insert default camera if empty
    cursor.execute('SELECT count(*) FROM camera_config')
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO camera_config (id, stream_url) VALUES (1, '0')")
        
    conn.commit()
    conn.close()

def load_pet_cache():
    """Startup Vector Caching: Maps DB blobs directly into global RAM dictionary"""
    global GLOBAL_PET_CACHE
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT pet_name, vector_blob FROM pet_profiles')
    rows = cursor.fetchall()
    
    for pet_name, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        GLOBAL_PET_CACHE[pet_name] = vector
        
    conn.close()
    print(f"Loaded {len(GLOBAL_PET_CACHE)} profiles into RAM cache.")

def get_camera_url():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT stream_url FROM camera_config WHERE id = 1')
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else '0'

def log_detection_event(tracking_id, name, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO detection_logs (tracking_id, identified_name, confidence)
    VALUES (?, ?, ?)
    ''', (tracking_id, name, confidence))
    conn.commit()
    conn.close()

# Initialize on import
init_db()
load_pet_cache()
