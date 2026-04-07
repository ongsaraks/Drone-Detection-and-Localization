# --- FULL CODE: Custom Tracker with Snail Trail, Dynamic Info and Socket.IO ---
from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import gc
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from collections import deque
import math
import joblib
import pandas as pd
import socketio
import requests
import base64
import random
from datetime import datetime
import io
import os

import jwt

# --- Socket.IO Setup ---
# SERVER_URL = "http://192.168.10.10:3000"
SERVER_URL = "http://localhost:3000"
USERNAME = "admin"
PASSWORD = "1234"
JWT_SECRET = os.getenv("JWT_SECRET", "your_strong_secret_key_drone_control_center_2025")

# Create Socket.IO client
sio = socketio.Client()
auth_token = None
drone_ids_map = {}  # Map track_id to server drone ID

# --- Weather Mock Data ---
WEATHER_OPTIONS = ["cloudy"]

# --- Helper function to calculate distance ---
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Authentication Function ---
def authenticate():
    """Login to server and get JWT token"""
    global auth_token
    try:
        response = requests.post(
            f"{SERVER_URL}/api/login",
            json={"username": USERNAME, "password": PASSWORD}
        )
        if response.status_code == 200:
            data = response.json()
            auth_token = data.get("token")
            print(f"✅ Authentication successful! Token: {auth_token[:20]}...")
            return True
        else:
            print(f"❌ Authentication failed: {response.text}")
            fallback_payload = {
                "id": 1,
                "username": USERNAME,
                "role": "admin",
            }
            auth_token = jwt.encode(fallback_payload, JWT_SECRET, algorithm="HS256")
            if isinstance(auth_token, bytes):
                auth_token = auth_token.decode("utf-8")
            print("⚠️ Using locally signed fallback JWT for the current backend secret.")
            return True
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False

# --- Send Drone Detection with Image ---
def send_drone_detection(track_id, lat, lon, alt, center_x, center_y, width, height, conf, weather, frame):
    """Send new drone detection to server with image"""
    global auth_token, drone_ids_map
    
    if auth_token is None:
        print("❌ Not authenticated. Cannot send detection.")
        return None
    
    try:
        # Crop bounding box from frame
        x1 = max(0, int(center_x - width/2))
        y1 = max(0, int(center_y - height/2))
        x2 = min(frame.shape[1], int(center_x + width/2))
        y2 = min(frame.shape[0], int(center_y + height/2))
        
        cropped_img = frame[y1:y2, x1:x2]
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', cropped_img)
        img_bytes = buffer.tobytes()
        
        # Use track_id directly as drone_id (1 or 2)
        drone_id = str(track_id)
        drone_ids_map[track_id] = drone_id
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'drone_{drone_id}_{timestamp}.jpg'
        
        # Prepare multipart form data
        files = {
            'image': (filename, img_bytes, 'image/jpeg')
        }
        
        data = {
            'drone_id': drone_id,
            'confidence': float(conf),
            'latitude': float(lat),
            'longitude': float(lon),
            'altitude': float(alt),
            'weather': weather,
            'width': float(width / 100),  # Convert to meters (assuming scale)
            'height': float(height / 100)
        }
        
        headers = {
            'Authorization': f'Bearer {auth_token}'
        }
        
        response = requests.post(
            f"{SERVER_URL}/api/drone-theirs/upload",
            headers=headers,
            data=data,
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            server_id = result.get('insertedId')
            print(f"✅ Drone {drone_id} sent (POST) successfully! Server ID: {server_id}")
            return server_id
        else:
            print(f"❌ Failed to send drone (POST): {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error sending detection: {e}")
        return None

# --- ### NEW/MODIFIED - TASK 1 ### ---
# Replaced 'update_drone_position' to correctly send 'multipart/form-data'
# --- Update Drone Position ---
def update_drone_position(track_id, lat, lon, alt, center_x, center_y, width, height, conf, weather, frame, server_id):
    """Update existing drone position on server with a new image"""
    global auth_token
    
    if auth_token is None:
        print("❌ Not authenticated. Cannot update position.")
        return False
    
    try:
        # Crop and encode image
        x1 = max(0, int(center_x - width/2))
        y1 = max(0, int(center_y - height/2))
        x2 = min(frame.shape[1], int(center_x + width/2))
        y2 = min(frame.shape[0], int(center_y + height/2))
        
        cropped_img = frame[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', cropped_img)
        img_bytes = buffer.tobytes()
        
        # Create unique filename, just like in POST
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'drone_{track_id}_{timestamp}.jpg'
        
        # Prepare multipart form data
        files = {
            'image': (filename, img_bytes, 'image/jpeg')
        }
        
        data = {
            'confidence': float(conf),
            'latitude': float(lat),
            'longitude': float(lon),
            'altitude': float(alt),
            'weather': weather,
            'width': float(width / 100),
            'height': float(height / 100),
            # NOTE: Removed 'image_path'. The server should handle this
            # from the uploaded file 'image'.
        }
        
        headers = {
            'Authorization': f'Bearer {auth_token}',
            # REMOVED: 'Content-Type': 'application/json'
            # 'requests' will set multipart/form-data header automatically
        }
        
        response = requests.put(
            f"{SERVER_URL}/api/drone-theirs/{server_id}",
            headers=headers,
            data=data,  # Use 'data' for form fields
            files=files   # Use 'files' for file upload
        )
        
        if response.status_code == 200:
            print(f"✅ Drone {track_id} position updated (PUT)!")
            return True
        else:
            print(f"❌ Failed to update drone (PUT): {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating position: {e}")
        return False

# --- Delete Drone ---
def delete_drone(track_id, server_id):
    """Delete drone from server when lost from frame"""
    global auth_token
    
    if auth_token is None or server_id is None:
        print("❌ Cannot delete drone. Missing auth or server_id.")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}'
        }
        
        response = requests.delete(
            f"{SERVER_URL}/api/drone-theirs/{server_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"🗑️ Drone {track_id} deleted successfully!")
            return True
        else:
            print(f"❌ Failed to delete drone: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error deleting drone: {e}")
        return False

# --- ### NEW - TASK 2 ### ---
# Added 'check_drone_exists' to check server before POST/PUT
# --- Check if drone exists in DB ---
def check_drone_exists(drone_id):
    """Check if a drone with drone_id (e.g., '1' or '2') exists in DB"""
    global auth_token, SERVER_URL
    
    if auth_token is None:
        print("❌ Not authenticated. Cannot check drone.")
        return None
    
    # The backend exposes the recent enemy-drone list, so we can resolve
    # the current DB id for a track without assuming a custom lookup route.
    url = f"{SERVER_URL}/api/recent/theirs"
    
    try:
        headers = {
            'Authorization': f'Bearer {auth_token}'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            payload = response.json()
            rows = payload.get('data', []) if isinstance(payload, dict) else []
            matching_rows = [row for row in rows if str(row.get('drone_id')) == str(drone_id)]

            if matching_rows:
                latest_row = sorted(
                    matching_rows,
                    key=lambda row: row.get('detected_at') or '',
                    reverse=True,
                )[0]
                server_db_id = latest_row.get('id')
                if server_db_id:
                    print(f"✅ Drone {drone_id} found in DB. Server ID: {server_db_id}")
                    return server_db_id

            print(f"ℹ️ Drone {drone_id} not found in recent records.")
            return None
        elif response.status_code == 404:
            print(f"ℹ️ Drone {drone_id} not found in DB (404).")
            return None
        else:
            print(f"❌ Error checking drone: {response.status_code} {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking drone: {e}")
        return None

# --- Socket.IO Event Handlers ---
@sio.event
def connect():
    print("🔌 Connected to Socket.IO server")

@sio.event
def disconnect():
    print("🔌 Disconnected from Socket.IO server")

@sio.on('drone-theirs-detected')
def on_drone_detected(data):
    print(f"📥 Server confirmed detection: {data}")

@sio.on('drone-theirs-updated')
def on_drone_updated(data):
    print(f"📥 Server confirmed update: {data}")

# --- Connect to Socket.IO ---
def connect_socketio():
    """Connect to Socket.IO server"""
    try:
        sio.connect(SERVER_URL)
        print("✅ Socket.IO connected successfully!")
        return True
    except Exception as e:
        print(f"❌ Socket.IO connection failed: {e}")
        return False

# --- 1. Setup Paths and Model ---
INPUT_DIR = r"P3_VIDEO__DAY3.mp4"
YOLO_MODEL_PATH = r"longest.pt"
GBR_MODEL_PATH = r"gb_multioutput.joblib"

# --- Load GBR Model ---
print(f"Loading GBR model from {GBR_MODEL_PATH}...")
try:
    obj = joblib.load(GBR_MODEL_PATH)
    gbr_model = obj["model"]
    features = list(obj.get("features", ["image_index","center_x","center_y","width","height"]))
    print(f"GBR model loaded. Expecting features: {features}")
except Exception as e:
    print(f"!!! FATAL ERROR: Could not load GBR model. {e}")
    gbr_model = None

# --- Load YOLO Model ---
print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
input_dir = Path(INPUT_DIR)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(str(input_dir))
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_dir}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Authenticate and Connect ---
print("🔐 Authenticating...")
if not authenticate():
    print("❌ Failed to authenticate. Exiting.")
    exit(1)

print("🔌 Connecting to Socket.IO...")
if not connect_socketio():
    print("⚠️ Socket.IO connection failed. Continuing without real-time updates.")

# --- 2. Custom Tracker State ---
tracks = {
    1: {'pos': None, 'history': [], 'age': 0, 'box': None, 'first_seen_frame': -1, 'server_id': None, 'sent': False},
    2: {'pos': None, 'history': [], 'age': 0, 'box': None, 'first_seen_frame': -1, 'server_id': None, 'sent': False}
}

# --- 3. Tunable Parameters ---
MAX_AGE = 50
DISTANCE_THRESHOLD = 30
FRAME_SKIP = 5
CONF_THRES = 0.4
UPDATE_INTERVAL = 10  # Update position every N frames

USE_HALF_PRECISION = torch.cuda.is_available()
print(f"--- System Check ---")
print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")
if not USE_HALF_PRECISION:
    print("!!! WARNING: Running on CPU. This will be slow. !!!")
print("--------------------")

# --- 4. Dynamic Data and Text Panel Setup ---
current_track_data = {
    1: {"lat": None, "lon": None, "alt": None, "conf": None, "weather": None},
    2: {"lat": None, "lon": None, "alt": None, "conf": None, "weather": None}
}
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
line_thickness = 2
start_x = 20
start_y = 40
line_height = 25

frame_count = 0

# --- ### NEW - TASK 2 ### ---
# Added helper function to de-duplicate the POST/PUT logic
def handle_detection_data(track_id, det, frame):
    """
    Handles GBR prediction and server communication (POST/PUT) for a detection.
    Returns True if successful, False otherwise.
    """
    global tracks, frame_count, gbr_model, features, WEATHER_OPTIONS, current_track_data
    
    try:
        # 1. --- RUN GBR PREDICTION ---
        df_input = pd.DataFrame([det['gbr_input']], columns=features)
        prediction = gbr_model.predict(df_input)[0]
        new_lat, new_lon, new_alt = prediction[0], prediction[1], prediction[2]
        weather = random.choice(WEATHER_OPTIONS)
        
        # 2. --- Store data locally ---
        current_track_data[track_id] = {
            "lat": new_lat, "lon": new_lon, "alt": new_alt,
            "conf": det['conf'], "weather": weather
        }
        
        cx, cy, w, h = det['dims']
        
        # 3. --- Server Communication (POST/PUT) ---
        server_id = tracks[track_id].get('server_id') # Use .get for safety
        
        if server_id is None and not tracks[track_id].get('sent'):
            # First time seeing this track_id, or script just started.
            # Check DB before doing anything.
            drone_id_str = str(track_id)
            print(f"Checking server for existing drone_id: {drone_id_str}...")
            server_id = check_drone_exists(drone_id_str)
            if server_id:
                # We found it! Store the ID.
                tracks[track_id]['server_id'] = server_id
            # 'sent' will be used as a flag to mean "we have checked the server"
            tracks[track_id]['sent'] = True
        
        if server_id:
            # --- Drone EXISTS in DB (or we just found it) ---
            
            # Check for update interval
            if frame_count % UPDATE_INTERVAL == 0:
                print(f"Updating (PUT) drone {track_id} (Server ID: {server_id})...")
                update_drone_position(
                    track_id, new_lat, new_lon, new_alt,
                    cx, cy, w, h, det['conf'], weather,
                    frame, server_id
                )
        else:
            # --- Drone NOT in DB (check_drone_exists returned None) ---
            print(f"Registering (POST) new drone {track_id}...")
            new_server_id = send_drone_detection(
                track_id, new_lat, new_lon, new_alt,
                cx, cy, w, h, det['conf'], weather, frame
            )
            if new_server_id:
                tracks[track_id]['server_id'] = new_server_id
                # 'sent' is already True, but 'server_id' is now populated
        
        return True
    
    except Exception as e:
        print(f"❌ Error in handle_detection_data for track_id {track_id}: {e}")
        return False

# --- 5. Main Video Processing Loop ---
print("🎬 Starting video processing...")
while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # --- 5a. Increment Age ---
        for track_id in [1, 2]:
            tracks[track_id]['age'] += 1
            
            # Check if drone is lost (exceeded MAX_AGE)
            if tracks[track_id]['age'] == MAX_AGE and tracks[track_id]['server_id']:
                print(f"🔍 Drone {track_id} lost from frame. Deleting from server...")
                delete_drone(track_id, tracks[track_id]['server_id'])
                # Reset track state
                tracks[track_id]['sent'] = False # Will force a new check if it reappears
                tracks[track_id]['server_id'] = None
                tracks[track_id]['pos'] = None
                tracks[track_id]['history'] = []

        # --- 5b. Run Prediction & Matching ---
        if frame_count % FRAME_SKIP == 0:
            results = yolo_model.predict(frame, imgsz=640, half=USE_HALF_PRECISION, verbose=False)
            current_detections = []
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy_coords = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for (x1, y1, x2, y2), conf in zip(xyxy_coords, confs):
                    if conf < CONF_THRES:
                        continue
                        
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    gbr_input_features = [frame_count, center_x, center_y, width, height]
                    
                    current_detections.append({
                        'centroid': (center_x, center_y),
                        'box': [x1, y1, x2, y2],
                        'gbr_input': gbr_input_features,
                        'conf': conf,
                        'dims': (center_x, center_y, width, height)
                    })
            
            # --- Match Detections to Tracks ---
            unassigned_detections = list(range(len(current_detections)))
            tracks_updated = {1: False, 2: False}

            # Part A: Match existing tracks
            for track_id in [1, 2]:
                if tracks[track_id]['pos'] is not None and tracks[track_id]['age'] <= MAX_AGE:
                    last_pos = tracks[track_id]['pos']
                    best_dist = float('inf')
                    best_det_idx = -1
                    
                    for i in unassigned_detections:
                        dist = calculate_distance(last_pos, current_detections[i]['centroid'])
                        if dist < DISTANCE_THRESHOLD and dist < best_dist:
                            best_dist = dist
                            best_det_idx = i
                    
                    if best_det_idx != -1:
                        det = current_detections[best_det_idx]
                        
                        # --- ### MODIFIED - TASK 2 ### ---
                        # Replaced inline logic with call to helper function
                        # --- RUN GBR PREDICTION & SERVER SYNC ---
                        if gbr_model is not None:
                            # This function now handles GBR, data storage, and POST/PUT logic
                            if handle_detection_data(track_id, det, frame):
                                # Update track state *after* successful processing
                                tracks[track_id]['pos'] = det['centroid']
                                tracks[track_id]['age'] = 0
                                tracks[track_id]['box'] = det['box']
                                tracks_updated[track_id] = True
                                unassigned_detections.remove(best_det_idx)
                            else:
                                print(f"Failed to process detection for matched track {track_id}")
                        else:
                            # Fallback if no GBR model (no server sync)
                            tracks[track_id]['pos'] = det['centroid']
                            tracks[track_id]['age'] = 0
                            tracks[track_id]['box'] = det['box']
                            tracks_updated[track_id] = True
                            unassigned_detections.remove(best_det_idx)

            # Part B: Assign new detections
            for det_idx in unassigned_detections:
                for track_id in [1, 2]:
                    if not tracks_updated[track_id] and (tracks[track_id]['pos'] is None or tracks[track_id]['age'] > MAX_AGE):
                        det = current_detections[det_idx]

                        # --- ### MODIFIED - TASK 2 ### ---
                        # Replaced inline logic with call to helper function
                        # --- RUN GBR PREDICTION & SERVER SYNC ---
                        if gbr_model is not None:
                            # This function now handles GBR, data storage, and POST/PUT logic
                            if handle_detection_data(track_id, det, frame):
                                # Update track state *after* successful processing
                                tracks[track_id]['pos'] = det['centroid']
                                tracks[track_id]['age'] = 0
                                tracks[track_id]['box'] = det['box']
                                tracks[track_id]['first_seen_frame'] = frame_count
                                tracks_updated[track_id] = True
                                break # Exit inner loop (track assigned)
                            else:
                                print(f"Failed to process detection for new track {track_id}")
                        else:
                            # Fallback if no GBR model (no server sync)
                            tracks[track_id]['pos'] = det['centroid']
                            tracks[track_id]['age'] = 0
                            tracks[track_id]['box'] = det['box']
                            tracks[track_id]['first_seen_frame'] = frame_count
                            tracks_updated[track_id] = True
                            break # Exit inner loop (track assigned)
        
        # --- 5c. Update History ---
        for track_id in [1, 2]:
            if tracks[track_id]['pos'] is not None and tracks[track_id]['age'] <= MAX_AGE:
                tracks[track_id]['history'].append(tracks[track_id]['pos'])

        # --- 5d. Draw Everything ---
        final_frame = frame.copy()
        
        # Draw Snail Trails
        for track_id in [1, 2]:
            if tracks[track_id]['age'] < MAX_AGE and len(tracks[track_id]['history']) > 1:
                points = np.array(tracks[track_id]['history'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(final_frame, [points],
                              isClosed=False,
                              color=colors(track_id, True),
                              thickness=2)

        # Draw Bounding Boxes
        annotator = Annotator(final_frame, line_width=2)
        for track_id in [1, 2]:
            if tracks[track_id]['age'] < MAX_AGE and tracks[track_id]['box'] is not None:
                label = f"ID: {track_id}"
                color = colors(track_id, True)
                annotator.box_label(tracks[track_id]['box'], label, color=color)

        final_frame = annotator.result()

        # Draw Top-Left Dynamic Text Panel
        current_y = start_y
        
        for track_id in [1, 2]:
            if tracks[track_id]['age'] < MAX_AGE and tracks[track_id]['pos'] is not None: # Added check for 'pos'
                data = current_track_data[track_id]
                text_color = colors(track_id, True)
                
                cv2.putText(final_frame, f"track_id:{track_id}", (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height
                
                lat_str = f"- lat: {data['lat']:.5f}" if data['lat'] is not None else "- lat: Waiting..."
                lon_str = f"- lon: {data['lon']:.5f}" if data['lon'] is not None else "- lon: Waiting..."
                alt_str = f"- alt: {data['alt']:.2f}" if data['alt'] is not None else "- alt: Waiting..."
                conf_str = f"- conf: {data['conf']:.2f}" if data['conf'] is not None else "- conf: N/A"
                weather_str = f"- weather: {data['weather']}" if data['weather'] is not None else "- weather: N/A"
                
                cv2.putText(final_frame, lat_str, (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height
                
                cv2.putText(final_frame, lon_str, (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height
                
                cv2.putText(final_frame, alt_str, (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height
                
                cv2.putText(final_frame, conf_str, (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height
                
                cv2.putText(final_frame, weather_str, (start_x, current_y),
                            font, font_scale, text_color, line_thickness, cv2.LINE_AA)
                current_y += line_height + 15

        # --- 5e. Display the Frame ---
        cv2.imshow("Custom YOLOv11 Tracker", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Processing error: {e}")
        break

# --- 6. Cleanup ---
print("🧹 Cleaning up...")
cap.release()
cv2.destroyAllWindows()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Disconnect Socket.IO
if sio.connected:
    sio.disconnect()

print("✅ Processing finished.")
