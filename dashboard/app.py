import os
import sys
import time
import json
import logging
import threading
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
import Arduino
import webbrowser
from threading import Timer
import io
import webview
import tempfile
import urllib.request
import yaml
import zipfile
import glob
import torch

VAL_ZIP_URL = "https://github.com/abbas-pt/ExpoChallenge_AbbasLotfi/releases/download/data/val.zip"

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

def get_resource_path(relative_path):
   
    if hasattr(sys, '_MEIPASS'):
       
        return os.path.join(sys._MEIPASS, relative_path)
    
    return os.path.join(os.path.abspath("."), relative_path)



def load_config():
    config_path = get_resource_path("config.yaml") 
    if not os.path.exists(config_path):
        config_path = "config.yaml" 
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()


arduino_log_out=[]
payload=[]

CONVEYOR_DIRECTION = config["conveyor"]["direction"]


# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. Startup paths verification
try:
   device="cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print(e)

MODEL_PATH = get_resource_path("create_exe_file/best_abbas.pt")
DEVICE=device

DATA_YAML_PATH =get_resource_path("create_exe_file/data1.yaml")  # for validating model accuracy on dashboard

if not os.path.exists(MODEL_PATH):
    logging.warning(f"⚠️ Model file not found at '{MODEL_PATH}'. Ensure correct path before running detection.")

# 2. Hardware connection
try:
    arduino, status_msg = Arduino.connect()


except Exception as e:
    print(f"Warning: Arduino not connected or port not found ({e}). Running in offline mode.")


# 3. Model & CLAHE setup
model = YOLO(MODEL_PATH).to(device=device) if os.path.exists(MODEL_PATH) else None
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))



_CV_MAJOR, _CV_MINOR = map(int, cv2.__version__.split(".")[:2])
USES_LEGACY_ANGLE_CONVENTION = (_CV_MAJOR, _CV_MINOR) < (4, 5)
logging.info(
    f"OpenCV version detected: {cv2.__version__} -> "
    f"{'legacy' if USES_LEGACY_ANGLE_CONVENTION else 'modern'} minAreaRect angle convention in use."
)



# set gripper force for each class(0,100N)
GRIP_FORCE_MAP = config["grip_forces"]





# 18-Class mapping
CATEGORY_5_MAP = {
    "Aluminium foil": "Metal",
    "Bottle cap": "Plastic",
    "Bottle": "Plastic",
    "Broken glass": "Glass",
    "Can": "Metal",
    "Carton": "Paper",
    "Cigarette": "Waste",
    "Cup": "Paper",
    "Lid": "Plastic",
    "Other litter": "Waste",
    "Other plastic": "Plastic",
    "Paper": "Paper",
    "Plastic bag - wrapper": "Plastic",
    "Plastic container": "Plastic",
    "Pop tab": "Metal",
    "Straw": "Plastic",
    "Styrofoam piece": "Waste",
    "Unlabeled litter": "Waste",
}


TARGET_5_CLASSES = ["Glass", "Metal", "Paper", "Plastic", "Waste"]

ENVIRONMENTAL_PRIORITY = {
    "Metal": 1,
    "Plastic": 2,
    "Glass": 3,
    "Paper": 4,
    "Waste": 5
}


# WASTE_VALUES (USD)
WASTE_VALUES = {
    "Metal": 0.08,
    "Plastic": 0.05,
    "Glass": 0.04,
    "Paper": 0.03,
    "Waste": 0.00
}

n=config["bin_capacities"]
BIN_CAPACITIES = {k: n for k in TARGET_5_CLASSES}

TRIGGER_LINE_RATIO = config["vision"]["trigger_line_ratio"]
TRIGGER_TOLERANCE = config["vision"]["trigger_tolerance_px"]
SCALE_FACTOR_MM = config["vision"]["scale_factor_mm"]
GRASPING_ZONE_Y_MM = config["conveyor"]["grasping_zone_y_mm"]
CONVEYOR_SPEED_MM_S = config["conveyor"]["speed_mm_s"]

# For Calculate Availability
PLANNED_PRODUCTION_TIME = config['PLANNED_PRODUCTION_TIME']  # 3600(Second) == 1H



# For Calculate Performance
"""
The shortest possible time for a robot to process and separate
a piece of waste in seconds (here 0.5 seconds, or the ideal speed
of 2 pieces of waste per second).
"""
IDEAL_CYCLE_TIME = config['IDEAL_CYCLE_TIME']

# Thread synchronization
state_lock = threading.Lock()

# Global state
is_emergency_stopped = False
total_downtime = 0.0 #Total stop times
downtime_start_marker = None # Stop start time marker
is_conveyor_halted = False

track_last_seen = {} # Memory management (Garbage Collection) and cleaning up old identifiers.

processed_track_ids = set() # Prevent the robot from issuing repeated commands for a specific waste.

prev_frame_time = time.time() # For Calculate FPS

start_time = time.time()

system_metrics = {
    "total_count": 0,
    "confidence_sum": 0.0,
    "total_revenue": 0.0,
}

for k in TARGET_5_CLASSES:  # Add All CLASS_MAPPING Values in system_metrics
    system_metrics[k] = 0


bin_fill_level = {k: 0 for k in TARGET_5_CLASSES}
sorting_timestamps = []
log_history = []
time_series_data = {"Time": [0], "Total Sorted": [0]}


def send_serial_cmd(command_payload: str,arduino):
    if arduino and hasattr(arduino, "is_open") and arduino.is_open:
        try:
            Arduino.send_to_arduino(message=command_payload,arduino=arduino)
           
            return True

        except Exception as e:
            logging.error(f"Serial transmission error: {e}")
    return False


def extract_object_orientation(frame: np.ndarray, bbox: tuple) -> float:
    """
    Estimate object rotation angle from a CLEAN (undrawn) frame.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:  # Supporter For contours
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 20:
            rect = cv2.minAreaRect(c)
            (cx, cy), (width, height), angle = rect 

            if USES_LEGACY_ANGLE_CONVENTION:
               
                if angle < -45:
                    angle += 90.0
            else:
              
                if width < height:
                    angle -= 90.0

            return float(angle)

    return 0.0



def cleanup_tracking_memory(current_time: float, max_age_seconds: float = 30.0):
    global processed_track_ids, track_last_seen
    expired_ids = [tid for tid, last_seen in track_last_seen.items() if current_time - last_seen > max_age_seconds]
    for tid in expired_ids:
        processed_track_ids.discard(tid)
        del track_last_seen[tid]




def calculate_kinematics_and_send(obj, frame, frame_width, frame_height):

    cx, cy = obj["center_x"], obj["center_y"] # object x,y position in picture

  


    xw_mm = (cx - (frame_width / 2)) * SCALE_FACTOR_MM # convert pixles to mm
    zw_mm = config['zm'] #mm


    trigger_y_px = int(frame_height * TRIGGER_LINE_RATIO) # trigger line position on picture(pixle)
    
    if CONVEYOR_DIRECTION == "UPWARD":
   
        delta_y_px = trigger_y_px - cy # object center_y distance untill triggerline
    else:
       
        delta_y_px = cy - trigger_y_px
    delta_y_mm = delta_y_px * SCALE_FACTOR_MM 


    dist_to_grab_mm = GRASPING_ZONE_Y_MM - delta_y_mm # object center_y distance untill gripper 


    if CONVEYOR_SPEED_MM_S > 0:
        time_to_grab_ms = int((dist_to_grab_mm / CONVEYOR_SPEED_MM_S) * 1000) 
    else:
        time_to_grab_ms = 0


    angle_deg = extract_object_orientation(frame, obj["bbox"])
    required_force = GRIP_FORCE_MAP.get(obj["class"], 50)

    payload = {
        "cmd": "PICK",
        "cls": obj["class"],
        "x": round(xw_mm, 2),
        "y": round(dist_to_grab_mm, 2),  
        "z": zw_mm,
        "force": required_force,
        "theta": round(angle_deg, 2),
        "ttg_ms": time_to_grab_ms,
        "ts": int(time.time()) # Unix Timestamp
    }
    
    payload_json = json.dumps(payload)



    if time_to_grab_ms >= 0:
        send_serial_cmd(payload_json,arduino=arduino)

    timestamp = time.strftime("%H:%M:%S")
    formatted_payload_display = (
        
        f"[{timestamp}]  TRANSMITTING TO ROBOT ARM\n"
        "--------------------------------------------------\n"
        f"• Target Class      : {obj['class']} (ID: {obj['track_id']})\n"
        f"• Image Center (px): X={cx}, Y={cy}\n"
        f"• Trigger Line (px): Y={trigger_y_px}\n"
        f"• World Coords (mm): Xw={xw_mm:+.1f}, Remaining Dist={dist_to_grab_mm:.1f}mm\n"
        f"• Real Angle (deg) : {angle_deg:.1f}°\n"
        f"• Time-to-Grab (ms): {time_to_grab_ms} ms\n"
        f"• Gripper_Force (N): {required_force} N\n"
        
    )

    

    return formatted_payload_display, payload_json





def get_current_rates_df():
    total = system_metrics["total_count"]
    categories = list(TARGET_5_CLASSES)
    return pd.DataFrame({
        "Waste Category": categories,
        "Sorted Count (Total)": [system_metrics[k] for k in categories],
        "Share (%)": [f"{(system_metrics[k]/total)*100:.1f}%" if total > 0 else "0%" for k in categories],
        "Bin Fill Status": [f"{bin_fill_level[k]}/{BIN_CAPACITIES[k]}" for k in categories],
    })


def check_and_update_conveyor_status():
    global is_conveyor_halted, downtime_start_marker, total_downtime, log_history

    if is_emergency_stopped:
        return

    full_bins = [k for k, capacity in BIN_CAPACITIES.items() if bin_fill_level[k] >= capacity]

    if full_bins and not is_conveyor_halted:
        is_conveyor_halted = True
        
        downtime_start_marker = time.time()
        send_serial_cmd(json.dumps({"cmd": "STOP_CONVEYOR", "reason": f"BIN_FULL_{full_bins[0]}"}),arduino=arduino)
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] SYSTEM HALTED: Bin [{full_bins[0]}] is FULL! Conveyor Stopped."
        if log_msg not in log_history[:3]:
            log_history.insert(0, log_msg)
            log_history = log_history[:50]

    elif not full_bins and is_conveyor_halted:
        is_conveyor_halted = False
        if downtime_start_marker:
            total_downtime += (time.time() - downtime_start_marker)
            downtime_start_marker = None
        send_serial_cmd(json.dumps({"cmd": "START_CONVEYOR"}),arduino=arduino)
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}]  system resumed: full bins cleared. Conveyor restarted."
        log_history.insert(0, log_msg)
        log_history = log_history[:50]


def advanced_robot_logic(detected_objects, frame, frame_width, frame_height):
 
    global processed_track_ids, log_history, is_conveyor_halted, is_emergency_stopped

    if is_emergency_stopped:
        return None, "🚨 emergency stop active: Operations halted.", "🚨 HARDWARE LOCK (e-stop)",None

    if is_conveyor_halted:
        return None, "🚨 Conveyor Halted: Waiting for bin evacuation.", "Conveyor Stopped (HardWare Lock)",None

    if not detected_objects:
        return None, "Conveyor belt is empty in this frame.", None,None

    trigger_y = int(frame_height * TRIGGER_LINE_RATIO)
    valid_objects = []

    for obj in detected_objects:
        center_y = obj["center_y"]
        track_id = obj["track_id"]
        if abs(center_y - trigger_y) <= TRIGGER_TOLERANCE and track_id not in processed_track_ids:
            valid_objects.append(obj)

    if not valid_objects:
        return None, "Monitoring conveyor belt (Waiting for new item)...", None,None

    sorted_queue = sorted(
        valid_objects,
        key=lambda x: (ENVIRONMENTAL_PRIORITY.get(x["class"], 99), -x["confidence"]),
    ) # Two_Step_Verification Filter For Final_Output

    target_object = sorted_queue[0]

    processed_track_ids.add(target_object["track_id"])
    timestamp = time.strftime("%H:%M:%S")

    payload_display,payload_json = calculate_kinematics_and_send(target_object, frame, frame_width, frame_height)
    
    


    log_msg = (
        f"[{timestamp}] 🤖 COMMAND: Sort [{target_object['class']}] (ID: {target_object['track_id']})"
        f" ({target_object['confidence']:.1%}) | Center: ({target_object['center_x']}, {target_object['center_y']})"
    )

    return target_object, log_msg, payload_display,payload_json
def process_single_frame(frame):
    global log_history, prev_frame_time, processed_track_ids, track_last_seen, total_downtime, bin_fill_level, is_emergency_stopped

    df_rates = get_current_rates_df()
    df_chart = pd.DataFrame(time_series_data)

    if is_emergency_stopped:
        display_frame = np.array(frame) if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(display_frame, "EMERGENCY STOPPED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        return (
            display_frame, "0 FPS", "0.0%", "0 WPM", "emergency stop", "$0.00",
            df_rates, "\n".join(log_history[:8]), "🚨 hardware & processing halted via e-stop", df_chart, "emergency stop activated"
        )

    if frame is None or model is None:
        status_txt = "⚠️ Model file missing!" if model is None else "No Frame Input"
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return empty_frame, "0 FPS", "0.0%", "0 WPM", "0.0%", "$0.00", df_rates, status_txt, "Waiting...", df_chart, "", ""

    current_time = time.time()
    cleanup_tracking_memory(current_time)

    fps = 1.0 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0.0
    prev_frame_time = current_time
    fps_display = f"{int(fps)} FPS"

    enhanced_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    frame_h, frame_w, _ = enhanced_frame.shape
    trigger_y = int(frame_h * TRIGGER_LINE_RATIO)
    
   
    results = model.track(
        np.array(frame), imgsz=640, conf=0.35, iou=0.60, persist=True, tracker="bytetrack.yaml", verbose=False, device=device
    )[0]

    clean_frame_for_analysis = enhanced_frame.copy()

    detected_batch = []
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        cls_ids = results.boxes.cls.int().cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
            raw_class_name = model.names[cls_id]
            
            
            final_5_category = CATEGORY_5_MAP.get(raw_class_name, "Waste")
            
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track_last_seen[int(track_id)] = current_time

            detected_batch.append({
                "track_id": int(track_id),
                "raw_class": raw_class_name,        
                "class": final_5_category,         
                "confidence": float(conf),
                "bbox": (x1, y1, x2, y2),
                "center_x": cx,
                "center_y": cy,
            })

        
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(enhanced_frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(
                enhanced_frame, f"ID:{track_id} {final_5_category} ({raw_class_name}) {conf:.1%}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    with state_lock:
        check_and_update_conveyor_status()
        target, control_log, payload_display, payload_json = advanced_robot_logic(
            detected_batch, clean_frame_for_analysis, frame_w, frame_h
        )

        if target:
            cls_target = target["class"]
            system_metrics[cls_target] += 1
            system_metrics["total_count"] += 1
            system_metrics["confidence_sum"] += target["confidence"]
            system_metrics["total_revenue"] += WASTE_VALUES.get(cls_target, 0.0)

            bin_fill_level[cls_target] += 1

            sorting_timestamps.append(current_time)
            log_history.insert(0, control_log)
            log_history = log_history[:50]

    line_color = (255, 0, 0) if is_conveyor_halted else ((0, 255, 0) if target else (255, 255, 0))
    cv2.line(enhanced_frame, (0, trigger_y), (frame_w, trigger_y), line_color, 2)
    cv2.putText(enhanced_frame, "TRIGGER LINE", (10, trigger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

    if is_conveyor_halted:
        cv2.putText(enhanced_frame, "CONVEYOR HALTED", (frame_w // 4, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.putText(enhanced_frame, f"FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    total = system_metrics["total_count"]
    df_rates = get_current_rates_df()

    sorting_timestamps[:] = [t for t in sorting_timestamps if current_time - t <= 60]
    wpm_speed = f"{len(sorting_timestamps)} WPM"

    avg_conf_raw = (system_metrics["confidence_sum"] / total) if total > 0 else 0.0
    avg_conf = f"{avg_conf_raw * 100:.1f}%"

    current_dt = total_downtime + ((current_time - downtime_start_marker) if downtime_start_marker else 0.0)

    elapsed_time = max(0.001, current_time - start_time)
    operating_time = max(0.001, elapsed_time - current_dt)
    availability = max(0.0, min(1.0, operating_time / elapsed_time))

    if total > 0:
        actual_cycle_time = operating_time / total
        performance = max(0.0, min(1.0, IDEAL_CYCLE_TIME / actual_cycle_time))
    else:
        performance = 0.0
      
    quality = max(0.0, min(1.0, avg_conf_raw)) if total > 0 else 0.0

    oee_score = (availability * performance * quality) * 100.0 if total > 0 else 0.0
    oee_display = f"{oee_score:.1f}% (DT: {int(current_dt)}s)"

    revenue_display = f"${system_metrics['total_revenue']:.2f}"

    elapsed_time_int = int(elapsed_time)
    if not time_series_data["Time"] or elapsed_time_int != time_series_data["Time"][-1]:
        time_series_data["Time"].append(elapsed_time_int)
        time_series_data["Total Sorted"].append(total)
        if len(time_series_data["Time"]) > 60:
            time_series_data["Time"].pop(0)
            time_series_data["Total Sorted"].pop(0)

    df_chart = pd.DataFrame(time_series_data)
    logs_display = "\n".join(log_history[:8])

    if payload_json is not None:
        arduino_log_out.insert(0, payload_json) 
    
    if len(arduino_log_out) > 3:
        arduino_log_out.pop(-1)

    if payload_display is not None:
        payload.insert(0, payload_display) 
    
    if len(payload) > 3:
        payload.pop(-1)

    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    return (
        enhanced_frame,
        fps_display,
        avg_conf,
        wpm_speed,
        oee_display,
        revenue_display,
        df_rates,
        logs_display,
        payload[0] if payload else " ",
        df_chart,
        arduino_log_out[0] if arduino_log_out else " ",
    )


# --- E-Stop & State Handlers ---
def trigger_emergency_stop():
    global is_emergency_stopped, is_conveyor_halted, downtime_start_marker, log_history

    with state_lock:
        is_emergency_stopped = True
        if not is_conveyor_halted:
            is_conveyor_halted = True
        if not downtime_start_marker:
            downtime_start_marker = time.time()

    send_serial_cmd(json.dumps({"cmd": "EMERGENCY_STOP", "reason": "OPERATOR_E_STOP"}),arduino=arduino)

    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] 🚨 emergency stop activated by operator! All Operations Halted."
    log_history.insert(0, log_msg)
    log_history = log_history[:50]

    df_rates = get_current_rates_df()
    df_chart = pd.DataFrame(time_series_data)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_frame, "EMERGENCY STOPPED", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    return (
        empty_frame, "0 FPS", "0.0%", "0 WPM", "emergency stop", f"${system_metrics['total_revenue']:.2f}",
        df_rates, "\n".join(log_history[:8]), "🚨 HardWare & Processing Halted Via e-stop", df_chart
    )


def release_emergency_stop():
    global is_emergency_stopped, log_history

    with state_lock:
        is_emergency_stopped = False
        timestamp = time.strftime("%H:%M:%S")
        log_history.insert(0, f"[{timestamp}] emergency stop released: Re-evaluating bin status...")
        log_history = log_history[:50]
        check_and_update_conveyor_status()

    if is_conveyor_halted:
        status_msg_txt = "⚠️ E-Stop released, but a bin is still full — conveyor remains halted."
    else:
        status_msg_txt = " System fully resumed."
    log_history.insert(0, status_msg_txt)
    df_rates = get_current_rates_df()
    return "\n".join(log_history[:8]), df_rates


def empty_selected_bin(bin_name):
   
    global bin_fill_level, log_history

    if not bin_name or bin_name not in BIN_CAPACITIES:
        return get_current_rates_df(), "\n".join(log_history[:8])

    with state_lock:

        
        bin_fill_level[bin_name] = 0

        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}]  OPERATOR ACTION: Bin [{bin_name}] physically emptied. Total metrics preserved."
        log_history.insert(0, log_msg)
        log_history = log_history[:50]

        check_and_update_conveyor_status()

    return get_current_rates_df(), "\n".join(log_history[:8])




def ensure_val_dataset_exists(val_dir_path, zip_path):

    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images_val_path = os.path.join(val_dir_path, "images", "val")

    has_images = False
    if os.path.exists(val_dir_path) and os.listdir(val_dir_path):
        if os.path.exists(images_val_path):
            for ext in valid_extensions:
                if glob.glob(os.path.join(images_val_path, ext)):
                    has_images = True
                    break

        
        if has_images:
            return True, "Dataset exists and is valid."

    try:
        print("Downloading validation dataset from GitHub...")
        
        urllib.request.urlretrieve(VAL_ZIP_URL, zip_path)

       
        extract_to_dir = os.path.dirname(val_dir_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_dir)

        
        if os.path.exists(zip_path):
            os.remove(zip_path)

        return True, "Dataset downloaded successfully!"
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False, f"Failed to download dataset: {str(e)}"


def evaluate_model_benchmark():
    if not os.path.exists(DATA_YAML_PATH):
        return f"⚠️ **Benchmark Unavailable**: '{DATA_YAML_PATH}' not found in root directory."
    if model is None:
        return f"⚠️ **Benchmark Unavailable**: Model weights not loaded from '{MODEL_PATH}'."

    
    temp_dir = tempfile.gettempdir()
    zip_download_path = os.path.join(temp_dir, "val_temp.zip")
    val_extract_path = os.path.join(temp_dir, "val")

  
    success, msg = ensure_val_dataset_exists(
        val_extract_path, zip_download_path
    )
    if not success:
        return f"⚠️ **Benchmark Error**: {msg}"

    try:
        

        with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)


        yaml_data["path"] = val_extract_path

       
        yaml_data["val"] = "images/val"

      
        yaml_data["train"] = "images/train"


        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp_file:
            yaml.dump(yaml_data, tmp_file)
            temp_yaml_path = tmp_file.name

        runs_temp_dir = os.path.join(temp_dir, "yolo_runs")
        metrics = model.val(
            data=temp_yaml_path,
            verbose=False,
            imgsz=640,
            save=True,
            plots=True,
            project=runs_temp_dir, 
            name="val_results",
        )

        
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

        map50 = metrics.box.map50
        precision = metrics.box.mp
        recall = metrics.box.mr

        return (
            f"📊 **Offline Model Benchmarks (Validation Set)**:\n"
            f"• **mAP@50**: {map50:.3f}\n"
            f"• **Precision**: {precision:.3f}\n"
            f"• **Recall**: {recall:.3f}\n"
            f"* Note: Video streaming paused temporarily during benchmark evaluation."
        )

    except Exception as e:
        return f"⚠️ Benchmark Error: {str(e)}"


def handle_reconnect():
    global arduino
    if arduino and hasattr(arduino, "is_open") and arduino.is_open:
        try:
            arduino.close()
        except Exception:
            pass
    arduino,new_status = Arduino.connect()
    return new_status


def reset_system_metrics():
    global system_metrics, bin_fill_level, sorting_timestamps, log_history, time_series_data, start_time, prev_frame_time, processed_track_ids, track_last_seen, total_downtime, downtime_start_marker, is_conveyor_halted, is_emergency_stopped

    with state_lock:
        system_metrics = {
            "total_count": 0, "confidence_sum": 0.0, "total_revenue": 0.0,
        }
        for k in TARGET_5_CLASSES:
            system_metrics[k] = 0

        bin_fill_level = {k: 0 for k in BIN_CAPACITIES}

        prev_frame_time = time.time()
        total_downtime = 0.0
        downtime_start_marker = None
        is_conveyor_halted = False
        is_emergency_stopped = False

        processed_track_ids.clear()
        track_last_seen.clear()
        sorting_timestamps.clear()
        log_history.clear()
        time_series_data = {"Time": [0], "Total Sorted": [0]}
        start_time = time.time()

    reset_log = "🔄 System Metrics, Bin Fill Levels & Downtime Resetted!"
    log_history.append(reset_log)

    df_rates = get_current_rates_df()
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return (
        empty_frame, "0 FPS", "0.0%", "0 WPM", "0.0%", "$0.00", df_rates,
        reset_log, "Waiting for object on Trigger Line...", pd.DataFrame(time_series_data)
    )





def resize_with_aspect_ratio(image, target_size=(640, 640)):
    
    h, w = image.shape[:2]
    target_w, target_h = target_size

   
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

   
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

  
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_image[top:top + new_h, left:left + new_w] = resized_image

    return padded_image



def analyze_uploaded_video(video_file):
    
    video_path = video_file.name if hasattr(video_file, 'name') else video_file

    if not video_path:
        df_rates = get_current_rates_df()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yield (
            empty_frame, "0 FPS", "0.0%", "0 WPM", "0.0%", "$0.00", 
            df_rates, "⚠️ Please upload a video first!", "Waiting...", 
            pd.DataFrame({"Time": [0], "Total Sorted": [0]})
        )
        return

    cap = cv2.VideoCapture(video_path)
    
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    target_delay = 1.0 / video_fps

    frame_count = 0
    
    skip_frames=1
    if DEVICE=="cpu":
        skip_frames = 2
    
    last_processed_output = None

    while cap.isOpened():
        loop_start_time = time.time() 
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        frame_resized = resize_with_aspect_ratio(frame, target_size=(640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        if frame_count % skip_frames == 0 or last_processed_output is None:
            last_processed_output = process_single_frame(frame_rgb)

        if last_processed_output is not None:
            yield last_processed_output

        processing_time = time.time() - loop_start_time
        sleep_time = max(0.0, (target_delay * skip_frames) - processing_time)
        time.sleep(sleep_time)

    cap.release()







def toggle_source(mode):
    if mode == " Video File":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)





custom_css = """
/* set the total screen space*/
.gradio-container {
  
    max-width: 2600px !important;
    margin: 30px !important;
    padding: 20px !important;
}

/* spacing between rows and columns */
.row {
    gap: 16px !important;
}

.column {
    gap: 12px !important;
}


/* header styling*/

.header-box {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;       
    padding: 20px 28px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    margin-bottom: 24px;
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.header-icon {
    font-size: 32px;
    background: rgba(16, 185, 129, 0.15);
    padding: 10px;
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    flex-shrink: 0;
}

.header-title {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
    margin: 0;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.3;
}

.header-badge {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.4);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap; 
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    flex-shrink: 0;
}


@media (max-width: 640px) {
    .header-box {
        flex-direction: column;     
        align-items: flex-start;     
        padding: 16px 18px;          
        gap: 12px;
    }
    
    .header-icon {
        font-size: 24px;
        padding: 8px;
    }

    .header-title {
        font-size: 18px;            
    }

    .header-subtitle {
        font-size: 11px;
    }
    
    .header-badge {
        font-size: 11px;
        padding: 4px 10px;
        align-self: flex-start;      
    }
}

/* styling gradio labels */

.gradio-container label span {
    background: transparent !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0 !important;
    border: none !important;
}


/* styling gradio's boxes text*/

.gradio-container input[type="text"] {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}


.metric-card {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}



.metric-card_btns {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    max-width:150px;
}




/*.metric-card-metrics {
    background: rgba(17, 197 ,217  ,0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}*/


.btn-start_to_analyze {
    background-color: rgb(5, 150, 105) !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-start_to_analyze:hover {
    background-color: #047857 !important;
    box-shadow: 0 0 12px rgba(5, 150, 105, 0.5) !important;
}





.btn-map {
    background-color: rgba(10, 87, 56 , 0.58) !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-map:hover {
    background-color: rgba(3, 75, 40 , 0.58)   !important;
    box-shadow: 0 0 12px rgba(10, 87, 56, 0.5) !important;
}



.btn-reset-m {
    background-color: rgba(17, 197 ,217  ,0.65) !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-reset-m:hover {
    background-color: rgba(5, 180 ,185  ,0.65) !important;
    box-shadow: 0 0 12px rgba(17, 197, 217, 0.5) !important;
}






.btn_empty_bin {
    background-color: #F44336 !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn_empty_bin:hover {
    background-color: rgb(215, 40 ,43)!important;
    box-shadow: 0 0 12px rgba(244, 67, 54, 0.5) !important;
}






.btn-reconnect {
    background-color: #6366f1 !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-reconnect:hover {
    background-color: #4f46e5 !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.5) !important;
}


.btn-estop {
    background-color: #dc2626 !important;
    color: white !important;
    font-weight: 800 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-estop:hover {
    background-color: #b91c1c !important;
    box-shadow: 0 0 15px rgba(220, 38, 38, 0.6) !important;
}


.btn-resume {
    background-color: #059669 !important;
    color: white !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.btn-resume:hover {
    background-color: #047857 !important;
    box-shadow: 0 0 12px rgba(5, 150, 105, 0.5) !important;
}


#arduino-log textarea {
    background-color: #1e1e1e !important;
    color: #00ff66 !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 13px !important;
}
"""



custom_head = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
"""

with gr.Blocks(title="ECO-SORT AI | Smart Waste Automation") as demo:
    

    gr.HTML(
        """
        <div class="header-box">
            <div class="header-brand">
                <div class="header-icon">♻️</div>
                <div>
                    <div class="header-title">ECO-SORT AI <span style="font-size:12px; opacity:0.6; vertical-align:super;">v1.1</span></div>
                    <div class="header-subtitle">Industrial Vision & Robotic Sorting Dashboard</div>
                </div>
            </div>
            <div class="header-badge">
                <span class="status-dot"></span> AI AUTOMATION SYSTEM
            </div>
        </div>
        <br>
    
        """
    )

   
    with gr.Row():
        
        # BLOCK 1: System Infrastructure
        with gr.Column(scale=1, elem_classes=["metric-card"]):
            gr.Markdown("  System Status")
            arduino_status_box = gr.Textbox(
                label="Hardware Connection (Arduino)",
                value=status_msg,
                interactive=False,
            )
            metric_fps = gr.Textbox(label="FPS (Processing Speed)", value="0 FPS", interactive=False)
            metric_oee = gr.Textbox(label="Overall Effectiveness (OEE)", value="0.0%", interactive=False)






        # BLOCK 2: Performance Metrics
        with gr.Column(scale=1, elem_classes=["metric-card"]):
            gr.Markdown("  Performance Metrics")
            metric_speed = gr.Textbox(label="Real-time Speed (WPM)", value="0 WPM", interactive=False)
            metric_conf = gr.Textbox(label="Model Accuracy (Quality)", value="0.0%", interactive=False)
            metric_rev = gr.Textbox(label="Economic Value Generated", value="$0.00", interactive=False)

        # BLOCK 3: Control Center
        with gr.Column(scale=1, elem_classes=["metric-card_btns"]):
            gr.Markdown("  Control Center")
            btn_reconnect = gr.Button(" Reconnect Hardware",elem_classes=["btn-reconnect"])
            btn_estop = gr.Button(" EMERGENCY STOP",elem_classes=["btn-estop"])
            btn_release_estop = gr.Button(" Resume Operations",elem_classes=["btn-resume"])

    gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>")

    

    #  MAIN MONITORING & ANALYSIS SECTION
    with gr.Row():
        
        with gr.Column(scale=2,elem_classes=["metric-card"]):
            gr.Markdown("Live Conveyor Belt Monitoring")
            source_selector = gr.Radio(
                choices=[" Webcam", " Video File"],
                value=" Webcam",
                label="Select Input Stream Source",
                
            )
            
            input_cam = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Input Stream", visible=True
            )
            

            input_video = gr.File(
                label="Upload Conveyor Video File", 
                file_types=[".mp4", ".mpeg", ".mpg", ".avi", ".mov"], 
                visible=False
            )

            
            
            btn_analyze = gr.Button(" Start Analysis", visible=False,elem_classes=["btn-start_to_analyze"])

            output_cam = gr.Image(show_label=False, type="numpy", label="Processed Stream")

        
        with gr.Column(scale=1,elem_classes=["metric-card"]):
            gr.Markdown("Bin Capacities & Evacuation")
            rates_table = gr.Dataframe(value=get_current_rates_df(), interactive=False)
            
        
            with gr.Row():
                select_bin_dropdown = gr.Dropdown(
                    
                    choices=TARGET_5_CLASSES,
                    label="Select Bin to Clear",
                    value="Plastic",
                    scale=2,
                    elem_classes=["metric-card"]
                )

                btn_empty_bin = gr.Button("Empty Bin", scale=1,elem_classes=["btn_empty_bin"])

            gr.Markdown(" Kinematics & Serial Payload")
            kinematics_payload_box = gr.Textbox(
                lines=7, interactive=False, label="Real-time Command Payload",
            )

            gr.Markdown("Event Logs & Benchmarks")
            control_logs = gr.Textbox(lines=4, interactive=False, label="Priority Event Log")

            with gr.Row():
                btn_eval = gr.Button("Run Benchmark",elem_classes=["btn-map"])
                btn_reset = gr.Button("Reset Metrics",elem_classes=["btn-reset-m"])
            
            eval_output = gr.Markdown()



            with gr.Column():
                
                arduino_log_box = gr.Textbox(
                    label="Live Robot | Arduino Input",
                    placeholder="Waiting for object detection and command transmission...",
                    lines=12,
                    max_lines=20,
                    interactive=False, 
                    elem_id="arduino-log"
                )

    # TIME SERIES CHART SECTION 
    gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>")
    with gr.Row():
        live_chart = gr.LinePlot(
            value=pd.DataFrame(time_series_data),
            x="Time",
            y="Total Sorted",
            title="Total Sorted Waste vs. Elapsed Time (seconds)",
            x_title="Elapsed Time (Seconds)",
            y_title="Total Units Sorted",
            height=300,
            elem_classes=["metric-card"]
        )

    # Event Bindings
    source_selector.change(
        fn=toggle_source, inputs=source_selector, outputs=[input_video, input_cam, btn_analyze]
    )

    input_cam.stream(
        fn=process_single_frame,
        inputs=input_cam,
        outputs=[
            output_cam, metric_fps, metric_conf, metric_speed, metric_oee,
            metric_rev, rates_table, control_logs, kinematics_payload_box, live_chart,arduino_log_box
        ],
        queue=True,
        
    )

    btn_analyze.click(
        fn=analyze_uploaded_video,
        inputs=input_video,
        outputs=[
            output_cam, metric_fps, metric_conf, metric_speed, metric_oee,
            metric_rev, rates_table, control_logs, kinematics_payload_box, live_chart,arduino_log_box
        ],
    )

    btn_empty_bin.click(
        fn=empty_selected_bin,
        inputs=select_bin_dropdown,
        outputs=[rates_table, control_logs],
    )
    


    
    btn_eval.click(fn=evaluate_model_benchmark, inputs=None, outputs=eval_output)

    btn_reset.click(
        fn=reset_system_metrics,
        inputs=None,
        outputs=[
            output_cam, metric_fps, metric_conf, metric_speed, metric_oee, metric_rev,
            rates_table, control_logs, kinematics_payload_box, live_chart,
        ],
    )

    btn_reconnect.click(fn=handle_reconnect, inputs=None, outputs=arduino_status_box)

    btn_estop.click(
        fn=trigger_emergency_stop,
        inputs=None,
        outputs=[
            output_cam, metric_fps, metric_conf, metric_speed, metric_oee,
            metric_rev, rates_table, control_logs, kinematics_payload_box, live_chart,
        ],
    )

    btn_release_estop.click(
        fn=release_emergency_stop,
        inputs=None,
        outputs=[control_logs, rates_table],
    )




if __name__ == "__main__":
    demo.queue().launch(share=False,
    theme=gr.themes.Soft(),
    css=custom_css,
    head=custom_head)

