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
from pathlib import Path




BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
VIDEO_PATH = ASSETS_DIR / "industrial-bg.mp4"

gr.set_static_paths(paths=[ASSETS_DIR])

VIDEO_URL = f"/gradio_api/file={VIDEO_PATH.as_posix()}"

VAL_ZIP_URL = "https://github.com/abbas-pt/ExpoChallenge_AbbasLotfi/releases/download/data/val.zip"

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = str(BASE_DIR)
    return os.path.join(base_path, relative_path)

def load_config():
    config_path = get_resource_path("config.yaml")
    if not os.path.exists(config_path):
        config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

arduino_log_out = []
payload = []

CONVEYOR_DIRECTION = config["conveyor"]["direction"]

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


device = "cpu"
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    logging.exception("Device detection failed; falling back to CPU")


MODEL_PATH = get_resource_path("../model-weights/best_abbas.pt")
DEVICE = device

DATA_YAML_PATH = get_resource_path("data.yaml")

if not os.path.exists(MODEL_PATH):
    logging.warning(f"⚠️ Model file not found at '{MODEL_PATH}'. Ensure correct path before running detection.")


arduino, status_msg = None, "Disconnected"
try:
    arduino, status_msg = Arduino.connect()
except Exception as e:
    logging.exception(f"Arduino connection failed: {e}. Running in offline mode.")


model = YOLO(MODEL_PATH).to(device=device) if os.path.exists(MODEL_PATH) else None
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_CV_MAJOR, _CV_MINOR = map(int, cv2.__version__.split(".")[:2])
USES_LEGACY_ANGLE_CONVENTION = (_CV_MAJOR, _CV_MINOR) < (4, 5)
logging.info(
    f"OpenCV version detected: {cv2.__version__} -> "
    f"{'legacy' if USES_LEGACY_ANGLE_CONVENTION else 'modern'} minAreaRect angle convention in use."
)

GRIP_FORCE_MAP = config["grip_forces"]

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

WASTE_VALUES = {
    "Metal": 0.08,
    "Plastic": 0.05,
    "Glass": 0.04,
    "Paper": 0.03,
    "Waste": 0.00
}

n = config["bin_capacities"]
BIN_CAPACITIES = {k: n for k in TARGET_5_CLASSES}

TRIGGER_LINE_RATIO = config["vision"]["trigger_line_ratio"]
TRIGGER_TOLERANCE = config["vision"]["trigger_tolerance_px"]
SCALE_FACTOR_MM = config["vision"]["scale_factor_mm"]
GRASPING_ZONE_Y_MM = config["conveyor"]["grasping_zone_y_mm"]
CONVEYOR_SPEED_MM_S = config["conveyor"]["speed_mm_s"]

PLANNED_PRODUCTION_TIME = config['PLANNED_PRODUCTION_TIME']
IDEAL_CYCLE_TIME = config['IDEAL_CYCLE_TIME']

state_lock = threading.Lock()

is_emergency_stopped = False
total_downtime = 0.0
downtime_start_marker = None
is_conveyor_halted = False

track_last_seen = {}
processed_track_ids = set()
prev_frame_time = time.time()
start_time = time.time()

system_metrics = {
    "total_count": 0,
    "confidence_sum": 0.0,
    "total_revenue": 0.0,
}
for k in TARGET_5_CLASSES:
    system_metrics[k] = 0

bin_fill_level = {k: 0 for k in TARGET_5_CLASSES}
sorting_timestamps = []
log_history = []
time_series_data = {"Time": [0], "Total Sorted": [0]}


def send_serial_cmd(command_payload: str, arduino):
    if arduino and hasattr(arduino, "is_open") and arduino.is_open:
        try:
            Arduino.send_to_arduino(message=command_payload, arduino=arduino)
            return True
        except Exception as e:
            logging.error(f"Serial transmission error: {e}")
    return False


def extract_object_orientation(frame: np.ndarray, bbox: tuple) -> float:
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
    if not contours:
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
    cx, cy = obj["center_x"], obj["center_y"]
    xw_mm = (cx - (frame_width / 2)) * SCALE_FACTOR_MM
    zw_mm = config['zm']
    trigger_y_px = int(frame_height * TRIGGER_LINE_RATIO)
    if CONVEYOR_DIRECTION == "UPWARD":
        delta_y_px = trigger_y_px - cy
    else:
        delta_y_px = cy - trigger_y_px
    delta_y_mm = delta_y_px * SCALE_FACTOR_MM
    dist_to_grab_mm = GRASPING_ZONE_Y_MM - delta_y_mm
    if CONVEYOR_SPEED_MM_S > 0:
        time_to_grab_ms = int((dist_to_grab_mm / CONVEYOR_SPEED_MM_S) * 1000)
    else:
        time_to_grab_ms = 0
    angle_deg = extract_object_orientation(frame, obj["bbox"])
    required_force = GRIP_FORCE_MAP.get(obj["class"], 50)
    payload_dict = {
        "cmd": "PICK",
        "cls": obj["class"],
        "x": round(xw_mm, 2),
        "y": round(dist_to_grab_mm, 2),
        "z": zw_mm,
        "force": required_force,
        "theta": round(angle_deg, 2),
        "ttg_ms": time_to_grab_ms,
        "ts": int(time.time())
    }
    payload_json = json.dumps(payload_dict)
    if time_to_grab_ms >= 0:
        send_serial_cmd(payload_json, arduino=arduino)
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
        send_serial_cmd(json.dumps({"cmd": "STOP_CONVEYOR", "reason": f"BIN_FULL_{full_bins[0]}"}), arduino=arduino)
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
        send_serial_cmd(json.dumps({"cmd": "START_CONVEYOR"}), arduino=arduino)
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}]  system resumed: full bins cleared. Conveyor restarted."
        log_history.insert(0, log_msg)
        log_history = log_history[:50]


def advanced_robot_logic(detected_objects, frame, frame_width, frame_height):
    global processed_track_ids, log_history, is_conveyor_halted, is_emergency_stopped
    if is_emergency_stopped:
        return None, "🚨 emergency stop active: Operations halted.", "🚨 HARDWARE LOCK (e-stop)", None
    if is_conveyor_halted:
        return None, "🚨 Conveyor Halted: Waiting for bin evacuation.", "Conveyor Stopped (HardWare Lock)", None
    if not detected_objects:
        return None, "Conveyor belt is empty in this frame.", None, None
    trigger_y = int(frame_height * TRIGGER_LINE_RATIO)
    valid_objects = []
    for obj in detected_objects:
        center_y = obj["center_y"]
        track_id = obj["track_id"]
        if abs(center_y - trigger_y) <= TRIGGER_TOLERANCE and track_id not in processed_track_ids:
            valid_objects.append(obj)
    if not valid_objects:
        return None, "Monitoring conveyor belt (Waiting for new item)...", None, None
    sorted_queue = sorted(
        valid_objects,
        key=lambda x: (ENVIRONMENTAL_PRIORITY.get(x["class"], 99), -x["confidence"]),
    )
    target_object = sorted_queue[0]
    processed_track_ids.add(target_object["track_id"])
    timestamp = time.strftime("%H:%M:%S")
    payload_display, payload_json = calculate_kinematics_and_send(target_object, frame, frame_width, frame_height)
    log_msg = (
        f"[{timestamp}] 🤖 COMMAND: Sort [{target_object['class']}] (ID: {target_object['track_id']})"
        f" ({target_object['confidence']:.1%}) | Center: ({target_object['center_x']}, {target_object['center_y']})"
    )
    return target_object, log_msg, payload_display, payload_json


# ----- Helper to build status HTML -----
def _status_html(ok: bool, label_ok: str, label_bad: str) -> str:
    cls = "status-optimal" if ok else "status-alert"
    icon = "✓" if ok else "⚠"
    text = label_ok if ok else label_bad
    return f'<div class="metric-status {cls}">{icon} Status: {text}</div>'


def process_single_frame(frame):
    global log_history, prev_frame_time, processed_track_ids, track_last_seen, total_downtime, bin_fill_level, is_emergency_stopped
    df_rates = get_current_rates_df()
    df_chart = pd.DataFrame(time_series_data)
    
    # Default statuses (idle) in case of early exit
    fps_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    conf_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    oee_html = '<div class="metric-status status-idle">— Status: Idle</div>'

    if is_emergency_stopped:
        display_frame = np.array(frame) if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(display_frame, "EMERGENCY STOPPED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        return (
            display_frame, "0.0s", "0.0%", "0 WPM", "0.0%", "$0.00",
            df_rates, "\n".join(log_history[:8]), "🚨 hardware & processing halted via e-stop", df_chart,
            "emergency stop activated", fps_html, conf_html, oee_html
        )
    if frame is None or model is None:
        status_txt = "⚠️ Model file missing!" if model is None else "No Frame Input"
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return empty_frame, "0.0s", "0.0%", "0 WPM", "0.0%", "$0.00", df_rates, status_txt, "Waiting...", df_chart, "", fps_html, conf_html, oee_html
    
    current_time = time.time()
    cleanup_tracking_memory(current_time)
    fps = 1.0 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0.0
    prev_frame_time = current_time
    fps_display = f"{fps:.1f}s"   # now shows cycle time, not FPS (but we keep as seconds)
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
    oee_display = f"{oee_score:.1f}%"
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

    # Build dynamic statuses
    fps_html = _status_html(fps > 0, "Nominal", "Stalled")
    conf_html = _status_html(avg_conf_raw >= 0.5, "Optimal", "Low")
    oee_html = _status_html(oee_score >= 60.0, "Stable", "Degraded")

    return (
        enhanced_frame,
        fps_display,       # cycle time
        avg_conf,
        wpm_speed,
        oee_display,
        revenue_display,
        df_rates,
        logs_display,
        payload[0] if payload else " ",
        df_chart,
        arduino_log_out[0] if arduino_log_out else " ",
        fps_html,
        conf_html,
        oee_html
    )


def trigger_emergency_stop():
    global is_emergency_stopped, is_conveyor_halted, downtime_start_marker, log_history
    with state_lock:
        is_emergency_stopped = True
        if not is_conveyor_halted:
            is_conveyor_halted = True
        if not downtime_start_marker:
            downtime_start_marker = time.time()
    send_serial_cmd(json.dumps({"cmd": "EMERGENCY_STOP", "reason": "OPERATOR_E_STOP"}), arduino=arduino)
    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] 🚨 emergency stop activated by operator! All Operations Halted."
    log_history.insert(0, log_msg)
    log_history = log_history[:50]
    df_rates = get_current_rates_df()
    df_chart = pd.DataFrame(time_series_data)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_frame, "EMERGENCY STOPPED", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    # Statuses for emergency
    fps_html = '<div class="metric-status status-alert">⚠ Status: Halted</div>'
    conf_html = '<div class="metric-status status-alert">⚠ Status: Halted</div>'
    oee_html = '<div class="metric-status status-alert">⚠ Status: Halted</div>'
    # Return 14 items: include an empty string for arduino_log (index 10)
    return (
        empty_frame, "0.0s", "0.0%", "0 WPM", "0.0%", f"${system_metrics['total_revenue']:.2f}",
        df_rates, "\n".join(log_history[:8]), "🚨 HardWare & Processing Halted Via e-stop", df_chart,
        "", fps_html, conf_html, oee_html
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
        status_msg_txt = "✅ System fully resumed."
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
    success, msg = ensure_val_dataset_exists(val_extract_path, zip_download_path)
    if not success:
        return f"⚠️ **Benchmark Error**: {msg}"
    try:
        with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = val_extract_path
        yaml_data["val"] = "images/val"
        yaml_data["train"] = "images/train"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_file:
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
    arduino, new_status = Arduino.connect()
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
    # Idle statuses after reset
    fps_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    conf_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    oee_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    return (
        empty_frame, "0.0s", "0.0%", "0 WPM", "0.0%", "$0.00", df_rates,
        reset_log, "Waiting for object on Trigger Line...", pd.DataFrame(time_series_data), "",
        fps_html, conf_html, oee_html
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
    # gr.Video passes the file path directly as a string
    video_path = video_file if isinstance(video_file, str) else str(video_file)
    
    # idle statuses
    idle_html = '<div class="metric-status status-idle">— Status: Idle</div>'
    if not video_path:
        df_rates = get_current_rates_df()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yield (
            empty_frame,
            "0.0s",
            "0.0%",
            "0 WPM",
            "0.0%",
            "$0.00",
            df_rates,
            "⚠️ Please upload a video first!",
            "Waiting...",
            pd.DataFrame({"Time": [0], "Total Sorted": [0]}),
            "",
            idle_html, idle_html, idle_html
        )
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        df_rates = get_current_rates_df()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yield (
            empty_frame,
            "0.0s",
            "0.0%",
            "0 WPM",
            "0.0%",
            "$0.00",
            df_rates,
            f"⚠️ Unable to open video: {video_path}",
            "Waiting...",
            pd.DataFrame({"Time": [0], "Total Sorted": [0]}),
            "",
            idle_html, idle_html, idle_html
        )
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    target_delay = 1.0 / video_fps
    frame_count = 0
    skip_frames = 1
    if DEVICE == "cpu":
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


def toggle_source(mode: str) -> tuple:
    
    if mode.strip() == "Video File":
        return (
            gr.update(visible=False),  # webcam
            gr.update(visible=True),   # video_row
            gr.update(visible=True),  # start button
        )
    return (
        gr.update(visible=True),       # webcam
        gr.update(visible=False),      # video_row
        gr.update(visible=False),       # start button
    )


#  MODERN INDUSTRIAL DASHBOARD CSS 
custom_css = """
:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: rgba(30, 41, 59, 0.6);
    --accent-primary: #3b82f6;
    --accent-secondary: #8b5cf6;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --accent-danger: #ef4444;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: rgba(148, 163, 184, 0.1);
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.15);
    --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.2);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;

    /* === NEW INDUSTRIAL COLOURS === */
    --industrial-cyan: #38d5f5;
    --industrial-green: #42d883;
    --industrial-amber: #f5b942;
    --industrial-red: #ff5c67;
    --industrial-surface: #101827;
    --industrial-surface-raised: #172235;
    --industrial-border: rgba(148, 163, 184, 0.18);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

body {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    font-size: 16px;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

/* Modern Header */
.modern-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 32px;
    padding: 20px 24px;
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
}

.header-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.55);
}

.header-actions {
    display: flex;
    gap: 16px;
    align-items: center;
}

.search-bar {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 10px 16px;
    color: var(--text-primary);
    width: 250px;
}

/* Metric Cards */
.metric-card-modern {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
}

.metric-card-modern:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--accent-primary);
}

.metric-card-spaced {
    margin-top: 16px !important;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 8px 0;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
}

.metric-label {
    font-size: 15px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 500;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
}

.metric-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
}

.status-optimal {
    background: rgba(16, 185, 129, 0.2);
    color: var(--accent-success);
}

.status-stable {
    background: rgba(59, 130, 246, 0.2);
    color: var(--accent-primary);
}

/* === NEW statuses for idle and alert === */
.status-idle {
    background: rgba(148, 163, 184, 0.15);
    color: #9aa0a6;
}
.status-alert {
    background: rgba(239, 68, 68, 0.2);
    color: #e05252;
}

/* Main Visualization Area – using ID #main_viz */
#main_viz {
    position: relative !important;
    overflow: hidden !important;
    padding: 24px !important;
    box-sizing: border-box;
    border: 1px solid rgba(56, 213, 245, 0.24) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: var(--radius-lg) !important;
}

/* Left accent line – keep it */
#main_viz::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: var(--industrial-cyan);
    box-shadow: 0 0 16px rgba(56, 213, 245, 0.60);
}

/* Explicit badge – no pseudo-element duplication */
#main_viz .live-badge {
    position: absolute;
    top: 35px;
    right: 15px;
    z-index: 4;
    padding: 4px 8px;
    border: 1px solid rgba(56, 213, 245, 0.28);
    border-radius: 999px;
    color: var(--industrial-cyan);
    background: rgba(56, 213, 245, 0.08);
    font-family: ui-monospace, Consolas, monospace;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.08em; 
    pointer-events: none;
}

/* Suppress any accidental pseudo-elements from child elements */
#main_viz > ::before { content: none !important; }

/* Control row inside #main_viz */
#main_viz .control-row {
    display: flex !important;
    flex-wrap: wrap;
    gap: 12px;
    width: 100%;
    margin-top: 16px;
}

#main_viz .control-row > button.modern-btn {
    flex: 1 1 0;
    min-width: 0 !important;      /* overrides Gradio's default 160px */
    white-space: nowrap;
    padding: 12px 16px !important;
    font-size: 15px !important;
}

/* Dashboard Grid */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
    margin-bottom: 24px;
}

/* System Logs */
.system-logs-container {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    padding: 24px;
    box-shadow: var(--shadow-md);
}

.log-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: var(--radius-md);
    margin-bottom: 8px;
    background: rgba(15, 23, 42, 0.4);
    transition: all 0.2s ease;
}

.log-item:hover {
    background: rgba(59, 130, 246, 0.1);
}

.log-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

.badge-system { background: rgba(59, 130, 246, 0.2); color: var(--accent-primary); }
.badge-hardware { background: rgba(139, 92, 246, 0.2); color: var(--accent-secondary); }
.badge-security { background: rgba(239, 68, 68, 0.2); color: var(--accent-danger); }

/* Buttons */
.modern-btn {
    background: var(--gradient-1);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: var(--radius-md);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.modern-btn:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.btn-emergency {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.btn-resume {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

/* Hide default Gradio elements */
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-group {
    background: transparent !important;
    border: none !important;
}

/* Custom styling for Gradio components */
.gr-button {
    background: var(--gradient-1) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

.gr-input, .gr-textbox, .gr-dataframe {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
}



.system-logs-container,
.production-core {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 260px;
    background: linear-gradient(145deg, rgba(23, 34, 53, 0.72), rgba(15, 24, 39, 0.60)) !important;
    backdrop-filter: blur(12px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(12px) saturate(120%) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    padding: 14px !important;
    border-radius: 16px !important;
}

.core-title {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--text-primary);
    margin: 0;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.55);
}

.core-desc {
    font-size: 0.82rem;
    line-height: 1.45;
    color: var(--text-secondary);
    margin: 0;
    min-height: 2.4em;   /* reserve two lines so cards stay aligned */
}

.status-readonly textarea {
    font-family: var(--font-mono, ui-monospace, monospace);
    font-size: 0.8rem;
    line-height: 1.5;
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 12px;
    resize: none;
}

.core-link {
    color: var(--accent-primary);
    text-decoration: none;
    font-size: 0.8rem;
    margin-top: auto;   /* push link to the bottom of the card */
}

/* Make both columns in the third row equal height */
.gradio-container .row:last-of-type .column {
    display: flex;
    flex-direction: column;
}

.gradio-container .row:last-of-type .column > .gr-group {
    flex: 1;
}

/* Compact layout overrides */
.gradio-container {
    max-width: 1480px !important;
    padding: 14px 18px !important;
    margin: 0 auto !important;
}

.gradio-container .row,
.gradio-container .gr-row {
    gap: 12px !important;
    margin-top: 0 !important;
    margin-bottom: 12px !important;
}

.gradio-container .column,
.gradio-container .gr-column {
    gap: 10px !important;
}

.modern-header {
    min-height: 62px !important;
    margin-bottom: 14px !important;
    padding: 10px 16px !important;
    border-radius: 16px !important;
}

.header-title {
    font-size: 24px !important;
    line-height: 1.1 !important;
}

.system-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    min-height: 34px;
    padding: 0 11px;
    border: 1px solid rgba(66, 216, 131, 0.30);
    border-radius: 999px;
    color: #86efac;
    background: rgba(66, 216, 131, 0.08);
    font-family: ui-monospace, Consolas, monospace;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    white-space: nowrap;
}

.system-status-chip .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--industrial-green);
    box-shadow: 0 0 10px var(--industrial-green);
    animation: onlinePulse 2s ease-in-out infinite;
}

@keyframes onlinePulse {
    0%, 100% {
        opacity: 0.65;
        transform: scale(0.9);
    }
    50% {
        opacity: 1;
        transform: scale(1.15);
    }
}

.metric-card-modern,
.system-logs-container,
.production-core {
    padding: 14px !important;
    border-radius: 16px !important;
}

.metric-card-spaced {
    margin-top: 4px !important;
}

.metric-card-modern {
    min-height: 0 !important;
}

.metric-card-modern .metric-value {
    margin: 3px 0 !important;
    font-size: 25px !important;
    line-height: 1.1 !important;
}

.metric-card-modern .metric-status {
    margin-top: 6px !important;
}

.metric-card-modern .block,
.metric-card-modern .form,
.metric-card-modern .wrap {
    margin-top: 3px !important;
    margin-bottom: 3px !important;
}

.modern-image {
    border: 1px solid rgba(56, 213, 245, 0.34) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 1px rgba(56, 213, 245, 0.06), 0 12px 30px rgba(0, 0, 0, 0.24) !important;
}

.gradio-container .modern-btn {
    min-height: 42px !important;
    border-radius: 11px !important;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease !important;
}

.gradio-container .modern-btn:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.08) !important;
}

.btn-emergency {
    color: white !important;
    background: var(--industrial-red) !important;
    box-shadow: 0 8px 20px rgba(255, 92, 103, 0.22) !important;
}

.btn-resume {
    color: #d1fae5 !important;
    background: rgba(66, 216, 131, 0.20) !important;
    border: 1px solid rgba(66, 216, 131, 0.40) !important;
}

.gradio-container > .row:first-of-type > .column:first-child {
    flex: 2.2 1 0 !important;
}

.gradio-container > .row:first-of-type > .column:last-child {
    flex: 1 1 0 !important;
}

.modern-image {
    min-height: 360px !important;
}

.metric-card-modern {
    position: relative !important;
    overflow: hidden !important;
}

.metric-card-modern::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: var(--industrial-cyan);
    opacity: 0.9;
}

.metrics-column .metric-card-modern:nth-of-type(1)::before {
    background: var(--industrial-cyan);
}
.metrics-column .metric-card-modern:nth-of-type(2)::before {
    background: var(--industrial-green);
}
.metrics-column .metric-card-modern:nth-of-type(3)::before {
    background: var(--industrial-amber);
}
.metrics-column .metric-card-modern:nth-of-type(4)::before {
    background: var(--industrial-cyan);
}
.metrics-column .metric-card-modern:nth-of-type(5)::before {
    background: var(--industrial-green);
}

.status-optimal {
    color: var(--industrial-green) !important;
    background: rgba(66, 216, 131, 0.12) !important;
    border: 1px solid rgba(66, 216, 131, 0.24) !important;
}

.status-stable {
    color: var(--industrial-amber) !important;
    background: rgba(245, 185, 66, 0.12) !important;
    border: 1px solid rgba(245, 185, 66, 0.24) !important;
}

.status-danger,
.status-error {
    color: var(--industrial-red) !important;
    background: rgba(255, 92, 103, 0.12) !important;
    border: 1px solid rgba(255, 92, 103, 0.24) !important;
}

.main-viz-container .file-preview,
.main-viz-container .upload-container {
    border: 1px dashed rgba(56, 213, 245, 0.42) !important;
    border-radius: 12px !important;
    background: radial-gradient(circle at center, rgba(56, 213, 245, 0.08), transparent 65%) !important;
    transition: border-color 160ms ease, background 160ms ease !important;
}

.main-viz-container .file-preview:hover,
.main-viz-container .upload-container:hover {
    border-color: var(--industrial-cyan) !important;
    background: radial-gradient(circle at center, rgba(56, 213, 245, 0.14), transparent 65%) !important;
}



.fullscreen-bg-video {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    background: #8b9298;
}

.fullscreen-bg-video video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.50;
    filter: saturate(0.88) brightness(0.98) contrast(1.00);
}

.fullscreen-bg-video::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(226, 232, 240, 0.05) 0%, rgba(15, 23, 42, 0.14) 52%, rgba(15, 23, 42, 0.24) 100%);
}

.gradio-container {
    position: relative !important;
    z-index: 1 !important;
    background: transparent !important;
}

/* Keep dashboard above the video */
.modern-header,
#main_viz,
.metric-card-modern,
.system-logs-container,
.production-core {
    position: relative;
    z-index: 1;
}



.section-title,
.header-title,
.metric-label,
.core-title,
.core-desc {
    color: #f8fafc;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.55);
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.4px;
}

.core-desc,
.metric-label {
    color: #e2e8f0 !important;
}

/* =========================================================
   UNIFY ALL HEADERS – MAKE THEM LOOK IDENTICAL
   ========================================================= */
.core-title,
.section-title,
#arduino-log-header,
div[class*="section-title"] h3,
div[class*="section-title"] p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding-bottom: 4px !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.55) !important;
    line-height: 1.2 !important;
    text-transform: none !important;
}

/* Remove extra top margin from Markdown h3 inside .section-title */
.section-title h3 {
    margin-top: 0 !important;
    display: inline-block !important;
}

/* Subtitle style if needed */
.core-subtitle {
    font-size: 0.85rem !important;
    opacity: 0.7;
    font-weight: 400 !important;
}
"""

# MODERN DASHBOARD UI 
with gr.Blocks() as demo:
    # FULL‑SCREEN BACKGROUND VIDEO 
    gr.HTML(
        f"""
        <div class="fullscreen-bg-video" aria-hidden="true">
            <video
                id="industrial-background-video"
                autoplay
                muted
                loop
                playsinline
                preload="auto"
            >
                <source
                    src="{VIDEO_URL}"
                    type="video/mp4"
                >
            </video>
        </div>
        """
    )

    # Modern Header 
    gr.HTML("""
    <div class="modern-header">
        <h1 class="header-title">TraceSort Dashboard</h1>
        <div class="header-actions">
            <input type="text" class="search-bar" placeholder="Search...">
            <div class="system-status-chip">
                <span class="status-dot"></span>
                SYSTEM ONLINE
            </div>
            <div class="header-avatar" style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">AL</div>
        </div>
    </div>
    """)
    
    with gr.Row():
        # Left Column - Main Visualization
        with gr.Column(scale=2):
            # Main Video Feed Container – ID added, explicit badge via HTML
            with gr.Group(elem_id="main_viz", elem_classes="main-viz-container"):
                # Explicit LIVE FEED badge
                gr.HTML('<span class="live-badge">LIVE FEED</span>')
                # Input source selector
                with gr.Row():
                    source_radio = gr.Radio(
                        choices=["Webcam", "Video File"],
                        value="Webcam",
                        label="Input Source",
                        elem_classes="modern-input",
                    )
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    height=390,
                    width=640,
                    label="",
                    show_label=False,
                    visible=True,
                    elem_classes="modern-image"
                )
                
                with gr.Row(visible=False) as video_row:
                    video_input = gr.File(
                        label="Source Video",
                        file_types=[".mp4", ".mpeg", ".mpg", ".avi", ".mov"],
                        interactive=True,
                        height=360,
                        elem_classes="source-video",
                    )
                    processed_view = gr.Image(
                        label="AI Detection",
                        type="numpy",
                        height=360,
                        interactive=False,
                    )
                    
                    
                # Controls inside the card – with custom row class
                with gr.Row(elem_classes="control-row"):
                    
                    start_btn = gr.Button("Analyzing", elem_classes="modern-btn", variant="primary",visible=False)
                    benchmark_btn = gr.Button("Benchmark", elem_classes="modern-btn")
                    reconnect_btn = gr.Button("Reset Arduino", elem_classes="modern-btn")
        
        # Right Column - Metrics
        with gr.Column(
            scale=1,
            elem_classes=["metrics-column"]
        ):
            gr.Markdown("### Real-time Metrics", elem_classes="section-title")
            
            # Cycle Velocity
            with gr.Group(elem_classes="metric-card-modern"):
                gr.Markdown('<div class="metric-label">Cycle Velocity</div>')
                fps_display = gr.Textbox(
                    label="", show_label=False,
                    value="0.0s", interactive=False,
                    elem_classes="metric-value"
                )
                fps_status = gr.HTML('<div class="metric-status status-idle">— Status: Idle</div>')
            
            # Motion Accuracy
            with gr.Group(elem_classes="metric-card-modern"):
                gr.Markdown('<div class="metric-label">Motion Accuracy</div>')
                conf_display = gr.Textbox(
                    label="", show_label=False,
                    value="0.0%", interactive=False,
                    elem_classes="metric-value"
                )
                conf_status = gr.HTML('<div class="metric-status status-idle">— Status: Idle</div>')
            
            # System OEE
            with gr.Group(elem_classes="metric-card-modern"):
                gr.Markdown('<div class="metric-label">System OEE</div>')
                oee_display = gr.Textbox(
                    label="", show_label=False,
                    value="0.0%", interactive=False,
                    elem_classes="metric-value"
                )
                oee_status = gr.HTML('<div class="metric-status status-idle">— Status: Idle</div>')
            
            # Additional Metrics
            with gr.Group(elem_classes="metric-card-modern"):
                gr.Markdown('<div class="metric-label">Sorting Speed</div>')
                wpm_display = gr.Textbox(
                    label="", show_label=False,
                    value="0 WPM", interactive=False,
                    elem_classes="metric-value"
                )
            
            with gr.Group(elem_classes="metric-card-modern"):
                gr.Markdown('<div class="metric-label">Total Revenue</div>')
                revenue_display = gr.Textbox(
                    label="", show_label=False,
                    value="$0.00", interactive=False,
                    elem_classes="metric-value"
                )
    
    # Second Row - Charts and Bin Status
    with gr.Row():
        with gr.Column(scale=2):
            # Performance Analytics
            with gr.Group(elem_classes="chart-container"):
                gr.Markdown("### Performance Analytics", elem_classes="section-title")
                chart_plot = gr.LinePlot(
                    x="Time",
                    y="Total Sorted",
                    title="Total Sorted over Time",
                    height=300,
                    label="Total Sorted"
                )
        
        with gr.Column(scale=1):
            with gr.Group(elem_classes="system-logs-container"):
                gr.Markdown("### Bin Status", elem_classes="section-title")
                bin_df = gr.Dataframe(
                    headers=["Waste Category", "Sorted Count (Total)", "Share (%)", "Bin Fill Status"],
                    interactive=False,
                    value=get_current_rates_df(),
                    elem_classes="modern-table"
                )
                with gr.Row():
                    bin_select = gr.Dropdown(
                        choices=TARGET_5_CLASSES,
                        label="Select bin to empty",
                        value=TARGET_5_CLASSES[0]
                    )
                    empty_bin_btn = gr.Button("Empty Bin", elem_classes="modern-btn")
    
    # Third Row - System Logs and Production Core 
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="system-logs-container"):
                gr.Markdown('<div class="core-title">System Logs</div>')
                gr.Markdown('<div class="core-desc">Real-time event stream from the sorting pipeline</div>')
                log_box = gr.Textbox(
                    label="", show_label=False, lines=5,
                    interactive=False, elem_classes="status-readonly",
                )
                gr.Markdown(
                    '<a href="#" class="core-link">See All →</a>'
                )
        
        with gr.Column(scale=1):
            with gr.Group(elem_classes="production-core"):
                gr.Markdown('<div class="core-title">Production Core</div>')
                gr.Markdown('<div class="core-desc">Streamline automated cycles with real-time hardware analytics</div>')
                payload_box = gr.Textbox(
                    label="", show_label=False, lines=5,
                    interactive=False, elem_classes="status-readonly",
                )
            # Arduino Log placed inside the same column, below Production Core
            with gr.Group(elem_classes=["metric-card-modern", "metric-card-spaced"]):
                gr.Markdown("### Arduino Log", elem_classes="section-title")
                arduino_log_box = gr.Textbox(
                    label="", show_label=False, lines=3,
                    interactive=False, elem_classes="status-readonly",
                    elem_id="arduino-log"
                )
    
    # Emergency Controls 
    with gr.Row():
        estop_btn = gr.Button("EMERGENCY STOP", elem_classes="modern-btn btn-emergency", size="lg")
        resume_btn = gr.Button("Resume", elem_classes="modern-btn btn-resume", size="lg")
        reset_btn = gr.Button("Reset Metrics", elem_classes="modern-btn", size="lg")

    # Output placeholders
    video_output = gr.State()
    reconnect_status = gr.State()

    # ----- Updated outputs list including the three status HTML components -----
    outputs = [
        webcam,                
        fps_display,           
        conf_display,          
        wpm_display,           
        oee_display,           
        revenue_display,       
        bin_df,               
        log_box,               
        payload_box,           
        chart_plot,           
        arduino_log_box,       
        fps_status,            
        conf_status,           
        oee_status             
    ]
    
    video_outputs = outputs.copy()
    video_outputs[0] = processed_view

    # Events
    webcam.stream(
        fn=process_single_frame,
        inputs=[webcam],
        outputs=outputs,queue=True
    )

    start_btn.click(
        fn=analyze_uploaded_video,
        inputs=[video_input],
        outputs=video_outputs
    )

    source_radio.change(
        fn=toggle_source,
        inputs=[source_radio],
        outputs=[webcam, video_row, start_btn]
    )

    estop_btn.click(
        fn=trigger_emergency_stop,
        inputs=[],
        outputs=outputs
    )

    resume_btn.click(
        fn=release_emergency_stop,
        inputs=[],
        outputs=[log_box, bin_df]
    )

    empty_bin_btn.click(
        fn=empty_selected_bin,
        inputs=[bin_select],
        outputs=[bin_df, log_box]
    )

    reset_btn.click(
        fn=reset_system_metrics,
        inputs=[],
        outputs=outputs
    )

    benchmark_btn.click(
        fn=evaluate_model_benchmark,
        inputs=[],
        outputs=[log_box]
    )

    reconnect_btn.click(
        fn=handle_reconnect,
        inputs=[],
        outputs=[reconnect_status]
    ).then(
        fn=lambda x: f"Reconnection status: {x}",
        inputs=[reconnect_status],
        outputs=[log_box]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8001,
        share=False,
        css=custom_css,
        theme=gr.themes.Soft(),
        allowed_paths=[str(ASSETS_DIR)]
    )
