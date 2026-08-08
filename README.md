# ♻️ Smart Waste Sorting Robot

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI_Model-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![GUI](https://img.shields.io/badge/Dashboard-Gradio_%2B_PyWebview-orange.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Hardware-Arduino_Serial_Bridge-red.svg)](https://www.arduino.cc/)

**Vision-guided waste classification system combining AI detection, industrial control, and RAG-powered documentation.** The project delivers two deployment modes—an industrial dashboard for real-time robot control and a FastAPI backend for scalable API integration—both powered by custom-trained YOLO models.

Developed for **Innoverse Competition** by Reza Esmaeili Mood (Lead), Abbas (AI Training), Ara (Documentation), and Sina (Support).

---

## 💡 Overview

This project tackles automated waste sorting through computer vision and robotic control. At its core:

- **AI Vision:** Custom-trained YOLO11n detects and tracks waste items across 18 fine-grained classes (TACO dataset), mapped to 5 operational categories for industrial routing
- **Dual Architecture:** Industrial dashboard (ECO-SORT AI) for robot control + FastAPI backend for API-driven integration
- **Hardware Bridge:** JSON-based serial protocol for Arduino-controlled robotic arms with kinematic calculations
- **Smart Documentation:** RAG chatbot provides on-the-fly answers from 12 technical knowledge base documents
- **OEE Monitoring:** Real-time industrial KPIs (Availability × Performance × Quality)

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV (headless) · Gradio · PyWebview · SQLite · Serial Communication

---

## 🏗️ System Architecture
```text
┌─────────────────────────────────────────────────────────────────┐
│                    Camera Feed / Video Input                    │
└────────────────────────────┬────────────────────────────────────┘
│
┌────────────┴────────────┐
│                         │
┌──────▼──────┐          ┌──────▼──────┐
│  ECO-SORT AI │          │   FastAPI   │
│  (Dashboard) │          │   Backend   │
│              │          │             │
│ • Gradio UI  │          │ • REST API  │
│ • PyWebview  │          │ • Auth Keys │
│ • OEE Metrics│          │ • RAG Chat  │
│ • Real-time  │          │ • SQLite DB │
└──────┬──────┘          └──────┬──────┘
│                         │
│                         │
┌──────▼──────────────────────┬──┘
│                             │
┌────▼────┐                  ┌────▼─────┐
│ Arduino │                  │ Clients  │
│ Robot   │                  │ (API)    │
│ Control │                  │          │
└─────────┘                  └──────────┘

**Core AI Pipeline (Shared):**
text
Frame Input → Letterbox (640x640) → YOLO11n Detection → ByteTrack Tracking
→ 18-to-5 Class Mapping → Trigger-Line Filter → Priority Queue
→ Kinematics Engine (X, Y, Z, θ, TTG) → Output (Serial JSON / API Response)

---

## 🤖 AI Model Configuration

### YOLO11n Training (Google Colab)
- **Dataset:** TACO (Trash Annotations in Context) — 18 litter classes
- **Epochs:** 100 (Early stopping: 15) | **Image Size:** 640×640 | **Batch:** 16
- **Augmentation:** Mosaic (1.0), MixUp (0.15), Copy-Paste (0.10), Rotation (±15°), Perspective (0.0005), HSV Jitter, Random Erasing (0.40)

### 18-to-5 Class Mapping

| TACO Classes (18) | Operational Category | Priority | Gripper Force |
|:------------------|:---------------------|:---------|:--------------|
| Aluminium foil, Can, Pop tab | **Metal** | 1 (Highest) | 70 N |
| Bottle cap, Bottle, Lid, Other plastic, Plastic bag, Plastic container, Straw | **Plastic** | 2 | 50 N |
| Broken glass | **Glass** | 3 | 20 N |
| Carton, Cup, Paper | **Paper** | 4 | 85 N |
| Cigarette, Other litter, Styrofoam piece, Unlabeled litter | **Waste** | 5 (Lowest) | 60 N |

**Rationale:** Environmental priority queue resolves conflicts when multiple items cross the trigger line simultaneously (*Metal > Plastic > Glass > Paper > Waste*).

---

## 🌱 Documentation Index

All documents live under [`backend/docs/kb/`](backend/docs/kb/).

### 🧩 Core Architecture

| # | 📄 Document | Description |
|---|---|---|
| 01 | [**Overview**](backend/docs/kb/01-overview.md) | Goals, components & high-level diagram |
| 02 | [**Classes**](backend/docs/kb/02-classes.md) | Specs for the five waste classes |
| 03 | [**Pipeline**](backend/docs/kb/03-pipeline.md) | Data flow, inference & routing |
| 04 | [**API**](backend/docs/kb/04-api.md) | REST endpoints, request / response formats |

### ⚛️ AI & Models

| # | 📄 Document | Description |
|---|---|---|
| 05 | [**Training**](backend/docs/kb/05-training.md) | Datasets, methodology & evaluation |
| 10 | [**YOLO Versions**](backend/docs/kb/10-yolo-versions.md) | Version comparison & selection rationale |
| 11 | [**Alternative Models**](backend/docs/kb/11-alternative-models.md) | Other CV/DL approaches examined |
| 12 | [**Future Models**](backend/docs/kb/12-future-models.md) | Roadmap for research & optimisation |

### 💬 Chatbot & Knowledge Base

| # | 📄 Document | Description |
|---|---|---|
| 06 | [**RAG & Chat**](backend/docs/kb/06-rag-and-chat.md) | Architecture & generation pipeline |
| 07 | [**FAQ**](backend/docs/kb/07-faq.md) | Common questions & troubleshooting tips |

### 🛡️ Operations & Security

| # | 📄 Document | Description |
|---|---|---|
| 08 | [**Failure Modes**](backend/docs/kb/08-failure-modes.md) | Known issues & recovery strategies |
| 09 | [**Security & Keys**](backend/docs/kb/09-security-and-keys.md) | Auth workflow, API-key handling |

---

## 🔌 Hardware Communication Protocol

Commands transmit over USB Serial (**9600 Baud**) as newline-terminated JSON payloads.

### Sample `PICK` Command
json
{
  "cmd": "PICK",
  "cls": "Plastic",
  "x": -42.5,
  "y": 187.3,
  "z": -150.0,
  "force": 50,
  "theta": 12.4,
  "ttg_ms": 1248,
  "ts": 1730000000
}

### Protocol Fields
| Field | Type | Description |
|:------|:-----|:-----------|
| `cmd` | `String` | Command type (`PICK`, `STOP_CONVEYOR`, `START_CONVEYOR`, `EMERGENCY_STOP`) |
| `cls` | `String` | Operational category (*Glass, Metal, Paper, Plastic, Waste*) |
| `x`, `y`, `z` | `Float` | Target coordinates (mm, relative to arm base) |
| `force` | `Int` | Gripper force (Newtons) |
| `theta` | `Float` | Gripper orientation angle (degrees) |
| `ttg_ms` | `Int` | Time-to-Grab: delay before object reaches pick point (ms) |
| `ts` | `Int` | UNIX timestamp of command generation |

---

## ⚙️ Calibration Parameters

Physical constants for benchtop setup (top of `app.py`):

python
TRIGGER_LINE_RATIO = 0.50        # Vertical trigger position (50% frame height)
TRIGGER_TOLERANCE  = 25          # Acceptance band around trigger line (pixels)
SCALE_FACTOR_MM    = 1.5         # Pixel-to-mm scaling factor
CONVEYOR_SPEED_MM_S= 150.0       # Belt velocity (mm/s)
GRASPING_ZONE_Y_MM = 600.0       # Distance from trigger line to gripper
BIN_CAPACITIES     = 10          # Units per category before auto-halt

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Arduino with USB connection (optional—app falls back to offline mode)

### Installation

bash
git clone https://github.com/aratajaddini/smart-waste-robot.git
cd smart-waste-robot

### 1. Virtual Environment

bash
python -m venv backend/.venv

# Windows
backend\.venv\Scripts\activate

# Unix / macOS
source backend/.venv/bin/activate

### 2. Dependencies

bash
pip install -r backend/requirements.txt

### 3. Environment Configuration

bash
# Wind
