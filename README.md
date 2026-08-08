# ♻️ Smart Waste Sorting Robot

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI_Model-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![GUI](https://img.shields.io/badge/Dashboard-Gradio_%2B_PyWebview-orange.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Hardware-Arduino_Serial_Bridge-red.svg)](https://www.arduino.cc/)

**Vision-guided waste classification system combining AI detection, industrial control, and a local documentation assistant.** The project ships two independent applications that share the same waste taxonomy and training lineage but are **not** the same runtime pipeline (see [System Architecture](#-system-architecture) below) — an industrial control dashboard for a physical sorting rig, and a FastAPI backend for API-driven image/video classification.

Developed for the **Innoverse Competition** by Reza Esmaeili Mood (Lead), Abbas (AI Training), Ara (Documentation), and Sina (Support).

---

## 💡 Overview

This project tackles automated waste sorting through computer vision and robotic control. At its core:

- **AI Vision (dashboard):** A custom-trained YOLO11n model detects and tracks waste items live on a conveyor belt across 18 fine-grained classes (TACO dataset), mapped to 5 operational categories for routing.
- **AI Vision (API):** A separate YOLO11n classification head returns the top waste category for a single uploaded image or video — no bounding boxes or tracking, see [Architecture](#-system-architecture).
- **Dual Application:** Industrial dashboard (ECO-SORT AI) for live robot control + FastAPI backend for API-driven integration. They are two independent apps, not two front-ends on one shared pipeline.
- **Hardware Bridge:** JSON-based serial protocol for Arduino-controlled robotic arms with kinematic calculations (dashboard only).
- **Local Documentation Assistant:** A retrieval-based (BM25 + embedding) search over 12 technical knowledge-base documents, with cited, extractive answers. It is **not** a generative/LLM chatbot yet — see [Known Limitations](#-known-limitations--honest-scope).
- **OEE Monitoring:** Real-time industrial KPIs (Availability × Performance × Quality) in the dashboard.

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV · Gradio · PyWebview · SQLite · Serial Communication

---

## 🏗️ System Architecture

```text
                    Camera Feed / Video Input
                              │
              ┌───────────────┴───────────────┐
              │                               │
      ┌───────▼───────┐               ┌───────▼───────┐
      │  ECO-SORT AI   │               │    FastAPI    │
      │  (Dashboard)   │               │    Backend    │
      │                │               │               │
      │ • Gradio UI    │               │ • REST API    │
      │ • PyWebview    │               │ • Auth Keys   │
      │ • OEE Metrics  │               │ • Local KB    │
      │ • Real-time    │               │   search      │
      └───────┬────────┘               └───────┬───────┘
              │                                 │
      ┌───────▼────────┐               ┌────────▼────────┐
      │    Arduino     │               │     Clients     │
      │  Robot Control │               │      (API)      │
      └────────────────┘               └─────────────────┘
```

**These two applications run two different AI pipelines. They are not interchangeable and do not share a runtime.**

**1. Dashboard pipeline (`dashboard/app.py`) — real-time detection + control:**
```text
Frame Input → Letterbox (640×640) → YOLO11n Detection → ByteTrack Tracking
→ 18-to-5 Class Mapping → Trigger-Line Filter → Priority Queue
→ Kinematics Engine (X, Y, Z, θ, TTG) → Serial JSON → Arduino
```

**2. Backend API pipeline (`backend/inference.py`) — single-shot classification:**
```text
Uploaded Image / Video → YOLO11n Classification Head → Top-1 Category + Confidence
→ JSON API Response
```
The API pipeline has **no bounding boxes, no ByteTrack tracking, no trigger line, no priority queue, and no kinematics** — it answers "what is the dominant class in this image?", not "where is each object and when should it be picked?". If your integration needs per-object coordinates, use the dashboard's serial protocol, not `/predict`.

---

## 🤖 AI Model Configuration

### YOLO11n Training (Google Colab)
- **Dataset:** TACO (Trash Annotations in Context) — 18 litter classes
- **Epochs:** 100 (Early stopping: 15) | **Image Size:** 640×640 | **Batch:** 16
- **Augmentation:** Mosaic (1.0), MixUp (0.15), Copy-Paste (0.10), Rotation (±15°), Perspective (0.0005), HSV Jitter, Random Erasing (0.40)

> **Note on evaluation metrics:** trained weights (`.pt`) are intentionally not committed to this repository (binary size). To reproduce or verify mAP@50 / precision / recall, run `dashboard`'s benchmark tool or `model.val()` from `backend/tools/` against your own copy of the weights — see [Known Limitations](#-known-limitations--honest-scope).

### 18-to-5 Class Mapping

| TACO Classes (18) | Operational Category | Priority | Gripper Force |
|:------------------|:---------------------|:---------|:--------------|
| Aluminium foil, Can, Pop tab | **Metal** | 1 (Highest) | 70 N |
| Bottle cap, Bottle, Lid, Other plastic, Plastic bag, Plastic container, Straw | **Plastic** | 2 | 50 N |
| Broken glass | **Glass** | 3 | 20 N |
| Carton, Cup, Paper | **Paper** | 4 | 85 N |
| Cigarette, Other litter, Styrofoam piece, Unlabeled litter | **Waste** | 5 (Lowest) | 60 N |

**Rationale:** the dashboard's priority queue resolves conflicts when multiple items cross the trigger line simultaneously (*Metal > Plastic > Glass > Paper > Waste*). This priority order applies to the **dashboard only** — the backend API returns a single classification with no queueing logic.

---

## 🌱 Documentation Index

All documents live under [`backend/docs/kb/`](backend/docs/kb/) and are what the local documentation assistant (`/chat`) searches over.

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

### 💬 Documentation Assistant & Knowledge Base

| # | 📄 Document | Description |
|---|---|---|
| 06 | [**Retrieval & Chat**](backend/docs/kb/06-rag-and-chat.md) | Architecture of the search/answer pipeline |
| 07 | [**FAQ**](backend/docs/kb/07-faq.md) | Common questions & troubleshooting tips |

### 🛡️ Operations & Security

| # | 📄 Document | Description |
|---|---|---|
| 08 | [**Failure Modes**](backend/docs/kb/08-failure-modes.md) | Known issues & recovery strategies |
| 09 | [**Security & Keys**](backend/docs/kb/09-security-and-keys.md) | Auth workflow, API-key handling |

---

## 🔌 Hardware Communication Protocol (Dashboard Only)

Commands transmit over USB Serial (**9600 Baud**) as newline-terminated JSON payloads. This protocol is emitted by `dashboard/app.py` / `dashboard/Arduino.py` — the FastAPI backend never talks to serial hardware.

### Sample `PICK` Command
```json
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
```

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

## ⚙️ Calibration Parameters (Dashboard Only)

Physical constants for the benchtop setup, defined in [`dashboard/config.yaml`](dashboard/config.yaml) and read by `dashboard/app.py`:

```yaml
vision:
  trigger_line_ratio: 0.50     # Vertical trigger position (50% of frame height)
  trigger_tolerance_px: 25     # Acceptance band around the trigger line (pixels)
  scale_factor_mm: 1.5         # Pixel-to-mm scaling factor

conveyor:
  speed_mm_s: 150.0            # Belt velocity (mm/s)
  grasping_zone_y_mm: 600.0    # Distance from trigger line to gripper
  direction: "DOWNWARD"

bin_capacities: 10             # Units per category before auto-halt
```

These are **benchtop test values**, not safety-rated production constants — recalibrate every value in `dashboard/config.yaml` against your physical rig before deployment.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Arduino with USB connection (optional — the dashboard falls back to offline mode automatically)

### Clone

```bash
git clone https://github.com/aratajaddini/smart-waste-robot.git
cd smart-waste-robot
```

---

### Option A — FastAPI Backend (API + local documentation assistant)

**1. Virtual environment**
```bash
python -m venv backend/.venv

# Windows
backend\.venv\Scripts\activate

# Unix / macOS
source backend/.venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r backend/requirements.txt
# for tests / linting instead:
# pip install -r backend/requirements-dev.txt
```

**3. Environment configuration**

Create a `.env` file in the project root (see [`.env.example`](.env.example)):
```bash
# Required — the server refuses to start without these.
API_KEY=choose-a-long-random-admin-key
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5500

# Optional — override only if your paths differ from the defaults.
# MODEL_PATH=backend/weights/best.pt
# REQUIRE_MODEL=1          # set to 0 to boot without weights (chat/history still work)
```
`API_KEY` is your first admin key — use it to mint additional shift-scoped keys later with `backend/tools/mint_key.py`. `ALLOWED_ORIGINS` must never be `*`; the app fails fast at startup if it is (see `backend/config.py::assert_configured`).

**4. Model weights**

Trained weights are not committed to this repo. Place your own classification checkpoint at `backend/weights/best.pt`, or set `MODEL_PATH` to point elsewhere. Without weights, set `REQUIRE_MODEL=0` to boot the server anyway — `/predict` will 500 until weights are supplied, but `/chat`, `/history`, and `/health` still work.

**5. Build the local documentation index** (required for `/chat`)
```bash
python -m backend.tools.build_kb
```

**6. Run**
```bash
uvicorn backend.main:app --reload
```
The API is served at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`.

**7. Mint a scoped API key** (optional — the admin `API_KEY` from `.env` already works)
```bash
python -m backend.tools.mint_key issue --label "demo shift" --hours 8
```

**8. Run tests**
```bash
pytest backend/tests/ -q
```

---

### Option B — ECO-SORT AI Dashboard (live detection + robot control)

The dashboard is a separate application with its own dependencies (heavier: torch, ultralytics, OpenCV GUI build, Gradio, PyWebview).

**1. Virtual environment**
```bash
python -m venv dashboard/.venv

# Windows
dashboard\.venv\Scripts\activate

# Unix / macOS
source dashboard/.venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r dashboard/requirements.txt
```

**3. Model weights**

Place your trained detection weights where `dashboard/app.py` expects them (see `MODEL_PATH` near the top of the file), and confirm `dashboard/data.yaml` points at your dataset if you plan to run the in-app benchmark tool.

**4. Calibrate**

Edit `dashboard/config.yaml` to match your physical rig (trigger line, belt speed, gripper forces, bin capacities) — the shipped values are benchtop test defaults, not production calibration.

**5. Run**
```bash
python dashboard/app.py
```
This launches a native desktop window (via PyWebview) hosting the Gradio UI. If no Arduino is detected on startup, the dashboard continues in offline mode — reconnect later from the UI's "Reconnect Hardware" button.

---

## 🛡️ Known Limitations / Honest Scope

To avoid overstating what's implemented, here's what is and isn't real in this repository today:

| Claim | Status |
|---|---|
| Dashboard: real-time YOLO11n detection + ByteTrack tracking + kinematics | ✅ Implemented (`dashboard/app.py`) |
| Backend API: same detection/tracking/kinematics pipeline as the dashboard | ❌ **Not the case** — the API performs whole-image/video classification only, no bounding boxes or tracking (`backend/inference.py`) |
| Local documentation search with citations | ✅ Implemented — hybrid BM25 + embedding retrieval with reciprocal rank fusion (`backend/chat/retriever.py`) |
| "RAG chatbot" that generates conversational answers | ❌ **Not yet** — `/chat` returns extractive, stitched-together excerpts from the knowledge base, not LLM-generated prose. Generative LLM integration is planned but not wired in (see `backend/requirements-dev.txt` comment on the future `/chat` LLM client) |
| Physical robotic arm hardware/firmware | ❌ Out of scope for this repository — the dashboard emits a ready-to-consume serial protocol; building the arm side is left to the integrator |
| Bin-evacuation mechanism | ❌ Not implemented — the dashboard's "Empty Bin" button is a UI-side simulation for testing the halt/resume cycle |
| Trained model weights / verifiable accuracy metrics | ❌ Not included in this repository (binary size) — bring your own `.pt` file and run the benchmark tooling to reproduce metrics |
| `backend/analytics/` module | ⚠️ Present in the codebase but explicitly unused in the delivery path (see comment in `backend/config.py`) — kept for future analytics work |

If you're evaluating this project: the dashboard and backend are two separate, honestly-scoped applications that share a waste taxonomy and training lineage, not one shared real-time AI pipeline.

---

## 🧪 Development

```bash
pip install -r backend/requirements-dev.txt
ruff check backend/
mypy backend/
bandit -r backend/
pytest backend/tests/ -q
```

---
