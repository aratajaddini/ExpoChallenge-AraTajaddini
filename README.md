# ♻️ Smart Waste Sorting Robot (TRACE-SORT-AI / TraceSort)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI_Model-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![GUI](https://img.shields.io/badge/Dashboard-Gradio_%2B_PyWebview-orange.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Hardware-Arduino_Serial_Bridge-red.svg)](https://www.arduino.cc/)
[![License](https://img.shields.io/badge/License-See_LICENSE-lightgrey.svg)](LICENSE)

**An end-to-end, vision-guided waste classification platform** combining a real-time AI sorting rig, a public-facing web app, a REST API with a local documentation assistant, and an industrial robotic control dashboard.

Developed for the **Innoverse Competition** by Reza Esmaeili Mood (Lead), Abbas (AI Training), Ara (Documentation), and Sina (Support).

---

## 📚 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Repository Layout](#-repository-layout)
4. [AI / Computer Vision](#-ai--computer-vision)
5. [Dashboard — Trace-Sort-AI](#-dashboard--trace-sort-ai)
6. [Backend API](#-backend-api)
7. [Frontend — TraceSort Website](#-frontend--tracesort-website)
8. [Hardware Communication Protocol](#-hardware-communication-protocol)
9. [Calibration Parameters](#-calibration-parameters)
10. [Getting Started](#-getting-started)
11. [Testing & CI](#-testing--ci)
12. [Documentation Index](#-documentation-index)
13. [Roadmap & Future Vision](#-roadmap--future-vision)

---

## 💡 Overview

This project tackles automated waste sorting through computer vision and robotic control, delivered as **three coordinated applications** that share a common waste taxonomy and training lineage:

| Application | Purpose | Stack |
|---|---|---|
| **Dashboard (Trace-SORT AI)** | Real-time detection, tracking, kinematics, and robotic-arm control for a physical sorting rig | Python, Ultralytics YOLO, OpenCV, Gradio, PyWebview, PySerial |
| **Backend API** | Stateless REST service for image/video classification, plus a local documentation assistant | FastAPI, Ultralytics YOLO, SQLite |
| **Frontend (TraceSort)** | Public-facing marketing site and live demo client that talks to the Backend API | HTML5, CSS3, vanilla JavaScript |

**Core capabilities:**

- **AI Vision (dashboard):** A custom-trained YOLO11n model detects and tracks waste items live on a conveyor belt across 18 fine-grained classes (TACO dataset), mapped to 5 operational categories for routing.
- **AI Vision (API):** A separate YOLO11n classification head returns the top waste category for a single uploaded image or video.
- **Hardware Bridge:** A JSON-based serial protocol drives an Arduino-controlled robotic arm, complete with 2D kinematics (position, angle, time-to-grab) computed per detected object.
- **Local Documentation Assistant:** A retrieval-based (hybrid BM25 + sentence-embedding) search over the project's technical knowledge base, returning cited, extractive answers via the `/chat` endpoint.
- **OEE Monitoring:** Real-time industrial KPIs (Availability × Performance × Quality) surfaced live on the dashboard.
- **Web Client:** A polished landing page and interactive demo (TraceSort) that authenticates against the API and drives `/predict` and `/chat` directly from the browser.

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV · Gradio · PyWebview · SQLite · Sentence-Transformers · Vanilla JS/HTML/CSS · Serial Communication

---

## 🏗️ System Architecture

```text
                     Camera Feed / Video Input
                               │
               ┌───────────────┴───────────────┐
               │                               │
       ┌───────▼───────┐               ┌───────▼───────┐
       │  TRACE-SORT AI   │               │    FastAPI    │
       │  (Dashboard)   │               │    Backend    │
       │                │               │               │
       │ • Gradio UI    │               │ • REST API    │
       │ • PyWebview    │               │ • Auth Keys   │
       │ • OEE Metrics  │               │ • Local KB    │
       │ • Real-time    │               │   search      │
       └───────┬────────┘               └───────┬───────┘
               │                                 │
       ┌───────▼────────┐               ┌────────▼────────┐
       │    Arduino     │               │   TraceSort     │
       │  Robot Control │               │  Web Frontend   │
       └────────────────┘               └─────────────────┘
```

The dashboard and the backend run **two independent AI pipelines** — they share a waste taxonomy and training lineage, not a runtime.

**1. Dashboard pipeline (`dashboard/app.py`) — real-time detection + control:**
```text
Frame Input → Letterbox (640×640) → YOLO11n Detection → ByteTrack Tracking
→ 18-to-5 Class Mapping → Trigger-Line Filter → Priority Queue
→ Contour-Based Orientation Estimation → Kinematics Engine (X, Y, Z, θ, TTG)
→ Serial JSON → Arduino
```

**2. Backend API pipeline (`backend/inference.py`) — single-shot classification:**
```text
Uploaded Image / Video (from TraceSort or any client) → YOLO11n Classification Head
→ Top-1 Category + Confidence → JSON API Response
```

The API pipeline answers *"what is the dominant class in this image?"*, not *"where is each object and when should it be picked?"*. Integrations that need per-object coordinates should consume the dashboard's serial protocol, not `/predict`.

---

## 🗂️ Repository Layout

```text
smart-waste-robot/
├── backend/           # FastAPI service: prediction API, auth, RAG chat, admin
│   ├── chat/          # Hybrid retriever (BM25 + embeddings) + small-talk handling
│   ├── routers/       # /predict, /auth, /history, /feedback, /admin/keys
│   ├── models/        # SQLite schema/access layer
│   ├── schemas/        # Pydantic request/response models
│   ├── tools/          # CLI utilities: build_kb, mint_key, calibrate_kb, etc.
│   ├── docs/kb/         # 12 knowledge-base documents indexed by /chat
│   ├── tests/           # Pytest suite (auth, CORS, predict, chat, small talk)
│   ├── config.py        # Centralized, env-driven configuration with fail-fast checks
│   ├── security.py      # API-key auth dependencies
│   └── inference.py     # YOLO classification logic (model-agnostic of FastAPI)
├── dashboard/          # TRACE-SORT AI: real-time detection + robot control app
│   ├── app.py           # Gradio + PyWebview application (detection loop, OEE, UI)
│   ├── Arduino.py        # Serial connection/reconnection + message transport
│   ├── train.py           # Training entry point for the detection model
│   └── config.yaml        # Physical rig calibration (belt, gripper, bins, ports)
├── frontend/            # TraceSort marketing site + live demo client
│   ├── index.html         # Landing page, live demo, chat widget markup
│   ├── script.js           # API-key handling, /predict & /chat integration
│   └── style.css            # Site styling
└── .github/workflows/ci.yml # Test + lint pipeline (pytest, ruff, mypy, bandit)
```

---

## 🤖 AI / Computer Vision

### Model & Training
- **Base model:** YOLO11n, trained on Google Colab.
- **Dataset:** TACO (Trash Annotations in Context) — 18 fine-grained litter classes.
- **Training config:** 100 epochs (early stopping at 15), 640×640 image size, batch size 16.
- **Augmentation:** Mosaic (1.0), MixUp (0.15), Copy-Paste (0.10), Rotation (±15°), Perspective (0.0005), HSV jitter, Random Erasing (0.40).
- **Two model heads, two jobs:** the dashboard runs a *detection* model (bounding boxes + tracking); the backend API runs a *classification* model (single top-class label). They are trained from the same lineage but serve different purposes and are **not interchangeable**.

> Trained weights (`.pt`) are intentionally not committed to this repository (binary size). To reproduce or verify mAP@50 / precision / recall, run the dashboard's built-in benchmark tool or `model.val()` from `backend/tools/` against your own copy of the weights.

### 18-to-5 Operational Class Mapping

| TACO Classes (18) | Operational Category | Priority | Gripper Force |
|:------------------|:---------------------|:---------|:--------------|
| Aluminium foil, Can, Pop tab | **Metal** | 1 (Highest) | 70 N |
| Bottle cap, Bottle, Lid, Other plastic, Plastic bag, Plastic container, Straw | **Plastic** | 2 | 50 N |
| Broken glass | **Glass** | 3 | 20 N |
| Carton, Cup, Paper | **Paper** | 4 | 85 N |
| Cigarette, Other litter, Styrofoam piece, Unlabeled litter | **Waste** | 5 (Lowest) | 60 N |

**Rationale:** the dashboard's priority queue resolves conflicts when multiple items cross the trigger line simultaneously (*Metal > Plastic > Glass > Paper > Waste*). This priority order applies to the dashboard only — the backend API returns a single classification with no queueing logic.

### Vision Pipeline Details (Dashboard)
- **Tracking:** ByteTrack assigns persistent IDs to detected objects across frames, with a garbage-collection routine (`cleanup_tracking_memory`) that expires stale track IDs after 30 seconds.
- **Trigger-line filtering:** objects are only actioned once their tracked center crosses a configurable horizontal line (`trigger_line_ratio`), within a pixel tolerance band.
- **Orientation estimation:** `extract_object_orientation()` isolates the object's contour (Otsu threshold, falling back to Canny edges when needed) and fits a minimum-area rectangle to estimate rotation angle — with explicit handling for OpenCV's legacy vs. modern `minAreaRect` angle conventions.
- **Kinematics engine:** converts pixel-space object position into real-world millimeter coordinates (X, Y), computes remaining distance to the gripper, time-to-grab (based on belt speed), and packages the result with class, force, and angle into a single JSON payload for the Arduino.
- **Local Documentation Assistant (`/chat`):** hybrid BM25 + sentence-embedding retrieval with Reciprocal Rank Fusion over 12 knowledge-base documents, gated by a minimum cosine-similarity threshold before answering — it returns cited, extractive excerpts rather than free-form generated text.

---

## 🖥️ Dashboard — TRACE-SORT AI

`dashboard/app.py` is the operator-facing application: a native desktop window (via PyWebview) hosting a Gradio UI, built for a live physical sorting rig.

**Key features:**
- Live camera or uploaded-video inference with on-frame overlays.
- Real-time OEE metrics: **Availability × Performance × Quality**, computed from planned production time, ideal cycle time, and accumulated downtime.
- Per-category sorted counts, share percentages, and live bin-fill status against configurable capacities.
- Automatic conveyor halt/resume when a bin reaches capacity, with full audit logging.
- Emergency-stop / release controls that immediately lock out robot commands.
- Automatic Arduino discovery on startup, with graceful fallback to offline (simulation) mode and a UI "Reconnect Hardware" action.
- Built-in benchmark tool to validate model accuracy (mAP/precision/recall) against a held-out validation set.
- Revenue estimation per category using a configurable dollar-value table, feeding the dashboard's financial KPIs.

**Configuration:** all physical constants (conveyor speed, trigger line, gripper forces, bin capacities, serial port/baud rate, planned production time, ideal cycle time) live in `dashboard/config.yaml` and are loaded at startup — no hard-coded rig parameters in the application code.

---

## 🔌 Backend API

FastAPI service exposing prediction, authentication, history, feedback, admin, and chat endpoints.

### Endpoints

| Route | Method | Description |
|---|---|---|
| `/predict` | `POST` | Classify an uploaded image or video (`mode=auto/image/video`) |
| `/predict/classes` | `GET` | List the supported waste categories from the loaded model |
| `/auth/verify` | `GET` | Verify an `X-API-Key` and return its identity |
| `/history` | `GET` | Retrieve past prediction records |
| `/feedback` | `POST` | Submit correction/feedback on a prediction |
| `/admin/keys` | `*` | Mint, list, and revoke shift-scoped API keys (admin-only) |
| `/chat` | `POST` | Ask a question against the local documentation knowledge base |
| `/health` | `GET` | Liveness probe |

### Security model
- Every protected route requires an `X-API-Key` header.
- The **admin key** (from `.env`) is compared with `secrets.compare_digest` to avoid timing attacks.
- **Shift-scoped keys** are minted with an expiry, stored as **SHA-256 hashes only** (never in plaintext), and can be revoked individually — full lifecycle managed via `backend/tools/mint_key.py` and the `/admin/keys` routes.
- CORS origins are explicit and env-driven; the app **refuses to start** if `ALLOWED_ORIGINS` is unset or set to `"*"`, since the API relies on header-based auth rather than cookies.
- `backend/config.py::assert_configured()` performs a full fail-fast validation pass at startup (API key presence, model path, CORS, KB parameters, upload/video limits) so misconfiguration is caught immediately rather than surfacing as a runtime error later.

### Upload handling
- Uploads are streamed to disk in 1 MB chunks with a hard byte-size ceiling (`MAX_UPLOAD_BYTES`), rejecting oversized files mid-stream rather than after a full buffer.
- Destination filenames are randomly generated (`uuid4`), not derived from client input.
- Temporary files are always cleaned up, including on error paths.

---

## 🌐 Frontend — TraceSort Website

`frontend/` is a self-contained static site (no build step, no framework) that serves as both the project's public landing page and a live client for the Backend API.

- **Landing experience:** hero section, "how it works" walkthrough, and project storytelling, served directly by FastAPI's static file mount when the backend is running.
- **Live demo:** an in-browser panel that uploads an image/video straight to `/predict` and renders the returned classification.
- **Chat widget:** a UI on top of `/chat` for querying the project's documentation assistant with cited answers.
- **API-key handling:** keys are kept in `sessionStorage` (tab-scoped, cleared on tab close) rather than `localStorage`, with an explicit in-code warning that this pattern is appropriate for a demo, not for a production multi-tenant deployment.
- **Configurable API base:** `window.__API_BASE__` lets the same static site point at a different backend host without a rebuild.

---

## 🔌 Hardware Communication Protocol

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

## ⚙️ Calibration Parameters

Physical constants for the benchtop setup, defined in [`dashboard/config.yaml`](dashboard/config.yaml) and read by `dashboard/app.py`:

```yaml
hardware:
  default_port: "COM3"
  baudrate: 9600

vision:
  trigger_line_ratio: 0.50     # Vertical trigger position (50% of frame height)
  trigger_tolerance_px: 25     # Acceptance band around the trigger line (pixels)
  scale_factor_mm: 1.5         # Pixel-to-mm scaling factor

conveyor:
  speed_mm_s: 150.0            # Belt velocity (mm/s)
  grasping_zone_y_mm: 600.0    # Distance from trigger line to gripper
  direction: "DOWNWARD"

bin_capacities: 10             # Units per category before auto-halt

grip_forces:
  Glass: 20
  Paper: 85
  Plastic: 50
  Metal: 70
  Waste: 60

PLANNED_PRODUCTION_TIME: 3600.0  # seconds, for OEE Availability
IDEAL_CYCLE_TIME: 3.0            # seconds, for OEE Performance
zm: -150.0                       # fixed Z height (mm)
```

These are **benchtop test values**, not safety-rated production constants — recalibrate every value in `dashboard/config.yaml` against your physical rig before deployment.

---

## 🚀 Getting Started

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

### Option A — FastAPI Backend (API + Frontend + Documentation Assistant)

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

Create a `.env` file in the project root:
```bash
# Required — the server refuses to start without these.
API_KEY=choose-a-long-random-admin-key
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5500

# Optional — override only if your paths differ from the defaults.
# MODEL_PATH=backend/weights/best.pt
# REQUIRE_MODEL=1          # set to 0 to boot without weights (chat/history still work)
```
`API_KEY` is your first admin key — use it to mint additional shift-scoped keys later with `backend/tools/mint_key.py`. `ALLOWED_ORIGINS` must never be `*`; the app fails fast at startup if it is.

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
The API is served at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`, and the TraceSort frontend is served at `http://127.0.0.1:8000/` (static mount).

**7. Mint a scoped API key** (optional — the admin `API_KEY` from `.env` already works)
```bash
python -m backend.tools.mint_key issue --label "demo shift" --hours 8
```

**8. Run tests**
```bash
pytest backend/tests/ -q
```

---

### Option B — TRACE-SORT AI Dashboard (live detection + robot control)

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

Edit `dashboard/config.yaml` to match your physical rig (serial port, trigger line, belt speed, gripper forces, bin capacities) — the shipped values are benchtop test defaults, not production calibration.

**5. Run**
```bash
python dashboard/app.py
```
This launches a native desktop window (via PyWebview) hosting the Gradio UI. If no Arduino is detected on startup, the dashboard continues in offline mode — reconnect later from the UI's "Reconnect Hardware" button.

---

## 🧪 Testing & CI

```bash
pip install -r backend/requirements-dev.txt
ruff check backend/
mypy backend/
bandit -r backend/
pytest backend/tests/ -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR against `main`:
- **Test job:** installs dev dependencies and runs `pytest` with `REQUIRE_MODEL=0` so the suite doesn't depend on committed weights.
- **Lint job:** `ruff` (fatal-error rules block the build; full rule set runs advisory), `mypy`, and `bandit` for static analysis and security scanning.

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

## 🗺️ Roadmap & Future Vision

| # | Initiative | Description |
|---|---|---|
| 1 | **Physical Robotic Arm Integration** | Integrate a physical robotic arm that consumes the existing serial payload to perform real picks. |
| 2 | **Field Calibration** | Field-calibrate conveyor speed, bin capacities, and cycle-time constants against the real hardware. |
| 3 | **Automated Bin Evacuation** | Add a second actuator for automated bin evacuation using the same output protocol. |
| 4 | **Next-Gen Model Migration & Oriented Bounding Boxes (YOLO26-OBB)** | Upgrade the vision engine to YOLO26-OBB to natively predict object rotation angles for precise gripper alignment, replacing OpenCV contour-based angle estimation while reducing inference latency and improving mAP under heavy occlusion. |
| 5 | **Human-in-the-Loop MLOps & Active Learning Pipeline** | Automatic caching of low-confidence detections (Conf < 0.55) during shifts; an Operator Review Panel inside the dashboard for one-click label corrections; automated generation of YOLO (`.txt`) annotations for continuous model fine-tuning (self-improving system). |
| 6 | **Physical Hardware & Sensor Expansion** | Integration of ultrasonic bin sensors for continuous physical depth/volume measurement; multi-arm routing with dynamic target allocation across multiple robotic arms on high-speed lines. |
| 7 | **Long-Term Business Intelligence (BI) Analytics** | SQLite/PostgreSQL historical database integration; monthly executive dashboards tracking overall recovery tonnage, OEE trends, and financial yield ($). |
| 8 | **Edge Hardware Deployment** | Port the vision model from the host PC to embedded edge AI platforms (e.g., NVIDIA Jetson Orin Nano or Raspberry Pi 5 + Hailo-8 AI Accelerator) to drastically reduce power consumption and deployment costs. |
| 9 | **Model Optimization (TensorRT / ONNX)** | Convert the trained YOLO26 model to TensorRT and ONNX formats to achieve ultra-low latency and maximum FPS on edge hardware. |
| 10 | **Cloud Analytics & Fleet Management** | Build a centralized cloud dashboard to monitor multiple waste-sorting lines across facilities in real time, aggregating data for executive reporting and operational insights. |

---



## 📬 Contact & Team

For inquiries, collaboration, or feedback regarding **Trace-Sort AI**:

* **Project Lead:** Reza Esmaeili Mood — [esmaeilireza1994@gmail.com](mailto:esmaeilireza1994@gmail.com)
* **GitHub:** [https://github.com/esmaeilireza](https://github.com/esmaeilireza)
  
---
## 📄 License

See [LICENSE](LICENSE) for details.
