<img width="40" height="40" alt="favicon-180" src="https://github.com/user-attachments/assets/475eb7fb-2049-4bf3-9726-f51679b45393" /> Trace Sort AI — Smart Waste Sorting Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI_Model-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![GUI](https://img.shields.io/badge/Dashboard-Gradio_%2B_PyWebview-orange.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Hardware-Arduino_Serial_Bridge-red.svg)](https://www.arduino.cc/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**An end-to-end, vision-guided waste classification platform** combining a public-facing web app with a classification API, and a standalone real-time industrial dashboard with robotic-arm control.

Developed for the **Innoverse Competition** by Reza Esmaeili Mood (Lead), Abbas Lotfi (AI Training), Ara Tajaddini (Documentation), and Sina Niknejad (Support).

**Repository:** [github.com/aratajaddini/ExpoChallenge-RezaEsmaeiliMood](https://github.com/aratajaddini/ExpoChallenge-AraTajaddini)

---

## 📚 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Repository Layout](#-repository-layout)
4. [Clone the Repository](#-clone-the-repository)
5. [Part 1 — Web Platform (Backend API & Frontend)](#-part-1--web-platform-backend-api--frontend)
   - [AI Classification Model](#ai-classification-model)
   - [Backend API](#backend-api)
   - [Frontend — TraceSort Website](#frontend--tracesort-website)
   - [Running the Web Platform](#running-the-web-platform)
6. [Part 2 — TRACE-SORT AI Dashboard (Real-Time Detection & Robotics)](#-part-2--trace-sort-ai-dashboard-real-time-detection--robotics)
   - [AI Detection Model & Vision Pipeline](#ai-detection-model--vision-pipeline)
   - [Dashboard Features](#dashboard-features)
   - [Model Evaluation & Results](#model-evaluation--results)
   - [Hardware Communication Protocol](#hardware-communication-protocol)
   - [Calibration Parameters](#calibration-parameters)
   - [Full Documentation (PDF)](#full-documentation-pdf)
   - [Downloads — Installers (Windows & macOS)](#downloads--installers-windows--macos)
   - [Test the Dashboard (Sample Video)](#test-the-dashboard-sample-video)
   - [Running the Dashboard from Source](#running-the-dashboard-from-source)
7. [Operational Class Mapping (Dashboard)](#-operational-class-mapping-dashboard)
8. [Testing & CI](#-testing--ci)
9. [Documentation Index (Chat Knowledge Base)](#-documentation-index-chat-knowledge-base)
10. [Roadmap & Future Vision](#-roadmap--future-vision)
11. [Contact & Team](#-contact--team)
12. [License](#-license)

---

## 💡 Overview

**Trace Sort AI** tackles automated waste sorting through computer vision and robotic control. The project ships as **two parts of a single product experience**:

| Part | Application | Purpose | Stack |
|---|---|---|---|
| **1** | **Web Platform** (Backend API + TraceSort Frontend) | Public-facing site where anyone can try the live waste-classification demo, plus a local documentation assistant | FastAPI, Ultralytics YOLO, SQLite, vanilla JS/HTML/CSS |
| **2** | **TRACE-SORT AI Dashboard** | The project's core: real-time detection, tracking, kinematics, and robotic-arm control for a physical sorting rig | Python, Ultralytics YOLO, OpenCV, Gradio, PyWebview, PySerial |

**The intended user journey:** a visitor first lands on the **TraceSort website (Part 1)**, uploads a photo or short clip, and instantly sees the classification model in action. From there, anyone who wants to see the project's main engine — the physical, real-time sorting system — can move on to the **TRACE-SORT AI Dashboard (Part 2)**.

Technically, however, **Part 1 and Part 2 run as two separate applications with two independently trained AI models** — the dashboard never calls the backend API, and the backend never talks to serial hardware. They are documented separately below (**Part 1** and **Part 2**) so each can be read, run, and deployed on its own.

**Core capabilities at a glance:**

- **AI Vision (Web Platform):** A YOLO8n classification model, trained on the **[TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet)**, returns the top waste category for a single uploaded image or video via the `/predict` endpoint.
- **AI Vision (Dashboard):** A custom YOLO11n detection model, trained on the **[TACO dataset](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format)**, detects and tracks waste items live on a conveyor belt across 18 fine-grained classes, mapped to 5 operational categories for routing.
- **Hardware Bridge (Dashboard only):** A JSON-based serial protocol drives an Arduino-controlled robotic arm, complete with 3D kinematics (position, angle, time-to-grab) computed per detected object.
- **Local Documentation Assistant (Web Platform only):** A retrieval-based (hybrid BM25 + sentence-embedding) search over the project's technical knowledge base, returning cited, extractive answers via the `/chat` endpoint.
- **OEE Monitoring (Dashboard only):** Real-time industrial KPIs (Availability × Performance × Quality) surfaced live on the dashboard.
- **Web Client:** A polished landing page and interactive demo (TraceSort) that authenticates against the API and drives `/predict` and `/chat` directly from the browser.

---

## 🏗️ System Architecture

```text
                     Camera Feed / Video Input
                               │
               ┌───────────────┴───────────────┐
               │                               │
       ┌───────▼────────┐              ┌───────▼───────┐
       │  TRACE-SORT AI  │              │    FastAPI    │
       │   (Dashboard)   │              │    Backend    │
       │   — Part 2 —    │              │   — Part 1 —  │
       │                 │              │               │
       │ • Gradio UI     │              │ • REST API    │
       │ • PyWebview     │              │ • Auth Keys   │
       │ • OEE Metrics   │              │ • Local KB    │
       │ • Real-time     │              │   search      │
       └───────┬─────────┘              └───────┬───────┘
               │                                 │
       ┌───────▼────────┐              ┌────────▼────────┐
       │    Arduino     │              │   TraceSort     │
       │  Robot Control │              │  Web Frontend   │
       └────────────────┘              └─────────────────┘
```

Part 1 and Part 2 run **two independent AI pipelines, trained on two different datasets** — they are connected only by belonging to the same overall product, not by a shared runtime or a shared model.

**Part 2 — Dashboard pipeline (`dashboard/src/app.py`) — real-time detection + control:**
```text
Frame Input → Letterbox (640×640) → YOLO11n Detection (trained on TACO) → ByteTrack Tracking
→ 18-to-5 Class Mapping → Trigger-Line Filter → Priority Queue
→ Contour-Based Orientation Estimation → Kinematics Engine (X, Y, Z, θ, TTG)
→ Serial JSON → Arduino
```

**Part 1 — Backend API pipeline (`backend/inference.py`) — single-shot classification:**
```text
Uploaded Image / Video (from TraceSort or any client) → YOLO8n Classification Model (trained on TrashNet)
→ Top-1 Category + Confidence → JSON API Response
```

The API pipeline answers *"what is the dominant class in this image?"*, not *"where is each object and when should it be picked?"*. Integrations that need per-object coordinates should consume the dashboard's serial protocol (Part 2), not `/predict`.

---

## 🗂️ Repository Layout

```text
ExpoChallenge-RezaEsmaeiliMood/
├── backend/                # PART 1 — FastAPI service: prediction API, auth, RAG chat, admin
│   ├── chat/                 # Hybrid retriever (BM25 + embeddings) + small-talk handling
│   ├── routers/               # /predict, /auth, /history, /feedback, /admin/keys
│   ├── models/                 # SQLite schema/access layer
│   ├── schemas/                 # Pydantic request/response models
│   ├── tools/                    # CLI utilities: build_kb, mint_key, calibrate_kb, etc.
│   ├── docs/kb/                    # 12 knowledge-base documents indexed by /chat
│   ├── tests/                       # Pytest suite (auth, CORS, predict, chat, small talk)
│   ├── weights/                      # 🧠 Classification model weights (.pt) — powers /predict
│   ├── config.py                      # Centralized, env-driven configuration with fail-fast checks
│   ├── security.py                     # API-key auth dependencies
│   └── inference.py                     # YOLO classification logic (model-agnostic of FastAPI)
├── frontend/                # PART 1 — TraceSort marketing site + live demo client
│   ├── index.html             # Landing page, live demo, chat widget markup
│   ├── script.js                # API-key handling, /predict & /chat integration
│   └── style.css                  # Site styling
├── dashboard/                # PART 2 — TRACE-SORT AI: real-time detection + robot control app
│   ├── src/                    # Application source code
│   │   ├── app.py                 # Gradio + PyWebview application (detection loop, OEE, UI)
│   │   ├── Arduino.py              # Serial connection/reconnection + message transport
│   │   ├── train.py                 # Training entry point for the detection model (TACO dataset)
│   │   ├── config.yaml               # Physical rig calibration (belt, gripper, bins, ports)
│   │   └── assets/                     # Static assets bundled with the app
│   │       └── industrial-bg.mp4         # Background video used by the dashboard UI
│   ├── model-weights/            # 🧠 Detection model weights (.pt) for the dashboard's YOLO11n model
│   ├── docs/                      # 📄 Full PDF technical documentation for the dashboard
│   └── results/                    # 📊 Confusion matrices, PR curves & sample detection outputs
├── .github/workflows/ci.yml   # Test + lint pipeline (pytest, ruff, mypy, bandit)
├── pyproject.toml         # Shared project metadata + pytest/ruff/mypy configuration
├── LICENSE                # MIT License
└── README.md
```

---

## 📥 Clone the Repository

Both Part 1 and Part 2 live in the same repository — clone it once, then follow whichever part(s) you need below.

```bash
git clone https://github.com/aratajaddini/ExpoChallenge-RezaEsmaeiliMood.git
cd ExpoChallenge-RezaEsmaeiliMood
```

---

# 🌐 Part 1 — Web Platform (Backend API & Frontend)

This part is the public-facing side of Trace Sort AI: a stateless classification API plus the TraceSort website that consumes it. It has **no dependency on Arduino, serial ports, or physical hardware** — it can be deployed to any standard web host.

## AI Classification Model

- **Base model:** YOLO8n, adapted as a classification head.
- **Dataset:** **[TrashNet](https://huggingface.co/datasets/garythung/trashnet)** — a labeled image dataset covering six household waste categories (cardboard, glass, metal, paper, plastic, trash).
- **Job:** a *classification* model — it returns a single top-class label and confidence score for an uploaded image or video, with **no bounding boxes, tracking, or queueing logic**.
- **Weights:** committed to the repository under [`backend/weights/`](backend/weights/) — the API and the live demo work out of the box, no external download required.

> This model is trained independently from the dashboard's detection model — different dataset (TrashNet vs. TACO), different job (classification vs. detection), and **not interchangeable** with it. The two are related only in that both serve the same broader goal: identifying what kind of waste an item is.

## Backend API

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

### Local Documentation Assistant
The `/chat` endpoint runs hybrid BM25 + sentence-embedding retrieval with Reciprocal Rank Fusion over 12 knowledge-base documents (see the [Documentation Index](#-documentation-index-chat-knowledge-base)), gated by a minimum cosine-similarity threshold before answering. It returns **cited, extractive excerpts**, not free-form generated text.

## Frontend — TraceSort Website

`frontend/` is a self-contained static site (no build step, no framework) that serves as both the project's public landing page and a live client for the Backend API — this is the **first stop for most visitors**.

- **Landing experience:** hero section, "how it works" walkthrough, and project storytelling, served directly by FastAPI's static file mount when the backend is running.
- **Live demo:** an in-browser panel that uploads an image/video straight to `/predict` and renders the returned classification, using the weights already bundled in `backend/weights/`.
- **Chat widget:** a UI on top of `/chat` for querying the project's documentation assistant with cited answers.
- **API-key handling:** keys are kept in `sessionStorage` (tab-scoped, cleared on tab close) rather than `localStorage`, with an explicit in-code warning that this pattern is appropriate for a demo, not for a production multi-tenant deployment.
- **Configurable API base:** `window.__API_BASE__` lets the same static site point at a different backend host without a rebuild.

## Running the Web Platform

### Prerequisites
- Python 3.11+ (< 3.13)
- Git

### 1. Virtual environment
```bash
python -m venv backend/.venv

# Windows
backend\.venv\Scripts\activate

# Unix / macOS
source backend/.venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
# for tests / linting instead:
# pip install -r backend/requirements-dev.txt
```

### 3. Environment configuration

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

### 4. Model weights

The classification weights already ship with this repository under [`backend/weights/`](backend/weights/), so `/predict` works immediately after installing dependencies — no separate download step. If you want to swap in your own checkpoint, set `MODEL_PATH` to point elsewhere.

### 5. Build the local documentation index (required for `/chat`)
```bash
python -m backend.tools.build_kb
```

### 6. Run
```bash
uvicorn backend.main:app --reload
```
The API is served at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`, and the TraceSort frontend is served at `http://127.0.0.1:8000/` (static mount).

### 7. Mint a scoped API key (optional — the admin `API_KEY` from `.env` already works)
```bash
python -m backend.tools.mint_key issue --label "demo shift" --hours 8
```

### 8. Run tests
```bash
pytest backend/tests/ -q
```

> These steps work identically on **Windows, macOS, and Linux** — only the virtual-environment activation command differs (shown above for both).

---

# 🤖 Part 2 — TRACE-SORT AI Dashboard (Real-Time Detection & Robotics)

This is the project's core: an operator-facing, physical-rig application that runs live object detection on a conveyor belt and drives a robotic arm over serial. It is fully independent from Part 1 at runtime — it never calls the backend API and has no web server component — but represents the deeper, "behind the scenes" experience that the TraceSort website invites visitors to explore.

## AI Detection Model & Vision Pipeline

- **Base model:** YOLO11n, trained on Google Colab as a **detection** model (bounding boxes + persistent tracking IDs), not a single-label classifier.
- **Dataset:** **[TACO — Trash Annotations in Context (YOLO format)](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format)** — 18 fine-grained litter classes.
- **Training config:** 100 epochs (early stopping at 15), 640×640 image size, batch size 16.
- **Augmentation:** Mosaic (1.0), MixUp (0.15), Copy-Paste (0.10), Rotation (±15°), Perspective (0.0005), HSV jitter, Random Erasing (0.40).
- **Tracking:** ByteTrack assigns persistent IDs to detected objects across frames, with a garbage-collection routine (`cleanup_tracking_memory`) that expires stale track IDs after 30 seconds.
- **Trigger-line filtering:** objects are only actioned once their tracked center crosses a configurable horizontal line (`trigger_line_ratio`), within a pixel tolerance band.
- **Orientation estimation:** `extract_object_orientation()` isolates the object's contour (Otsu threshold, falling back to Canny edges when needed) and fits a minimum-area rectangle to estimate rotation angle — with explicit handling for OpenCV's legacy vs. modern `minAreaRect` angle conventions.
- **Kinematics engine:** converts pixel-space object position into real-world millimeter coordinates (X, Y), computes remaining distance to the gripper, time-to-grab (based on belt speed), and packages the result with class, force, and angle into a single JSON payload for the Arduino.
- **Weights:** committed to the repository under [`dashboard/model-weights/`](dashboard/model-weights/) — the dashboard runs immediately after installing dependencies, no external download required. The training script (`dashboard/src/train.py`) is included if you want to retrain on your own data.

## Dashboard Features

`dashboard/src/app.py` is the operator-facing application: a native desktop window (via PyWebview) hosting a Gradio UI, built for a live physical sorting rig.

- Live camera or uploaded-video inference with on-frame overlays.
- Real-time OEE metrics: **Availability × Performance × Quality**, computed from planned production time, ideal cycle time, and accumulated downtime.
- Per-category sorted counts, share percentages, and live bin-fill status against configurable capacities.
- Automatic conveyor halt/resume when a bin reaches capacity, with full audit logging.
- Emergency-stop / release controls that immediately lock out robot commands.
- Automatic Arduino discovery on startup, with graceful fallback to offline (simulation) mode and a UI "Reconnect Hardware" action.
- Built-in benchmark tool to validate model accuracy (mAP/precision/recall) against a held-out validation set.
- Revenue estimation per category using a configurable dollar-value table, feeding the dashboard's financial KPIs.

**Configuration:** all physical constants (conveyor speed, trigger line, gripper forces, bin capacities, serial port/baud rate, planned production time, ideal cycle time) live in `dashboard/src/config.yaml` and are loaded at startup — no hard-coded rig parameters in the application code.

## Model Evaluation & Results

Confusion matrices, precision-recall curves, and sample detection/tracking outputs from the trained TACO model are available under [`dashboard/results/`](dashboard/results/). These reflect the same metrics the dashboard's built-in benchmark tool reports (mAP@50, precision, recall) against the held-out validation split.

## Hardware Communication Protocol

Commands transmit over USB Serial (**9600 Baud**) as newline-terminated JSON payloads. This protocol is emitted by `dashboard/src/app.py` / `dashboard/src/Arduino.py` — **the FastAPI backend from Part 1 never talks to serial hardware.**

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

## Calibration Parameters

Physical constants for the benchtop setup, defined in [`dashboard/src/config.yaml`](dashboard/src/config.yaml) and read by `dashboard/src/app.py`:

```yaml
hardware:
  default_port: "COM3"
  baudrate: 9600

conveyor:
  speed_mm_s: 150.0
  direction: "DOWNWARD"
  grasping_zone_y_mm: 600.0

vision:
  trigger_line_ratio: 0.50     # Vertical trigger position (50% of frame height)
  trigger_tolerance_px: 25     # Acceptance band around the trigger line (pixels)
  scale_factor_mm: 1.5         # Pixel-to-mm scaling factor

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

> These are **benchtop test values**, not safety-rated production constants — recalibrate every value in `dashboard/src/config.yaml` against your physical rig before deployment.

## Full Documentation (PDF)

A complete, in-depth technical PDF covering the dashboard's architecture, code walkthrough, operator manual, developer/setup guide, and roadmap is available at:

📄 **[`dashboard/docs/`](dashboard/docs/)**

Refer to this document for a full narrative walkthrough of the dashboard beyond what is summarized in this README.

## Downloads — Installers (Windows & macOS)

Pre-built, ready-to-run desktop installers for the dashboard are published on the **[GitHub Releases](https://github.com/aratajaddini/ExpoChallenge-AraTajaddini/releases/tag/Dashboard)** page of this repository. No Python environment or source setup is required to use these:

| Platform | Package | Notes |
|---|---|---|
| 🪟 **Windows** | Standalone installer | Single installer for all supported Windows machines |
| 🍎 **macOS (Apple Silicon)** | Native installer | For M-series Macs (arm64) |
| 🍎 **macOS (Intel)** | Native installer | For Intel-based Macs (x86_64) |

> Make sure to download the correct macOS build for your Mac's chip (Apple Silicon vs. Intel) — the two are not interchangeable.

## Test the Dashboard (Sample Video)

Don't have a physical conveyor rig or webcam handy? You can evaluate the dashboard's detection, tracking, and kinematics pipeline using a pre-recorded sample video:

▶️ **[Download the test video](https://github.com/abbas-pt/ExpoChallenge_AbbasLotfi/releases/download/dashboard_v1.2/test1.mp4)**

Launch the dashboard, switch the input source to **Video File**, and upload this clip to see the full detection → tracking → kinematics → (simulated) robot-command pipeline in action, even without any Arduino connected.

## Running the Dashboard from Source

The dashboard is a separate application with its own dependencies (heavier: torch, ultralytics, OpenCV GUI build, Gradio, PyWebview) and its own virtual environment, independent from Part 1. Make sure you've [cloned the repository](#-clone-the-repository) first.

### Prerequisites
- Python 3.10+ (< 3.13)
- Arduino with USB connection (optional — the dashboard falls back to offline mode automatically)

### 1. Virtual environment

**On Windows:**
```bash
python -m venv dashboard\.venv
dashboard\.venv\Scripts\activate
```

**On macOS (Intel or Apple Silicon):**
```bash
python3 -m venv dashboard/.venv
source dashboard/.venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r dashboard/src/requirements.txt
```

### 3. Model weights

Detection weights are already included under [`dashboard/model-weights/`](dashboard/model-weights/), so no separate download is needed. Confirm `dashboard/src/app.py`'s `MODEL_PATH` points at the correct file, and that `dashboard/src/data.yaml` points at your dataset if you plan to run the in-app benchmark tool or retrain with `dashboard/src/train.py`.

### 4. Calibrate

Edit `dashboard/src/config.yaml` to match your physical rig (serial port, trigger line, belt speed, gripper forces, bin capacities) — the shipped values are benchtop test defaults, not production calibration. Note that the default serial port format differs by OS:
- **Windows:** typically `COM3`, `COM4`, etc.
- **macOS:** typically `/dev/tty.usbmodemXXXX` or `/dev/tty.usbserial-XXXX`.

The dashboard auto-discovers the correct port on startup even if the configured default doesn't match, but setting the right value avoids the initial failed-connection delay.

### 5. Run

**On Windows:**
```bash
python dashboard\src\app.py
```

**On macOS:**
```bash
python3 dashboard/src/app.py
```

This launches a native desktop window (via PyWebview) hosting the Gradio UI. If no Arduino is detected on startup, the dashboard continues in offline mode — reconnect later from the UI's "Reconnect Hardware" button, or use the [sample test video](#test-the-dashboard-sample-video) above to evaluate the pipeline without any hardware at all.

> Prefer not to build from source? Use the prebuilt [Windows/macOS installers](#downloads--installers-windows--macos) instead.

---

## 🧬 Operational Class Mapping (Dashboard)

This 18-to-5 mapping applies to **Part 2 (the dashboard)** only — it is how the TACO-trained detection model's 18 raw classes get collapsed into the 5 operational categories used for routing, priority, and gripper force. Part 1's TrashNet-based classifier already outputs one of five categories directly (glass, metal, paper, plastic, waste) and has no downstream robot, so no priority/force mapping applies to it.

| TACO Classes (18) | Operational Category | Priority | Gripper Force |
|:------------------|:---------------------|:---------|:--------------|
| Aluminium foil, Can, Pop tab | **Metal** | 1 (Highest) | 70 N |
| Bottle cap, Bottle, Lid, Other plastic, Plastic bag, Plastic container, Straw | **Plastic** | 2 | 50 N |
| Broken glass | **Glass** | 3 | 20 N |
| Carton, Cup, Paper | **Paper** | 4 | 85 N |
| Cigarette, Other litter, Styrofoam piece, Unlabeled litter | **Waste** | 5 (Lowest) | 60 N |

**Rationale:** the dashboard's priority queue resolves conflicts when multiple items cross the trigger line simultaneously (*Metal > Plastic > Glass > Paper > Waste*).

---

## 🧪 Testing & CI

*(Applies to Part 1 — the Backend API. The dashboard is validated via its built-in benchmark tool and the assets in [`dashboard/results/`](dashboard/results/), see [Model Evaluation & Results](#model-evaluation--results).)*

```bash
pip install -r backend/requirements-dev.txt
ruff check backend/
mypy backend/
bandit -r backend/
pytest backend/tests/ -q
```

Shared linting/testing configuration lives in the root [`pyproject.toml`](pyproject.toml) (pytest paths, ruff rules, mypy settings).

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR against `main`:
- **Test job:** installs dev dependencies and runs `pytest` with `REQUIRE_MODEL=0` so the suite doesn't depend on committed weights.
- **Lint job:** `ruff` (fatal-error rules block the build; full rule set runs advisory), `mypy`, and `bandit` for static analysis and security scanning.

---

## 🌱 Documentation Index (Chat Knowledge Base)

All documents live under [`backend/docs/kb/`](backend/docs/kb/) and are what Part 1's local documentation assistant (`/chat`) searches over.

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

> For the dashboard's own documentation, see the [PDF in `dashboard/docs/`](#full-documentation-pdf) instead — the dashboard is not indexed by the `/chat` assistant.

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

For inquiries, collaboration, or feedback regarding **Trace Sort AI**:

**Repository:** [https://github.com/aratajaddini/ExpoChallenge-AraTajaddini](https://github.com/aratajaddini/ExpoChallenge-AraTajaddini)

* **Project Lead / Frontend & Backend Developer:** Reza Esmaeili Mood
  📧 [esmaeilireza1994@gmail.com](mailto:esmaeilireza1994@gmail.com) · 🔗 [github.com/esmaeilireza](https://github.com/esmaeilireza)

* **AI Training / Dashboard Developer:** Abbas Lotfi
  📧 [abbasproptrader@gmail.com](mailto:abbasproptrader@gmail.com) · 🔗 [github.com/abbas-pt](https://github.com/abbas-pt)

* **Documentation:** Ara Tajaddini
  📧 [aratajaddini@gmail.com](mailto:aratajaddini@gmail.com) · 🔗 [github.com/aratajaddini](https://github.com/aratajaddini)

* **Support:** Sina Niknejad
  📧 [sinaniknejadi@gmail.com](mailto:sinaniknejadi@gmail.com) · 🔗 [https://github.com/Myr-at](https://github.com/Myr-at)
  

---

## 📄 License

See [LICENSE](LICENSE) for details (MIT).
