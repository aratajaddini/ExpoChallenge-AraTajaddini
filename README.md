# ♻️ Smart Waste Robot

**AI‑assisted conveyor sorting • YOLO vision • RAG knowledge base**

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV (headless) · SQLite

> 💡 **Overview** — A simulated conveyor belt classifies waste into five categories
> (plastic, metal, paper, glass, organic) with a custom‑trained YOLO detector.
> The model is served through a FastAPI backend protected by API‑key authentication
> and complemented by a Retrieval‑Augmented Generation chatbot for on‑the‑fly documentation.

---

## 🌱 Documentation Index

All documents live under [`backend/docs/kb/`](backend/docs/kb/).

### 🧩 Core Architecture

| # | 📄 Document | Description |
|---|---|---|
| 01 | [**Overview**](backend/docs/kb/01-overview.md) | Goals, components & high‑level diagram |
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
| 09 | [**Security & Keys**](backend/docs/kb/09-security-and-keys.md) | Auth workflow, API‑key handling |

---

## 🚀 Quick Start

**Requirements:** Python 3.11 · Git
```bash
git clone https://github.com/aratajaddini/smart-waste-robot.git
cd smart-waste-robot

**1. Virtual environment**

bash
python -m venv backend/.venv
# Windows
backend\.venv\Scripts\activate
# Unix / macOS
source backend/.venv/bin/activate

**2. Dependencies**

bash
pip install -r backend/requirements.txt

**3. Environment**

bash
# Windows
Copy-Item .env.example .env
# Unix / macOS
cp .env.example .env

Then set `API_KEY` in `.env`. The app will not start without it.

**4. Model weights**

Place `best.pt` in `backend/weights/` (untracked, not in the repo).
Without weights, set `REQUIRE_MODEL=0` in `.env` — the UI runs and `/predict` returns `503`.

**5. Run**

bash
uvicorn backend.main:app --reload --port 8001

Open **http://127.0.0.1:8001/**

---

## 🧠 Rebuild the Knowledge Base

Run from the repository root:

bash
python -m backend.tools.build_kb

Generates `backend/data/kb_index.npz` (untracked).
```
