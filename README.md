♻️ Smart Waste Robot
AI‑assisted conveyor sorting • YOLO vision • RAG knowledge base

Stack: Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV (headless) · SQLite

💡 Overview – A simulated conveyor belt classifies waste into five categories (plastic, metal, paper, glass, organic) with a custom‑trained YOLO detector. The model is served through a FastAPI backend protected by API‑key authentication and complemented by a Retrieval‑Augmented Generation chatbot for on‑the‑fly documentation.

🌱 Documentation Index (all files live under backend/docs/kb/ on the release/backend-rag-v1.3 branch)
🧩 Core Architecture
#	📄 Document	🔗 Link
01	Overview – goals, components & high‑level diagram	01‑overview
02	Classes – specs for the five waste classes	02‑classes
03	Pipeline – data flow, inference & routing	03‑pipeline
04	API – REST endpoints, request / response formats	04‑api
⚛️ AI & Models
#	📄 Document	🔗 Link
05	Training – datasets, methodology & evaluation	05‑training
10	YOLO Versions – version comparison & selection rationale	10‑yolo‑versions
11	Alternative Models – other CV/DL approaches examined	11‑alternative‑models
12	Future Models – roadmap for research & optimisation	12‑future‑models
💬 Chatbot & Knowledge Base
#	📄 Document	🔗 Link
06	RAG & Chat – architecture & generation pipeline	06‑rag‑and‑chat
07	FAQ – common questions & troubleshooting tips	07‑faq
🛡️ Operations & Security
#	📄 Document	🔗 Link
08	Failure Modes – known issues & recovery strategies	08‑failure‑modes
09	Security & Keys – auth workflow, API‑key handling	09‑security‑and‑keys

## 🚀 Quick Start

Get the project running in three simple steps:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the knowledge-base index
# Required for the RAG documentation chatbot
python -m backend.tools.build_kb

# 3. Start the FastAPI server
uvicorn backend.main:app --reload
Server URL: http://127.0.0.1:8001

