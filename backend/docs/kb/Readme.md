# ♻️ Smart Waste Robot

**AI‑assisted conveyor sorting • YOLO vision • RAG knowledge base**

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV (headless) · SQLite

> 💡 **Overview** – A simulated conveyor belt classifies waste into five categories
> (plastic, metal, paper, glass, organic) with a custom‑trained YOLO detector.
> The model is served through a FastAPI backend protected by API‑key authentication
> and complemented by a Retrieval‑Augmented Generation chatbot for on‑the‑fly documentation.

---

## 🌱 Documentation Index

All files live under `backend/docs/kb/` on the `add-validate-model-fix` branch.

### 🧩 Core Architecture

| # | 📄 Document | 🔗 Link |
|---|---|---|
| 01 | **Overview** – goals, components & high‑level diagram | [01‑overview](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/01-overview.md) |
| 02 | **Classes** – specs for the five waste classes | [02‑classes](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/02-classes.md) |
| 03 | **Pipeline** – data flow, inference & routing | [03‑pipeline](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/03-pipeline.md) |
| 04 | **API** – REST endpoints, request / response formats | [04‑api](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/04-api.md) |

### ⚛️ AI & Models

| # | 📄 Document | 🔗 Link |
|---|---|---|
| 05 | **Training** – datasets, methodology & evaluation | [05‑training](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/05-training.md) |
| 10 | **YOLO Versions** – version comparison & selection rationale | [10‑yolo‑versions](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/10-yolo-versions.md) |
| 11 | **Alternative Models** – other CV/DL approaches examined | [11‑alternative‑models](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/11-alternative-models.md) |
| 12 | **Future Models** – roadmap for research & optimisation | [12‑future‑models](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/12-future-models.md) |

### 💬 Chatbot & Knowledge Base

| # | 📄 Document | 🔗 Link |
|---|---|---|
| 06 | **RAG & Chat** – architecture & generation pipeline | [06‑rag‑and‑chat](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/06-rag-and-chat.md) |
| 07 | **FAQ** – common questions & troubleshooting tips | [07‑faq](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/07-faq.md) |

### 🛡️ Operations & Security

| # | 📄 Document | 🔗 Link |
|---|---|---|
| 08 | **Failure Modes** – known issues & recovery strategies | [08‑failure‑modes](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/08-failure-modes.md) |
| 09 | **Security & Keys** – auth workflow, API‑key handling | [09‑security‑and‑keys](https://github.com/aratajaddini/smart-waste-robot/tree/add-validate-model-fix/backend/docs/kb/09-security-and-keys.md) |

---

## Quickstart

1. Copy the env template and fill `API_KEY`:
   - PowerShell: `Copy-Item .env.example .env`
   - Bash: `cp .env.example .env`
2. Weights: ask [@abbas-pt](https://github.com/abbas-pt) for `best.pt` → `backend/weights/best.pt` (untracked).
   No weights? set `REQUIRE_MODEL=0` — UI works, `/predict` returns 503.
3. Activate the virtualenv:
   - Windows: `backend\.venv\Scripts\activate`
   - Unix/macOS: `source backend/.venv/bin/activate`
4. From repo root: `uvicorn backend.main:app --reload --port 8001`
5. Open http://127.0.0.1:8001/


🔍 Model Version Clarification
The project uses a YOLO‑based production baseline with custom trained weights (best.pt).
For historical context and version comparisons, refer to 10-yolo-versions.md.
The deployed pipeline is not hard‑coded to a specific YOLO version; it loads the weights provided at runtime.

🤝 Contributing
Contributions are welcome! Please check the issue tracker and submit pull requests with clear descriptions.

📄 License
This project is open source under the MIT License – see the LICENSE file for details.
