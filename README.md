# ♻️ Smart Waste Robot – Technical Documentation

AI‑assisted waste sorting: a conveyor‑belt simulation classifies items into five
categories (**plastic, metal, paper, glass, organic**) via a YOLO‑based detector,
exposed through a FastAPI backend with API‑key authentication and a RAG‑powered
documentation chatbot.

**Stack:** Python 3.11 · FastAPI · Ultralytics YOLO · OpenCV (headless) · SQLite

---

## 📚 Documentation Index

### 🧩 Core Architecture
- [01 Overview](./kb/01-overview.md) – Project goals, key components, and high‑level system architecture.
- [02 Classes](./kb/02-classes.md) – Definitions and specifications for the five supported waste classes.
- [03 Pipeline](./kb/03-pipeline.md) – Data processing, classification, and inference pipeline.
- [04 API](./kb/04-api.md) – REST API endpoints, request formats, and response structures.

### 🔬 AI & Models
- [05 Training](./kb/05-training.md) – Model training methodology, datasets, and evaluation process.
- [10 YOLO Versions](./kb/10-yolo-versions.md) – Review of YOLO versions and the rationale behind the final model selection.
- [11 Alternative Models](./kb/11-alternative-models.md) – Comparison with alternative computer vision and deep learning approaches.
- [12 Future Models](./kb/12-future-models.md) – Roadmap for future model research, optimisation, and development.

### 💬 Chatbot & Knowledge Base
- [06 RAG and Chat](./kb/06-rag-and-chat.md) – Chatbot architecture and Retrieval‑Augmented Generation implementation.
- [07 FAQ](./kb/07-faq.md) – Frequently asked questions about the system, models, and supported workflows.

### 🔐 Operations & Security
- [08 Failure Modes](./kb/08-failure-modes.md) – Known failure scenarios, troubleshooting procedures, and recovery strategies.
- [09 Security & Keys](./kb/09-security-and-keys.md) – Authentication, API key management, and system security practices.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build the knowledge‑base index (required for the chat assistant)
python -m backend.tools.build_kb

# Start the server (default port 8000, use --port to change)
uvicorn backend.main:app --reload
Note: The trained model weights (backend/weights/best.pt) are not tracked in this repository – they are generated during training (e.g., on Colab) and must be placed at that path before running inference.

🔍 Model Version Clarification
The project uses a YOLO‑based production baseline with custom trained weights (best.pt).
For historical context and version comparisons, refer to 10-yolo-versions.md.
The deployed pipeline is not hard‑coded to a specific YOLO version; it loads the weights provided at runtime.

🤝 Contributing
Contributions are welcome! Please check the issue tracker and submit pull requests with clear descriptions.

📄 License
This project is open source under the MIT License – see the LICENSE file for details.
