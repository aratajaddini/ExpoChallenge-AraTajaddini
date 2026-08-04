# Alternative Approaches for Waste Sorting

While YOLO is the primary model in this project, the waste sorting domain includes many other deep‑learning techniques. This document summarises the most common alternatives.

## 1. Traditional Convolutional Neural Networks (CNNs)

**Examples**: ResNet, VGG, DenseNet, Inception.

**How they work**: These models are trained as image classifiers. They take an image and output a single class label (e.g., "plastic", "glass").

**Pros**:
- High accuracy on clean, single‑object images.
- Well‑studied, many pre‑trained weights available.

**Cons**:
- Cannot localise multiple objects in one image.
- Slower on edge devices (except MobileNet).

**Applied**: TrashNet benchmark achieves ~90% accuracy with ResNet‑50.

---

## 2. Lightweight Mobile Models

**Examples**: MobileNetV2, MobileNetV3, ShuffleNetV2, EfficientNet‑Lite.

**How they work**: Similar to CNNs but optimised for low‑power devices (Raspberry Pi, mobile phones).

**Pros**:
- Fast inference on CPU.
- Small model size (< 10 MB).

**Cons**:
- Lower accuracy compared to larger models.

**Applied**: Many edge‑based waste‑sorting systems use MobileNetV2 (see `02-classes.md` for details).

---

## 3. Object Detection (Two‑Stage Detectors)

**Examples**: Faster R‑CNN, Mask R‑CNN, Cascade R‑CNN.

**How they work**: First, a region proposal network suggests candidate boxes; then a classifier refines them.

**Pros**:
- Very high accuracy (often better than YOLO).

**Cons**:
- Too slow for real‑time (< 10 FPS on GPU).
- Requires more computational resources.

**Applied**: Used in high‑precision trash detection for underwater robotics (TrashICRA19 dataset).

---

## 4. Vision Transformers (ViT)

**Examples**: Vision Transformer, Swin Transformer, DETR.

**How they work**: Use self‑attention instead of convolutions; process images as sequences of patches.

**Pros**:
- Can capture long‑range dependencies.
- State‑of‑the‑art accuracy on some benchmarks.

**Cons**:
- Extremely heavy (hundreds of MBs).
- Slow inference, expensive to train.

**Applied**: Emerging in trash‑classification research (GCNet), but not yet practical for edge deployment.

---

## 5. Hybrid CNN + LSTM / Transformer

**Examples**: CNN‑LSTM, CNN‑Transformer.

**How they work**: A CNN extracts spatial features; then an LSTM/Transformer models temporal dependencies (useful for video).

**Pros**:
- Can leverage multiple frames for better decisions.

**Cons**:
- Added complexity and latency.

**Applied**: Used in some research for video‑based waste detection (e.g., Graph‑LSTM).

---

## 6. Federated Learning for Waste Sorting

**How it works**: Multiple waste‑sorting stations train their local models and share updates with a central server, combining their collective knowledge.

**Pros**:
- Improves generalisation without sharing private data.

**Cons**:
- Communication overhead, complex implementation.

**Applied**: Recent research (TrashBox dataset) suggests ResNeXt‑101 as a good candidate.

---

## Why We Use YOLO11 in This Project

- **Real‑time performance** – essential for physical sorting actuators.
- **Single‑shot detection** – handles localisation and classification in one pass.
- **Project baseline** – the deployed pipeline uses the trained `best.pt` weights.
- **Ultralytics integration** – supports the project's training and inference workflow.
- **Deployment suitability** – the model is evaluated against the project's five waste classes.

YOLO11 is currently the production baseline because it provides a strong balance of speed, accuracy, and integration for the implemented requirements. Earlier versions such as YOLOv8 remain historical alternatives, not the current production model.

The model is evaluated against the project's five waste classes.

---

*Related docs: `05-training.md` (training details), `03-pipeline.md` (inference pipeline), `10-yolo-versions.md` (YOLO details).*