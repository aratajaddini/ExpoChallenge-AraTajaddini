# YOLO Versions and Selection

## What is YOLO?

YOLO (You Only Look Once) is a family of real‑time object detection models. It frames detection as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation. This makes it extremely fast compared to two‑stage detectors like Faster R‑CNN.

## YOLO Versions Overview

| Version | Year | Key Features |
|---------|------|--------------|
| **YOLOv1** | 2016 | First version; unified detection, 45 FPS. |
| **YOLOv2** | 2017 | Better accuracy, multi‑scale training, anchor boxes. |
| **YOLOv3** | 2018 | Feature pyramid networks, better small‑object detection. |
| **YOLOv4** | 2020 | Bag of freebies (Mosaic augmentation, self‑adversarial training), CSPDarknet53 backbone. |
| **YOLOv5** | 2020 | PyTorch implementation, easy to use, good speed‑accuracy trade‑off. |
| **YOLOv6** | 2022 | RepVGG backbone, anchor‑free, faster inference. |
| **YOLOv7** | 2022 | Trainable bag‑of‑freebies, E‑ELAN architecture, state‑of‑the‑art at the time. |
| **YOLOv8** | 2023 | **Anchor‑free**, unified framework (detection, segmentation, classification), improved accuracy and speed, easy export to ONNX/TensorRT. |
| **YOLOv9** | 2024 | Programmable gradient information (PGI), GELAN architecture, even better accuracy. |
| **YOLOv10** | 2024 | Real‑time end‑to‑end, no NMS needed, extremely fast. |
| **YOLO11** | 2025 | Latest production baseline; improved efficiency, higher accuracy, better deployment flexibility. |

## Which Version Does This Project Use?

**This project currently uses YOLO11 as its production baseline.** The deployed custom weights are stored as `best.pt`.

Reasons for choosing YOLO11:

- **Strong real‑time performance** – suitable for conveyor and edge‑assisted waste sorting.
- **Good accuracy/speed trade‑off** – appropriate for the project's five waste classes.
- **Ultralytics integration** – supports the project's training, inference, and deployment workflow.
- **Existing project integration** – the trained `best.pt` weights are used by the deployed pipeline.

## Historical Comparison with Earlier Versions

- **YOLOv8** is a historically important Ultralytics version and is documented here for comparison.
- **YOLOv9** introduced Programmable Gradient Information (PGI) and the GELAN architecture.
- **YOLOv10** focused on end‑to‑end detection and reducing dependence on Non‑Maximum Suppression (NMS).
- The implemented production baseline for this project is **YOLO11**, not YOLOv8.

## Future Considerations

If higher accuracy is required, larger YOLO11 variants can be evaluated, or the current YOLO11-based pipeline can be retrained with additional data. If speed is the priority, the deployed lightweight configuration should be benchmarked before changing the production baseline.

Newer YOLO versions and alternative detectors can be evaluated during major model updates.

---

*Related docs: `05-training.md` (training details), `03-pipeline.md` (inference pipeline).*