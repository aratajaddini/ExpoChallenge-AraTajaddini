# Inference Pipeline

## Model loading
The model layer is independent of the web framework. It wraps Ultralytics
YOLOv11 and exposes plain functions that take an image or a file path and return
predictions. The weights are loaded once per process and cached, so the file on
disk is not re-read on every request. The weights live at
backend/weights/best.pt and are not tracked in Git; the server refuses to start
if they are missing.

## Image requests
For an image upload the pipeline validates the file extension, stores the file
in the uploads directory, runs a single forward pass, and returns the predicted
class together with its confidence. Supported image formats are JPG, JPEG, PNG,
BMP, and WEBP.

## Video requests
For a video upload the backend opens the clip with OpenCV and samples frames
instead of processing every frame. At most 120 frames are sampled per clip,
which is roughly one frame per second for a two-minute video. Each sampled
frame is classified, frames below the 0.35 confidence floor are counted as
uncertain, and the remaining per-class counts are aggregated into a summary. The
dominant class of the clip is the class with the highest count. Supported video
formats are MP4, MOV, AVI, MKV, and WEBM.

## Upload limits
Any upload larger than 150 MB is rejected. The limit exists to keep frame
sampling and disk usage bounded on a machine without a GPU, since all inference
runs on CPU.

## Detection model
Besides the classifier there is a separate detection model used for analytics,
stored at backend/weights/taco_det.pt. Its detections are kept only when their
confidence reaches 0.40.
