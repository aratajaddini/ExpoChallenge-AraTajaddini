# Smart Waste Robot — Overview

## What the system does
Smart Waste Robot is an AI-powered waste sorting system. A camera above a
conveyor belt captures items, a YOLOv11 classifier assigns each item to one of
five waste categories, and the result drives a sorting actuator in the
simulation. The five categories are plastic, metal, paper, glass, and organic.
Class names are always read from the trained model itself, never hard-coded, so
retraining with different labels does not require code changes.

## Architecture
The system has three layers. The model layer wraps Ultralytics YOLOv11 and is
completely independent of the web framework; it is loaded once and cached so the
weights are not re-read on every request. The API layer is a FastAPI service
that exposes upload, prediction, analytics, and key-management endpoints. The
frontend is a static HTML/CSS/JavaScript demo page that talks to the API over
HTTP.

## Inputs and limits
The API accepts both images and video clips. Supported image formats are JPG,
JPEG, PNG, BMP, and WEBP. Supported video formats are MP4, MOV, AVI, MKV, and
WEBM. Uploads are limited to 150 MB. For video, the backend samples up to 120
frames, roughly one frame per second for a two-minute clip. Frames whose
confidence falls below 0.35 are counted as uncertain rather than being forced
into a class.

## Security
Every protected endpoint requires an API key sent by the client. The key is
never stored in plaintext and never committed to the repository; it is read
from the environment at startup, and the server refuses to start if it is
missing or if the model weights are absent. Cross-origin access is restricted
to an explicit allow-list of origins.

## Storage
Detection results are persisted in a SQLite database. The database path is
configurable through an environment variable so packaged tools can point at the
real database instead of a temporary extraction folder.
