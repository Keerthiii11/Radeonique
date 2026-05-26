# Radeonique — Leaf Disease Detection System

A Raspberry Pi-based system that detects plant diseases from camera images 
and automatically triggers fertilizer spraying based on the classification result.

## What it does
- Captures live images using a camera connected to Raspberry Pi
- Classifies leaf diseases using a trained TensorFlow model
- Serves predictions through a Flask web API
- Displays results on a simple web dashboard

## Tech Stack
- **Hardware:** Raspberry Pi 3 Model B+
- **ML:** TensorFlow, OpenCV
- **Backend:** Flask (Python)
- **Frontend:** HTML, JavaScript

## Project Structure
| File | Purpose |
|------|---------|
| `Training.py` | Model training pipeline |
| `app.py` | Flask API and routing |
| `dataset.py` | Dataset loading and preprocessing |
| `Prediction_cam1.py` | Live camera prediction |
| `index.html` | Web dashboard |

## How to Run
1. Clone the repo
2. Install dependencies: `pip install tensorflow flask opencv-python numpy`
3. Run the app: `python app.py`
4. Open browser at `http://localhost:5000`

## Status
Completed as part of B.Tech final year project — 
model trained on common leaf disease datasets with reliable 
classification across varying lighting conditions.

