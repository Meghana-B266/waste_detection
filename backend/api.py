"""
FastAPI Web Server

Exposes the detection service over HTTP so any browser on the same network
(phones, laptops, tablets) can view the live annotated stream and detection
history — no local display and no separate OpenCV window required.

Run with:
    python run_web_server.py
or directly:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000
"""

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.service import detection_service
from backend.database import DatabaseManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Waste Detection API")
db = DatabaseManager()


class StartRequest(BaseModel):
    phone_ip: Optional[str] = None


@app.post("/api/start")
def start_detection(req: StartRequest):
    """Connect to the phone camera and start the detection loop."""
    return detection_service.start(phone_ip=req.phone_ip)


@app.post("/api/stop")
def stop_detection():
    """Stop the detection loop and release the camera."""
    return detection_service.stop()


@app.get("/api/stats")
def get_stats():
    """Live stats: current count, confidence, fps, status."""
    return detection_service.get_stats()


@app.get("/api/detections")
def get_detections(limit: int = 50):
    """Recent detection history from the database."""
    rows = db.get_recent_detections(limit=limit)
    return [
        {
            'id': r.id,
            'timestamp': r.timestamp.isoformat() if r.timestamp else None,
            'camera_name': r.camera_name,
            'location': r.location,
            'waste_count': r.waste_count,
            'confidence_avg': r.confidence_avg,
            'fps': r.fps,
            'alert_sent': r.alert_sent,
            'image_url': f"/api/detections/{r.id}/image" if r.image_path else None,
        }
        for r in rows
    ]


@app.get("/api/detections/{detection_id}/image")
def get_detection_image(detection_id: int):
    """Serve the saved annotated image for one detection."""
    rows = db.get_recent_detections(limit=1000)
    match = next((r for r in rows if r.id == detection_id), None)
    if not match or not match.image_path or not Path(match.image_path).exists():
        return JSONResponse(status_code=404, content={'error': 'Image not found'})
    return FileResponse(match.image_path)


def _mjpeg_generator():
    """Yields the latest annotated frame as a multipart JPEG stream."""
    while True:
        frame = detection_service.get_latest_jpeg()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)  # cap the stream at ~20 fps


@app.get("/api/video_feed")
def video_feed():
    """Live MJPEG video stream — point an <img> tag at this URL."""
    return StreamingResponse(_mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


# Serve the frontend (index.html, style.css, app.js) for every other path.
# This must be added LAST — routes above are matched first.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
