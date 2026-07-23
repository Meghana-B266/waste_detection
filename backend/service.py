"""
Detection Service

Runs the camera capture + detection loop in a background thread, so the
FastAPI web server (backend/api.py) can stream live video and stats to any
browser on the network — no local display, no cv2.imshow window required.

This mirrors the detection logic in run_complete_system.py, wired up to
run headlessly and expose its state (latest frame, live stats) for the API
to read.
"""

import os
import time
import threading
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from backend.database import DatabaseManager, init_db
from backend.alerts import AlertManager
from backend.water_enhancement import WaterEnhancer, MultiScaleDetector, AdaptiveThreshold, SmartBoxFilter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTION_FOLDER = PROJECT_ROOT / "detections"
DETECTION_FOLDER.mkdir(exist_ok=True)

COLOR_HIGH = (0, 255, 0)
COLOR_MED = (0, 255, 255)
COLOR_LOW = (0, 165, 255)


class DetectionService:
    """Owns the camera connection, detection loop, and all shared live state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.phone_ip = os.getenv('PHONE_IP', '').strip()
        self.camera_name = "Mobile Detection Camera"

        # Detection config (same defaults as run_complete_system.py)
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.35))
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.35))
        self.min_box_area = int(os.getenv('MIN_BOX_AREA', 800))
        self.image_size = 1280
        self.smoothing_window = 30
        self.stability_frames = 10
        self.stability_threshold = 0.65
        self.report_cooldown = 2.0  # seconds between DB writes for a changed count

        self.use_water_enhancement = os.getenv('USE_WATER_ENHANCEMENT', 'true').lower() == 'true'
        self.use_multiscale = os.getenv('USE_MULTISCALE', 'true').lower() == 'true'
        self.use_adaptive_threshold = os.getenv('USE_ADAPTIVE_THRESHOLD', 'true').lower() == 'true'
        self._tracking_available = True  # flips to False if 'lap' package is missing

        # State the API reads
        self.latest_frame: Optional[bytes] = None
        self.latest_stats = {
            'running': False,
            'camera_connected': False,
            'stable_count': 0,
            'raw_count': 0,
            'avg_confidence': 0.0,
            'fps': 0.0,
            'status': 'stopped',
            'phone_ip': self.phone_ip,
            'emails_sent_today': 0,
            'error': None,
        }

        self._build_placeholder_frame()
        self._load_model_and_services()

    def _build_placeholder_frame(self):
        """A simple 'not connected' image shown before detection starts."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.putText(img, "Camera not connected", (110, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(img, "Enter phone IP and press Start", (95, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        ok, buffer = cv2.imencode('.jpg', img)
        self.placeholder_jpeg = buffer.tobytes() if ok else b''

    def _load_model_and_services(self):
        model_path = os.getenv('MODEL_PATH', 'models/best.pt')
        print(f"[service] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        init_db()
        self.db = DatabaseManager()
        self.alert_mgr = AlertManager(db_manager=self.db)

        self.water_enhancer = WaterEnhancer()
        self.adaptive_threshold = AdaptiveThreshold(base_threshold=self.confidence_threshold)
        self.smart_filter = SmartBoxFilter(min_area=self.min_box_area)
        self.multiscale_detector = MultiScaleDetector(self.model) if self.use_multiscale else None

        print("[service] Model and services ready")

    # ---------------- control ----------------

    def start(self, phone_ip: Optional[str] = None) -> dict:
        with self.lock:
            if self.running:
                return {'ok': False, 'message': 'Detection is already running'}

            if phone_ip:
                self.phone_ip = phone_ip.strip()

            if not self.phone_ip:
                return {'ok': False, 'message': 'No phone IP provided. Enter one and try again.'}

            camera_url = f"http://{self.phone_ip}:8080/video"
            cap = cv2.VideoCapture(camera_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                cap.release()
                self.latest_stats['error'] = f'Could not connect to {camera_url}'
                return {'ok': False, 'message': self.latest_stats['error']}

            self.cap = cap
            self.running = True
            self.latest_stats.update({
                'running': True,
                'camera_connected': True,
                'phone_ip': self.phone_ip,
                'error': None,
            })

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return {'ok': True, 'message': f'Detection started on {camera_url}'}

    def stop(self) -> dict:
        with self.lock:
            self.running = False

        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.latest_stats.update({
            'running': False,
            'camera_connected': False,
            'status': 'stopped',
        })
        return {'ok': True, 'message': 'Detection stopped'}

    def get_stats(self) -> dict:
        return dict(self.latest_stats)

    def get_latest_jpeg(self) -> bytes:
        with self.lock:
            frame = self.latest_frame
        return frame if frame is not None else self.placeholder_jpeg

    # ---------------- helpers ----------------

    @staticmethod
    def _get_stable_count(history, min_frames, threshold):
        if len(history) < min_frames:
            return None
        recent = list(history)[-min_frames:]
        count_freq = Counter(recent)
        if not count_freq:
            return 0
        most_common_count, frequency = count_freq.most_common(1)[0]
        if frequency / min_frames >= threshold:
            return most_common_count
        return None

    def _save_detection_image(self, frame, count, confidence) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = DETECTION_FOLDER / f"waste_{count}items_{confidence:.0%}_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        return str(filename)

    def _annotate(self, frame, boxes, confidences, stable_count, avg_confidence, fps, status_text):
        annotated = frame.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = confidences[i]
            color = COLOR_HIGH if conf > 0.85 else COLOR_MED if conf > 0.70 else COLOR_LOW
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"Plastic {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (360, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.putText(annotated, f"COUNT: {stable_count}", (15, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated, f"Confidence: {avg_confidence:.1%}", (15, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, status_text, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return annotated

    # ---------------- main loop ----------------

    def _loop(self):
        detection_history = deque(maxlen=self.smoothing_window)
        confidence_history = deque(maxlen=self.smoothing_window)
        current_stable_count = 0
        last_report_time = 0.0
        fps_start = time.time()
        fps_count = 0
        current_fps = 0.0

        print("[service] Detection loop started")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[service] Frame read failed — reconnecting in 2s...")
                self.latest_stats['error'] = 'Lost connection to camera, retrying...'
                self.cap.release()
                time.sleep(2)
                if not self.running:
                    break
                self.cap = cv2.VideoCapture(f"http://{self.phone_ip}:8080/video")
                continue

            self.latest_stats['error'] = None

            fps_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0

            enhanced_frame = self.water_enhancer.auto_enhance(frame) if self.use_water_enhancement else frame

            current_conf = self.adaptive_threshold.calculate(confidence_history) \
                if self.use_adaptive_threshold else self.confidence_threshold

            try:
                if self._tracking_available:
                    if self.use_multiscale:
                        results = self.multiscale_detector.detect(enhanced_frame, current_conf)
                    else:
                        results = self.model.track(
                            enhanced_frame, conf=current_conf, iou=self.iou_threshold,
                            imgsz=self.image_size, verbose=False, persist=True,
                            tracker="bytetrack.yaml", max_det=20
                        )
                else:
                    # Tracking unavailable (missing 'lap' package) — plain detection still
                    # works fine for counting; we just lose per-object track IDs.
                    results = self.model.predict(
                        enhanced_frame, conf=current_conf, imgsz=self.image_size, verbose=False
                    )
            except Exception as e:
                if self._tracking_available and 'lap' in str(e).lower():
                    print("[service] NOTE: object tracking needs the 'lap' package, which isn't "
                          "installed (and failed to auto-install — often a Windows permissions issue "
                          "when not running inside a venv). Falling back to plain detection; counts "
                          "still work normally. To enable tracking, run: pip install lap")
                    self._tracking_available = False
                    continue
                print(f"[service] Detection error: {e}")
                continue

            raw_boxes = results[0].boxes
            filtered_boxes, confidences, _track_ids = self.smart_filter.filter(raw_boxes, enhanced_frame.shape)
            current_count = len(filtered_boxes)

            if current_count > 0:
                avg_conf = float(np.mean(confidences))
                confidence_history.append(avg_conf)
            else:
                confidence_history.append(0)

            detection_history.append(current_count)
            avg_confidence = float(np.mean([c for c in confidence_history if c > 0])) \
                if any(confidence_history) else 0.0

            stable_count = self._get_stable_count(detection_history, self.stability_frames, self.stability_threshold)

            current_time = time.time()
            if stable_count is not None and stable_count != current_stable_count:
                if current_time - last_report_time >= self.report_cooldown:
                    current_stable_count = stable_count
                    last_report_time = current_time

                    if stable_count > 0:
                        save_frame = frame.copy()
                        for i, box in enumerate(filtered_boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = confidences[i]
                            color = COLOR_HIGH if conf > 0.85 else COLOR_MED if conf > 0.70 else COLOR_LOW
                            cv2.rectangle(save_frame, (x1, y1), (x2, y2), color, 3)

                        img_path = self._save_detection_image(save_frame, stable_count, avg_confidence)

                        try:
                            loc_info = self.alert_mgr.get_current_location_info()
                            detection_id = self.db.save_detection({
                                'timestamp': datetime.now(),
                                'camera_name': self.camera_name,
                                'location': loc_info['location'],
                                'latitude': loc_info['latitude'],
                                'longitude': loc_info['longitude'],
                                'waste_count': stable_count,
                                'confidence_avg': avg_confidence,
                                'confidence_max': float(np.max(confidences)) if confidences else 0.0,
                                'fps': current_fps,
                                'image_path': img_path,
                            })

                            if self.alert_mgr.should_send_alert(stable_count):
                                self.alert_mgr.trigger_alert(detection_id, stable_count, avg_confidence, img_path)
                        except Exception as e:
                            print(f"[service] Error logging detection: {e}")

            if len(detection_history) >= self.stability_frames:
                recent = list(detection_history)[-self.stability_frames:]
                stability_pct = (recent.count(current_stable_count) / self.stability_frames) * 100
                if stability_pct >= 90:
                    status = "LOCKED"
                elif stability_pct >= 75:
                    status = "STABLE"
                elif stability_pct >= 60:
                    status = "TRACKING"
                else:
                    status = "UNSTABLE"
            else:
                status = "CALIBRATING"

            annotated = self._annotate(frame, filtered_boxes, confidences,
                                        current_stable_count, avg_confidence, current_fps, status)

            ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.latest_frame = buffer.tobytes()

            self.latest_stats.update({
                'running': True,
                'camera_connected': True,
                'stable_count': current_stable_count,
                'raw_count': current_count,
                'avg_confidence': avg_confidence,
                'fps': round(current_fps, 1),
                'status': status,
                'phone_ip': self.phone_ip,
                'emails_sent_today': self.alert_mgr.email_sent_today,
            })

        print("[service] Detection loop stopped")


# Single shared instance used by backend/api.py.
# Loading the model here (at import time) means the first request doesn't
# have to wait for it — it's ready as soon as the server starts.
detection_service = DetectionService()
