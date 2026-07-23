"""
YOLO Detection Engine with Object Tracking

This module wraps Ultralytics YOLO + ByteTrack into a small class so the
detection logic can be reused from a script, a FastAPI route, or a notebook,
instead of copy-pasting the detection loop everywhere.

Note: run_complete_system.py currently contains its own inline copy of this
logic (with extra water-enhancement steps). Swapping it to use WasteDetector
directly is a natural next refactor, but is left as-is for now so behavior
doesn't change unexpectedly.
"""

import os
from collections import deque, Counter

import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


class WasteDetector:
    """YOLO-based waste detection with temporal smoothing/tracking."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv('MODEL_PATH', 'models/best.pt')
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.65))
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.4))
        self.min_box_area = int(os.getenv('MIN_BOX_AREA', 2000))

        # Tracking parameters
        self.smoothing_window = 25
        self.stability_frames = 18
        self.detection_history = deque(maxlen=self.smoothing_window)
        self.confidence_history = deque(maxlen=self.smoothing_window)

        # Metrics
        self.total_detections = 0
        self.high_conf_detections = 0
        self.current_stable_count = 0

        self.load_model()

    def load_model(self):
        """Load the YOLO model and warm it up with a dummy frame."""
        try:
            print(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            print("Model loaded and warmed up")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calculate_box_area(self, box):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return (x2 - x1) * (y2 - y1)

    def filter_detections(self, boxes):
        """Filter raw detections by confidence and box size."""
        if boxes is None or len(boxes) == 0:
            return [], []

        valid_boxes = []
        valid_confidences = []

        for box in boxes:
            conf = box.conf[0].item()
            area = self.calculate_box_area(box)

            if conf >= self.confidence_threshold and area >= self.min_box_area:
                valid_boxes.append(box)
                valid_confidences.append(conf)

        return valid_boxes, valid_confidences

    def get_stable_count(self, min_frames: int = None):
        """Return the most frequent recent count if it's stable enough, else None."""
        if min_frames is None:
            min_frames = self.stability_frames

        if len(self.detection_history) < min_frames:
            return None

        recent = list(self.detection_history)[-min_frames:]
        count_freq = Counter(recent)

        if not count_freq:
            return 0

        most_common_count, frequency = count_freq.most_common(1)[0]

        if frequency >= min_frames * 0.85:
            return most_common_count

        return None

    def detect(self, frame, track: bool = True):
        """
        Run detection on a single frame.

        Args:
            frame: Input image (numpy array, BGR)
            track: Enable object tracking (ByteTrack)

        Returns:
            dict with detection results and running metrics
        """
        try:
            if track:
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=640,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",
                    max_det=15
                )
            else:
                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    imgsz=640,
                    verbose=False
                )

            raw_boxes = results[0].boxes
            filtered_boxes, confidences = self.filter_detections(raw_boxes)
            current_count = len(filtered_boxes)

            if current_count > 0:
                avg_conf = float(np.mean(confidences))
                max_conf = float(np.max(confidences))
                self.confidence_history.append(avg_conf)

                for c in confidences:
                    self.total_detections += 1
                    if c > 0.7:
                        self.high_conf_detections += 1
            else:
                avg_conf = 0.0
                max_conf = 0.0
                self.confidence_history.append(0.0)

            self.detection_history.append(current_count)

            avg_confidence = float(np.mean([c for c in self.confidence_history if c > 0])) \
                if any(self.confidence_history) else 0.0

            stable_count = self.get_stable_count()
            if stable_count is not None:
                self.current_stable_count = stable_count

            if len(self.detection_history) >= self.stability_frames:
                recent = list(self.detection_history)[-self.stability_frames:]
                stability = (recent.count(self.current_stable_count) / self.stability_frames) * 100
            else:
                stability = 0.0

            high_conf_ratio = (self.high_conf_detections / self.total_detections * 100) \
                if self.total_detections > 0 else 0.0

            return {
                'raw_count': len(raw_boxes),
                'filtered_count': current_count,
                'stable_count': self.current_stable_count,
                'boxes': filtered_boxes,
                'confidences': confidences,
                'avg_confidence': avg_confidence,
                'max_confidence': max_conf,
                'stability': stability,
                'high_conf_ratio': high_conf_ratio,
                'raw_boxes': raw_boxes
            }

        except Exception as e:
            print(f"Detection error: {e}")
            return {
                'raw_count': 0,
                'filtered_count': 0,
                'stable_count': 0,
                'boxes': [],
                'confidences': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'stability': 0.0,
                'high_conf_ratio': 0.0,
                'error': str(e)
            }

    def annotate_frame(self, frame, detection_result):
        """Draw detection boxes and an info panel on a copy of the frame."""
        annotated = frame.copy()
        boxes = detection_result['boxes']
        confidences = detection_result['confidences']
        raw_boxes = detection_result.get('raw_boxes')

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = confidences[i]

            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.7:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Plastic {conf:.2%}"
            if raw_boxes is not None and raw_boxes.id is not None and i < len(raw_boxes.id):
                track_id = int(raw_boxes.id[i])
                label = f"ID{track_id} {conf:.2%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        self._draw_info_panel(annotated, detection_result)
        return annotated

    def _draw_info_panel(self, frame, result):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (380, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 40
        cv2.putText(frame, f"COUNT: {result['stable_count']}",
                    (15, y), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)
        y += 45

        cv2.putText(frame, f"Raw: {result['raw_count']} | Filtered: {result['filtered_count']}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y += 28

        if result['avg_confidence'] > 0:
            color = (0, 255, 0) if result['avg_confidence'] > 0.75 else (0, 165, 255)
            cv2.putText(frame, f"Accuracy: {result['avg_confidence']:.1%}",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Accuracy: N/A",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        y += 30

        stability = result['stability']
        if stability >= 85:
            status, color = "LOCKED", (0, 255, 0)
        elif stability >= 70:
            status, color = "STABLE", (0, 200, 255)
        else:
            status, color = "TRACKING", (0, 100, 255)

        cv2.putText(frame, f"Status: {status} ({stability:.0f}%)",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28

        if result['high_conf_ratio'] > 0:
            cv2.putText(frame, f"High Conf: {result['high_conf_ratio']:.0f}%",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    def reset(self):
        """Clear tracking history and metrics."""
        self.detection_history.clear()
        self.confidence_history.clear()
        self.current_stable_count = 0
        self.total_detections = 0
        self.high_conf_detections = 0
        print("Detector reset")


if __name__ == "__main__":
    detector = WasteDetector()
    print("Detector initialized successfully")
