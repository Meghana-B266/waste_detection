"""
Water-Specific Enhancement Module
Maximizes plastic detection on water surfaces
"""

import cv2
import numpy as np
from typing import Tuple, List

class WaterEnhancer:
    """Advanced enhancement for plastic detection on water"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.sharpen_kernel = np.array([[-1,-1,-1],
                                        [-1, 9,-1],
                                        [-1,-1,-1]]) * 0.5
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all enhancement techniques
        """
        # Step 1: CLAHE (Contrast)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 2: Sharpening (Edge enhancement)
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        
        # Step 3: Color boost (Makes plastic labels stand out)
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)  # Saturation
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # Value
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 4: Denoise slightly
        final = cv2.bilateralFilter(final, 5, 50, 50)
        
        return final
    
    def enhance_for_dark_water(self, frame: np.ndarray) -> np.ndarray:
        """
        Special enhancement for dark/murky water
        """
        # Gamma correction for dark scenes
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        brightened = cv2.LUT(frame, table)
        
        # Then apply standard enhancement
        return self.enhance_frame(brightened)
    
    def auto_enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Automatically choose enhancement based on brightness
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 80:
            # Dark scene (murky water, shadow)
            return self.enhance_for_dark_water(frame)
        else:
            # Normal scene
            return self.enhance_frame(frame)


class MultiScaleDetector:
    """Run detection at multiple scales for better accuracy"""
    
    def __init__(self, model):
        self.model = model
    
    def detect(self, frame: np.ndarray, base_conf: float = 0.35) -> Tuple:
        """
        Multi-scale detection with confidence blending
        """
        # Scale 1: 640x640 (fast, good for medium objects)
        results_640 = self.model.track(
            frame,
            conf=base_conf,
            imgsz=640,
            persist=True,
            verbose=False
        )
        
        # Scale 2: 1280x1280 (slower, catches small/distant objects)
        results_1280 = self.model.track(
            frame,
            conf=base_conf - 0.05,  # Slightly lower for high-res
            imgsz=1280,
            persist=True,
            verbose=False
        )
        
        # Return result with more detections
        boxes_640 = len(results_640[0].boxes)
        boxes_1280 = len(results_1280[0].boxes)
        
        if boxes_1280 > boxes_640 * 1.2:  # 20% more detections
            return results_1280
        else:
            return results_640


class AdaptiveThreshold:
    """Dynamically adjust confidence threshold"""
    
    def __init__(self, base_threshold: float = 0.35):
        self.base_threshold = base_threshold
        self.min_threshold = 0.25
        self.max_threshold = 0.60
    
    def calculate(self, confidence_history: list) -> float:
        """
        Lower threshold if we're consistently detecting
        (model is probably right, just less confident)
        """
        if len(confidence_history) < 15:
            return self.base_threshold
        
        # Get recent confidences (last 15 frames)
        recent = [c for c in list(confidence_history)[-15:] if c > 0]
        
        if len(recent) < 5:
            # Not detecting much - keep base threshold
            return self.base_threshold
        
        avg_conf = np.mean(recent)
        detection_rate = len(recent) / 15
        
        # Adjust based on detection patterns
        if detection_rate > 0.7 and avg_conf < 0.65:
            # Detecting frequently but low confidence (probably water)
            return max(self.min_threshold, self.base_threshold - 0.10)
        elif detection_rate < 0.3:
            # Not detecting much - be more aggressive
            return max(self.min_threshold, self.base_threshold - 0.05)
        else:
            # Normal detection
            return self.base_threshold


class SmartBoxFilter:
    """Intelligent box filtering with relaxed rules for water"""
    
    def __init__(self, min_area: int = 800):
        self.min_area = min_area
        self.min_conf = 0.30  # Very low for maximum detection
    
    def filter(self, boxes, frame_shape) -> Tuple[List, List, List]:
        """
        Filter with RELAXED rules for water scenarios
        """
        if boxes is None or len(boxes) == 0:
            return [], [], []
        
        h, w = frame_shape[:2]
        valid_boxes = []
        valid_confidences = []
        valid_ids = []
        
        for idx, box in enumerate(boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            box_w = x2 - x1
            box_h = y2 - y1
            area = box_w * box_h
            
            # Basic validation
            if conf < self.min_conf:
                continue
            
            if area < self.min_area:
                continue
            
            # Relaxed aspect ratio (accept more shapes)
            aspect_ratio = box_w / box_h if box_h > 0 else 0
            if aspect_ratio < 0.05 or aspect_ratio > 20:  # Very relaxed
                continue
            
            # Accept edge detections (plastic at water edge is common)
            # No strict boundary checking
            
            # All checks passed
            valid_boxes.append(box)
            valid_confidences.append(conf)
            
            if boxes.id is not None and idx < len(boxes.id):
                valid_ids.append(int(boxes.id[idx]))
            else:
                valid_ids.append(None)
        
        return valid_boxes, valid_confidences, valid_ids


