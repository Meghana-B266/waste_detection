#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ========================================
# PRODUCTION-GRADE WASTE DETECTION SYSTEM
# WITH EMAIL ALERTS - COMPLETE VERSION
# ========================================

# ============ CELL 1: Setup ============
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import time
from collections import deque, Counter
from dotenv import load_dotenv

# Set working directory
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# Add backend to path for imports
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'backend'))

# Load environment
load_dotenv(override=True)

print("="*70)
print("PRODUCTION WASTE DETECTION SYSTEM v2.0 + EMAIL ALERTS")
print("="*70)
print(f"Directory: {os.getcwd()}")
print(f"Model: {Path('models/best.pt').exists()}")



# In[2]:


# ============ CELL 2: Import Backend & Load Model ============
from ultralytics import YOLO
from backend.database import DatabaseManager, init_db
from backend.alerts import AlertManager

MODEL_PATH = 'models/best.pt'
model_abs_path = Path(MODEL_PATH).absolute()

if not model_abs_path.exists():
    raise FileNotFoundError(f"Model not found: {model_abs_path}")

print(f"\nLoading model: {model_abs_path.name}")
model = YOLO(str(model_abs_path))

# Warm up
_ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
print("✓ Model loaded and ready")

# Initialize database and alerts
print("\n Initializing database...")
init_db()
db = DatabaseManager()
print("✓ Database ready")

print("\n Initializing alert system...")
alert_mgr = AlertManager(db_manager=db)


# In[3]:


# ============ CELL 2B: Import Water Enhancement Modules ============

print("\n Loading water enhancement modules...")

from backend.water_enhancement import (
    WaterEnhancer, 
    MultiScaleDetector, 
    AdaptiveThreshold, 
    SmartBoxFilter
)

# Get configuration from .env
USE_WATER_ENHANCEMENT = os.getenv('USE_WATER_ENHANCEMENT', 'true').lower() == 'true'
USE_MULTISCALE = os.getenv('USE_MULTISCALE', 'true').lower() == 'true'
USE_ADAPTIVE_THRESHOLD = os.getenv('USE_ADAPTIVE_THRESHOLD', 'true').lower() == 'true'
AUTO_BRIGHTNESS_ADJUST = True

# Initialize enhancement tools
water_enhancer = WaterEnhancer()
adaptive_threshold = AdaptiveThreshold(base_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.35')))
smart_filter = SmartBoxFilter(min_area=int(os.getenv('MIN_BOX_AREA', '800')))

if USE_MULTISCALE:
    multiscale_detector = MultiScaleDetector(model)

print("✓ Water enhancement modules loaded")
print(f"  Enhancement: {'ON' if USE_WATER_ENHANCEMENT else 'OFF'}")
print(f"  Multi-Scale: {'ON' if USE_MULTISCALE else 'OFF'}")
print(f"  Adaptive Threshold: {'ON' if USE_ADAPTIVE_THRESHOLD else 'OFF'}")


# In[4]:


# ============ CELL 3: AGGRESSIVE DETECTION CONFIGURATION ============

# Camera
PHONE_IP = os.getenv('PHONE_IP', '').strip()
if not PHONE_IP:
    PHONE_IP = input("\n📱 Enter Phone IP Address: ").strip()
os.environ['PHONE_IP'] = PHONE_IP
CAMERA_NAME = "Mobile Detection Camera"

# Detection (ULTRA-AGGRESSIVE FOR MAXIMUM DETECTION)
CONFIDENCE_THRESHOLD = 0.20    # VERY LOW for maximum catch
IOU_THRESHOLD = 0.35           # Lower for more overlaps
MIN_BOX_AREA = 500             # Catch even smaller objects
IMAGE_SIZE = 1280              # High resolution

# Tracking (RELAXED FOR BETTER ACCEPTANCE)
SMOOTHING_WINDOW = 30          
STABILITY_FRAMES = 10          # Lower = faster lock
STABILITY_THRESHOLD = 0.65     # Lower = easier to lock

# Enhancement (ALL ENABLED)
USE_WATER_ENHANCEMENT = True
USE_MULTISCALE = True
USE_ADAPTIVE_THRESHOLD = True
AUTO_BRIGHTNESS_ADJUST = True

# Alerts
ALERT_THRESHOLD = 1
REPORT_COOLDOWN = 2.0

COLORS = {
    'high': (0, 255, 0),      # Green
    'medium': (0, 255, 255),   # Yellow
    'low': (0, 165, 255)       # Orange
}

SAVE_DETECTIONS = True
DETECTION_FOLDER = PROJECT_DIR / 'detections'
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Get auto-detected location
location_info = alert_mgr.get_current_location_info()

print("\n" + "="*80)
print(" ULTRA-AGGRESSIVE DETECTION MODE")
print("="*80)
print(f"Phone IP: {PHONE_IP}")
print(f"GPS Source: {location_info.get('source', 'unknown')}")
if location_info.get('source') == 'ip_geolocation':
    print(f"Auto-Detected Area: {location_info.get('area', 'Unknown')}")
    print(f"City: {location_info.get('city', 'Unknown')}")
else:
    print(f"Location: {location_info.get('location', 'Unknown')}")
print(f"GPS: {location_info['latitude']:.6f}, {location_info['longitude']:.6f}")
print(f"Base Confidence: {CONFIDENCE_THRESHOLD} (ULTRA-AGGRESSIVE)")
print(f"Min Box Area: {MIN_BOX_AREA} (CATCHES SMALL OBJECTS)")
print(f"Stability: {STABILITY_THRESHOLD*100}% over {STABILITY_FRAMES} frames (FAST LOCK)")
print(f"Alert Threshold: {ALERT_THRESHOLD} items")
print(f"Water Enhancement: {'ON' if USE_WATER_ENHANCEMENT else 'OFF'}")
print(f"Multi-Scale: {'ON' if USE_MULTISCALE else 'OFF'}")
print(f"Adaptive Threshold: {'ON' if USE_ADAPTIVE_THRESHOLD else 'OFF'}")
print(f"Email Alerts: {'ON' if alert_mgr.email_enabled else 'OFF'}")
print("="*80)


# In[ ]:


# ============ CELL 4: ADVANCED HELPER FUNCTIONS ============

def calculate_box_area(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    return (x2 - x1) * (y2 - y1)

def advanced_filter_detections(boxes, frame_shape):
    """Enhanced filtering with multiple criteria"""
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
        aspect_ratio = box_w / box_h if box_h > 0 else 0
        
        valid = True
        
        if conf < CONFIDENCE_THRESHOLD:
            valid = False
        if area < MIN_BOX_AREA:
            valid = False
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            valid = False
        
        margin = 10
        if (x1 < margin or y1 < margin or x2 > w-margin or y2 > h-margin):
            if area < MIN_BOX_AREA * 2:
                valid = False
        
        image_area = w * h
        relative_size = area / image_area
        if relative_size < 0.001 or relative_size > 0.8:
            valid = False
        
        if valid:
            valid_boxes.append(box)
            valid_confidences.append(conf)
            if boxes.id is not None and idx < len(boxes.id):
                valid_ids.append(int(boxes.id[idx]))
            else:
                valid_ids.append(None)
    
    return valid_boxes, valid_confidences, valid_ids

def get_stable_count_advanced(history, min_frames, threshold=STABILITY_THRESHOLD):
    if len(history) < min_frames:
        return None
    
    recent = list(history)[-min_frames:]
    count_freq = Counter(recent)
    
    if not count_freq:
        return 0
    
    most_common_count, frequency = count_freq.most_common(1)[0]
    stability_ratio = frequency / min_frames
    
    if stability_ratio >= threshold:
        return most_common_count
    
    return None

def calculate_detection_quality(confidences):
    if not confidences:
        return 0.0
    
    avg_conf = np.mean(confidences)
    std_conf = np.std(confidences) if len(confidences) > 1 else 0
    consistency_score = 1 - min(std_conf, 0.3) / 0.3
    high_conf_ratio = sum(1 for c in confidences if c > 0.85) / len(confidences)
    
    quality_score = (
        avg_conf * 0.5 +
        consistency_score * 0.3 +
        high_conf_ratio * 0.2
    )
    
    return quality_score

def save_detection_image(frame, count, confidence):
    if not SAVE_DETECTIONS:
        return None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = DETECTION_FOLDER / f"waste_{count}items_{confidence:.0%}_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)

print("✓ Advanced helper functions loaded")


# In[ ]:


# ============ CELL 5: Connect Camera ============
camera_url = f"http://{PHONE_IP}:8080/video"
print(f"\nConnecting to: {camera_url}")

cap = cv2.VideoCapture(camera_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(" Camera connection failed!")
    raise ConnectionError("Cannot connect to camera")

print("✓ Camera connected")


# In[21]:


# ============================================================================
# CELL 6: MAIN DETECTION LOOP - ENHANCED WITH WATER OPTIMIZATION
# ============================================================================

import time
from collections import deque, Counter
from datetime import datetime
from pathlib import Path

# Initialize tracking variables
detection_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_history = deque(maxlen=SMOOTHING_WINDOW)
quality_history = deque(maxlen=SMOOTHING_WINDOW)
current_stable_count = 0
last_report_time = 0
frame_count = 0
total_detections = 0
high_conf_detections = 0

# FPS tracking
fps_start = time.time()
fps_count = 0
current_fps = 0

print("\n" + "="*80)
print("DETECTION STARTED - ENHANCED MODE WITH WATER OPTIMIZATION")
print("="*80)
print("Controls: 'q'=quit | 's'=save screenshot | 'r'=reset | 'p'=pause")
print("="*80 + "\n")

paused = False

try:
    while True:
        if not paused:
            # ===== FRAME CAPTURE =====
            ret, frame = cap.read()
            if not ret:
                print(" Connection lost. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(camera_url)
                continue
            
            frame_count += 1
            fps_count += 1
            
            # Calculate FPS
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0
            
            # ===== STEP 1: WATER ENHANCEMENT (NEW) =====
            if USE_WATER_ENHANCEMENT:
                if AUTO_BRIGHTNESS_ADJUST:
                    enhanced_frame = water_enhancer.auto_enhance(frame)
                else:
                    enhanced_frame = water_enhancer.enhance_frame(frame)
            else:
                enhanced_frame = frame
            
            # ===== STEP 2: ADAPTIVE CONFIDENCE (NEW) =====
            if USE_ADAPTIVE_THRESHOLD:
                current_conf = adaptive_threshold.calculate(confidence_history)
            else:
                current_conf = CONFIDENCE_THRESHOLD
            
            # ===== STEP 3: MULTI-SCALE DETECTION (NEW) =====
            try:
                if USE_MULTISCALE:
                    results = multiscale_detector.detect(enhanced_frame, current_conf)
                else:
                    results = model.track(
                        enhanced_frame,
                        conf=current_conf,
                        iou=IOU_THRESHOLD,
                        imgsz=IMAGE_SIZE,
                        verbose=False,
                        persist=True,
                        tracker="bytetrack.yaml",
                        max_det=20
                    )
            except Exception as e:
                print(f" Detection error: {e}")
                continue
            
            # ===== STEP 4: SMART FILTERING (NEW) =====
            raw_boxes = results[0].boxes
            filtered_boxes, confidences, track_ids = smart_filter.filter(
                raw_boxes, enhanced_frame.shape
            )
            current_count = len(filtered_boxes)
            
            # ===== UPDATE METRICS =====
            if current_count > 0:
                avg_conf = float(np.mean(confidences))
                max_conf = float(np.max(confidences))
                quality = calculate_detection_quality(confidences)
                confidence_history.append(avg_conf)
                quality_history.append(quality)
                
                for c in confidences:
                    total_detections += 1
                    if c > 0.85:
                        high_conf_detections += 1
            else:
                avg_conf = max_conf = quality = 0.0
                confidence_history.append(0)
                quality_history.append(0)
            
            detection_history.append(current_count)
            
            # Calculate averages
            avg_confidence = float(np.mean([c for c in confidence_history if c > 0])) if any(confidence_history) else 0
            avg_quality = float(np.mean([q for q in quality_history if q > 0])) if any(quality_history) else 0
            
            # ===== STABLE COUNT DETECTION =====
            stable_count = get_stable_count_advanced(detection_history, STABILITY_FRAMES, STABILITY_THRESHOLD)
            
            current_time = time.time()
            if stable_count is not None and stable_count != current_stable_count:
                if current_time - last_report_time >= REPORT_COOLDOWN:
                    current_stable_count = stable_count
                    last_report_time = current_time
                    
                    if stable_count > 0:
                        quality_str = f"{avg_quality*100:.1f}%"
                        print(f"\n DETECTED: {stable_count} plastic item(s)")
                        print(f"   Confidence: {avg_confidence:.1%} | Quality: {quality_str}")
                        
                        # ===== CREATE ANNOTATED IMAGE FOR SAVING =====
                        annotated_for_save = frame.copy()
                        for i, box in enumerate(filtered_boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = confidences[i]
                            
                            # Color based on confidence
                            color = COLORS['high'] if conf > 0.85 else COLORS['medium'] if conf > 0.70 else COLORS['low']
                            
                            # Draw box
                            cv2.rectangle(annotated_for_save, (x1, y1), (x2, y2), color, 3)
                            
                            # Label
                            label = f"Plastic {conf:.0%}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(annotated_for_save, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                            cv2.putText(annotated_for_save, label, (x1 + 5, y1 - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Save image
                        img_path = save_detection_image(annotated_for_save, stable_count, avg_confidence)
                        print(f"   📸 Saved: {Path(img_path).name}")
                        
                        # ===== DATABASE LOGGING =====
                        try:
                            loc_info = alert_mgr.get_current_location_info()
                            detection_id = db.save_detection({
                                'timestamp': datetime.now(),
                                'camera_name': CAMERA_NAME,
                                'location': loc_info['location'],
                                'latitude': loc_info['latitude'],
                                'longitude': loc_info['longitude'],
                                'waste_count': stable_count,
                                'confidence_avg': avg_confidence,
                                'confidence_max': max_conf,
                                'fps': current_fps,
                                'image_path': str(img_path)
                            })
                            print(f"    Database ID: {detection_id}")
                            
                            # ===== EMAIL ALERT =====
                            if alert_mgr.should_send_alert(stable_count):
                                print(f"    Sending email alert...")
                                try:
                                    success = alert_mgr.trigger_alert(
                                        detection_id, 
                                        stable_count, 
                                        avg_confidence, 
                                        str(img_path)
                                    )
                                    if not success:
                                        print(f"     Email sending failed")
                                except Exception as email_error:
                                    print(f"    Email error: {email_error}")
                            else:
                                print(f"    No email sent (threshold: {alert_mgr.alert_threshold})")
                            
                        except Exception as e:
                            print(f"     Error: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    else:
                        print("\n Clear - No plastic detected")
            
            # ============================================================================
            # VISUALIZATION - DRAW BOXES AND INFO PANEL
            # ============================================================================
            
            annotated = frame.copy()
            
            # Draw detection boxes
            for i, box in enumerate(filtered_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = confidences[i]
                track_id = track_ids[i]
                
                # Color based on confidence
                color = COLORS['high'] if conf > 0.85 else COLORS['medium'] if conf > 0.70 else COLORS['low']
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"Plastic {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw tracking ID (if available)
                if track_id is not None:
                    id_label = f"#{track_id}"
                    cv2.putText(annotated, id_label, (x1 + 5, y1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # ===== INFO PANEL (TOP LEFT) =====
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (450, 330), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            
            y = 40
            
            # Main count (large)
            cv2.putText(annotated, f"COUNT: {current_stable_count}", 
                       (15, y), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 3)
            y += 55
            
            # Detection counts
            cv2.putText(annotated, f"Raw: {len(raw_boxes)} | Filtered: {current_count}", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y += 32
            
            # Confidence
            if avg_confidence > 0:
                conf_color = (0, 255, 0) if avg_confidence > 0.80 else (0, 255, 255) if avg_confidence > 0.70 else (0, 165, 255)
                cv2.putText(annotated, f"Confidence: {avg_confidence:.1%}", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
            else:
                cv2.putText(annotated, "Confidence: N/A", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            y += 32
            
            # Quality
            if avg_quality > 0:
                quality_color = (0, 255, 0) if avg_quality > 0.85 else (0, 255, 255) if avg_quality > 0.70 else (0, 165, 255)
                cv2.putText(annotated, f"Quality: {avg_quality:.1%}", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
            else:
                cv2.putText(annotated, "Quality: N/A", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            y += 32
            
            # Adaptive confidence (NEW)
            if USE_ADAPTIVE_THRESHOLD:
                cv2.putText(annotated, f"Adaptive Conf: {current_conf:.2f}", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
                y += 30
            
            # Enhancement status (NEW)
            enh_color = (0, 255, 0) if USE_WATER_ENHANCEMENT else (100, 100, 100)
            cv2.putText(annotated, f"Enhancement: {'ON' if USE_WATER_ENHANCEMENT else 'OFF'}", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, enh_color, 2)
            y += 30
            
            # Multi-scale status (NEW)
            ms_color = (0, 255, 0) if USE_MULTISCALE else (100, 100, 100)
            cv2.putText(annotated, f"Multi-Scale: {'ON' if USE_MULTISCALE else 'OFF'}", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ms_color, 2)
            y += 30
            
            # Stability status
            if len(detection_history) >= STABILITY_FRAMES:
                recent = list(detection_history)[-STABILITY_FRAMES:]
                stability = (recent.count(current_stable_count) / STABILITY_FRAMES) * 100
                
                if stability >= 90:
                    status, status_color = "LOCKED", (0, 255, 0)
                elif stability >= 75:
                    status, status_color = "STABLE", (0, 200, 255)
                elif stability >= 60:
                    status, status_color = "TRACKING", (0, 165, 255)
                else:
                    status, status_color = "UNSTABLE", (0, 100, 255)
                
                cv2.putText(annotated, f"Status: {status} ({stability:.0f}%)", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            else:
                remaining = STABILITY_FRAMES - len(detection_history)
                cv2.putText(annotated, f"Calibrating: {remaining} frames", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            y += 30
            
            # FPS
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y += 28
            
            # Email counter
            email_color = (0, 255, 0) if alert_mgr.email_enabled else (100, 100, 100)
            cv2.putText(annotated, f"Emails: {alert_mgr.email_sent_today}/50", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, email_color, 2)
            
            # Display frame
            cv2.imshow('Waste Detection - Enhanced | Press Q to Quit', annotated)
        
        else:
            # Paused state
            paused_frame = frame.copy()
            cv2.putText(paused_frame, "PAUSED - Press 'p' to resume", 
                       (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow('Waste Detection - Enhanced | Press Q to Quit', paused_frame)
        
        # ===== KEYBOARD CONTROLS =====
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n Stopping detection...")
            break
        elif key == ord('s'):
            filename = save_detection_image(annotated, current_stable_count, avg_confidence)
            print(f"\n Screenshot saved: {filename}\n")
        elif key == ord('r'):
            detection_history.clear()
            confidence_history.clear()
            quality_history.clear()
            current_stable_count = 0
            total_detections = 0
            high_conf_detections = 0
            print("\n System reset - recalibrating...\n")
        elif key == ord('p'):
            paused = not paused
            print(f"\n{' PAUSED' if paused else ' RESUMED'}\n")

except KeyboardInterrupt:
    print("\n  Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # ============================================================================
    # FINAL STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("DETECTION SESSION SUMMARY")
    print("="*80)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Final Stable Count: {current_stable_count} plastic items")
    
    if avg_confidence > 0:
        print(f"Average Confidence: {avg_confidence:.1%}")
    
    if avg_quality > 0:
        print(f"Average Quality Score: {avg_quality:.1%}")
    
    if total_detections > 0:
        high_conf_ratio = (high_conf_detections / total_detections) * 100
        print(f"High Confidence Ratio: {high_conf_ratio:.1f}%")
        print(f"Total Detections Made: {total_detections}")
    
    # Peak count
    if len(detection_history) > 0:
        non_zero = [d for d in detection_history if d > 0]
        if non_zero:
            print(f"Peak Count: {max(non_zero)} items")
            print(f"Average Count (when detected): {np.mean(non_zero):.1f} items")
    
    if current_fps > 0:
        duration_minutes = frame_count / current_fps / 60
        print(f"Session Duration: {duration_minutes:.1f} minutes")
    
    print(f"Emails Sent: {alert_mgr.email_sent_today}")
    
    print("="*80)
    print("✓ Detection session ended")
    print("="*80)


# In[ ]:




