"""
Database Module - SQLAlchemy ORM Models
Compatible with SQLAlchemy 2.0
"""

import os
import warnings
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Boolean, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=DeprecationWarning)

load_dotenv()

# Database file always lives in <project_root>/data/waste_detection.db
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./data/waste_detection.db')
os.makedirs('data', exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==================== MODELS ====================

class Detection(Base):
    """Main detection events table"""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_name = Column(String(100))
    location = Column(String(200))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    waste_count = Column(Integer)
    confidence_avg = Column(Float)
    confidence_max = Column(Float)
    fps = Column(Float)
    image_path = Column(String(500))
    alert_sent = Column(Boolean, default=False)

    alerts = relationship("Alert", back_populates="detection")


class Alert(Base):
    """Alerts sent for detections"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    detection_id = Column(Integer, ForeignKey('detections.id'))
    alert_type = Column(String(50))   # email, sms, whatsapp
    message = Column(Text)
    status = Column(String(50))       # sent, failed, pending
    recipient = Column(String(200))

    detection = relationship("Detection", back_populates="alerts")


class DailySummary(Base):
    """Daily aggregated statistics"""
    __tablename__ = "daily_summary"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10), unique=True, index=True)
    camera_name = Column(String(100))
    total_detections = Column(Integer, default=0)
    max_count = Column(Integer, default=0)
    avg_count = Column(Float, default=0.0)
    total_alerts = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)


class SystemMetrics(Base):
    """Real-time system performance metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_name = Column(String(100))
    current_count = Column(Integer)
    avg_confidence = Column(Float)
    fps = Column(Float)
    high_conf_ratio = Column(Float)
    system_status = Column(String(50))  # active, error, stopped


class CameraConfig(Base):
    """Camera configuration and status"""
    __tablename__ = "camera_config"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String(100), unique=True)
    camera_ip = Column(String(50))
    location = Column(String(200))
    latitude = Column(Float)
    longitude = Column(Float)
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, default=datetime.utcnow)


# ==================== DATABASE FUNCTIONS ====================

def init_db():
    """Initialize database and create all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized")


def get_db():
    """Dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== CRUD OPERATIONS ====================

class DatabaseManager:
    """Database operations manager"""

    def __init__(self):
        self.SessionLocal = SessionLocal

    def save_detection(self, data: dict):
        """Save detection to database. Returns the new row's id."""
        db = self.SessionLocal()
        try:
            detection = Detection(**data)
            db.add(detection)
            db.commit()
            db.refresh(detection)
            return detection.id
        except Exception as e:
            db.rollback()
            print(f"Error saving detection: {e}")
            return None
        finally:
            db.close()

    def save_alert(self, data: dict):
        """Save alert to database and mark the related detection as alerted."""
        db = self.SessionLocal()
        try:
            alert = Alert(**data)
            db.add(alert)
            db.commit()
            db.refresh(alert)

            if data.get('detection_id'):
                detection = db.query(Detection).filter(
                    Detection.id == data['detection_id']
                ).first()
                if detection:
                    detection.alert_sent = True
                    db.commit()

            return alert.id
        except Exception as e:
            db.rollback()
            print(f"Error saving alert: {e}")
            return None
        finally:
            db.close()

    # Convenience wrapper used by AlertManager
    def log_alert(self, detection_id, alert_type, success, recipient=None, message=None):
        return self.save_alert({
            'detection_id': detection_id,
            'alert_type': alert_type,
            'message': message or '',
            'status': 'sent' if success else 'failed',
            'recipient': recipient or ''
        })

    def save_metrics(self, data: dict):
        """Save system metrics"""
        db = self.SessionLocal()
        try:
            metrics = SystemMetrics(**data)
            db.add(metrics)
            db.commit()
            return metrics.id
        except Exception as e:
            db.rollback()
            print(f"Error saving metrics: {e}")
            return None
        finally:
            db.close()

    def update_daily_summary(self, date: str, camera_name: str, waste_count: int, confidence: float):
        """Update or create daily summary"""
        db = self.SessionLocal()
        try:
            summary = db.query(DailySummary).filter(DailySummary.date == date).first()

            if summary:
                total = summary.total_detections
                summary.total_detections = total + 1
                summary.avg_count = (summary.avg_count * total + waste_count) / (total + 1)
                summary.avg_confidence = (summary.avg_confidence * total + confidence) / (total + 1)
                summary.max_count = max(summary.max_count, waste_count)
            else:
                summary = DailySummary(
                    date=date,
                    camera_name=camera_name,
                    total_detections=1,
                    max_count=waste_count,
                    avg_count=float(waste_count),
                    avg_confidence=confidence
                )
                db.add(summary)

            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Error updating summary: {e}")
            return False
        finally:
            db.close()

    def get_today_stats(self, camera_name: str):
        """Get today's statistics for a given camera"""
        db = self.SessionLocal()
        try:
            detections = db.query(Detection).filter(
                Detection.timestamp >= datetime.now().date(),
                Detection.camera_name == camera_name
            ).all()

            if not detections:
                return {
                    'total_detections': 0,
                    'total_waste': 0,
                    'max_count': 0,
                    'avg_confidence': 0,
                    'total_alerts': 0
                }

            return {
                'total_detections': len(detections),
                'total_waste': sum(d.waste_count for d in detections),
                'max_count': max(d.waste_count for d in detections),
                'avg_confidence': sum(d.confidence_avg for d in detections) / len(detections),
                'total_alerts': sum(1 for d in detections if d.alert_sent)
            }
        finally:
            db.close()

    def get_recent_detections(self, limit: int = 50):
        """Get most recent detections"""
        db = self.SessionLocal()
        try:
            return db.query(Detection).order_by(
                Detection.timestamp.desc()
            ).limit(limit).all()
        finally:
            db.close()

    def get_hourly_stats(self, hours: int = 24):
        """Get hourly aggregated statistics for the last N hours"""
        db = self.SessionLocal()
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            detections = db.query(Detection).filter(
                Detection.timestamp >= start_time
            ).all()

            hourly_data = {}
            for det in detections:
                hour = det.timestamp.strftime('%Y-%m-%d %H:00')
                if hour not in hourly_data:
                    hourly_data[hour] = {'count': 0, 'waste': 0}
                hourly_data[hour]['count'] += 1
                hourly_data[hour]['waste'] += det.waste_count

            return hourly_data
        finally:
            db.close()


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully")
