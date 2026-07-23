"""
Alert Management System with Email Notifications and Fixed GPS
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, date
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

load_dotenv()


class AlertManager:
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.email_enabled = os.getenv('ENABLE_EMAIL', 'false').lower() == 'true'
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.alert_email_to = os.getenv('ALERT_EMAIL_TO')
        self.use_phone_gps = os.getenv('USE_PHONE_GPS', 'false').lower() == 'true'
        self.phone_ip = os.getenv('PHONE_IP', '192.168.0.105')
        self.camera_lat = float(os.getenv('CAMERA_LATITUDE', '12.90111'))
        self.camera_lon = float(os.getenv('CAMERA_LONGITUDE', '77.51806'))
        self.camera_location = os.getenv('CAMERA_LOCATION', 'RNS Institute of Technology')
        self.geolocator = Nominatim(user_agent="waste_detection_v1")
        self.alert_threshold = int(os.getenv('ALERT_THRESHOLD', 5))
        self.email_sent_today = 0
        self.max_emails_per_day = 50
        self.last_gps_data = None
        self.cache_timestamp = 0
        self.cache_duration = 5
        print(f"Alert Manager initialized:")
        print(f"  Email: {'Enabled' if self.email_enabled else 'Disabled'}")
        print(f"  GPS Mode: Fixed Coordinates")
        print(f"  Location: {self.camera_location}")
    
    def get_phone_gps(self):
        return None
    
    def get_address_from_coords(self, latitude, longitude):
        try:
            location = self.geolocator.reverse(
                f"{latitude}, {longitude}",
                exactly_one=True,
                language='en',
                timeout=10
            )
            if location and location.address:
                return location.address
            else:
                return self.camera_location
        except:
            return self.camera_location
    
    def get_current_location_info(self):
        return {
            'latitude': self.camera_lat,
            'longitude': self.camera_lon,
            'location': self.camera_location,
            'source': 'fixed'
        }
    
    def should_send_alert(self, waste_count):
        if not self.email_enabled:
            return False
        if waste_count < self.alert_threshold:
            return False
        if self.email_sent_today >= self.max_emails_per_day:
            return False
        return True
    
    def send_email_alert(self, detection_id, waste_count, confidence, image_path=None):
        if not self.email_enabled:
            return False
        try:
            location_info = self.get_current_location_info()
            message = MIMEMultipart()
            message['From'] = self.email_from
            message['To'] = self.alert_email_to
            message['Subject'] = f"Waste Alert: {waste_count} items detected"
            body = f"""
Waste Detection Alert

DETECTION DETAILS
Detection ID: #{detection_id}
Waste Count: {waste_count} items
Confidence: {confidence:.1%}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LOCATION
Address: {location_info['location']}
Coordinates: {location_info['latitude']:.6f}, {location_info['longitude']:.6f}
Google Maps: https://www.google.com/maps?q={location_info['latitude']},{location_info['longitude']}

This is an automated alert from the Waste Detection System.
"""
            message.attach(MIMEText(body, 'plain'))
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        image = MIMEImage(img_data, name=os.path.basename(image_path))
                        message.attach(image)
                except:
                    pass
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                clean_password = self.email_password.replace(' ', '')
                server.login(self.email_from, clean_password)
                server.sendmail(self.email_from, self.alert_email_to, message.as_string())
            self.email_sent_today += 1
            print(f"   Email sent ({self.email_sent_today}/{self.max_emails_per_day})")
            return True
        except smtplib.SMTPAuthenticationError:
            print("   Email auth failed")
            return False
        except Exception as e:
            print(f"   Email error: {e}")
            return False
    
    def trigger_alert(self, detection_id, waste_count, confidence, image_path=None):
        if self.should_send_alert(waste_count):
            success = self.send_email_alert(detection_id, waste_count, confidence, image_path)
            if self.db:
                try:
                    # Logs the alert attempt (sent or failed) against the detection
                    self.db.log_alert(
                        detection_id=detection_id,
                        alert_type='email',
                        success=success,
                        recipient=self.alert_email_to,
                        message=f"{waste_count} items detected (confidence {confidence:.1%})"
                    )
                except Exception as e:
                    print(f"   Could not log alert to database: {e}")
            return success
        return False
    
    def reset_daily_counter(self):
        self.email_sent_today = 0
