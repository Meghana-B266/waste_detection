"""
Standalone email test — checks your .env email settings work BEFORE relying
on live camera detection to trigger one. Sends one real test email using a
fake detection (id=0, no image), so you can isolate "is email configured
correctly" from "is detection working."

Run with:
    python test_email.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

from backend.alerts import AlertManager  # noqa: E402
from backend.database import DatabaseManager, init_db  # noqa: E402


def main():
    print("=" * 50)
    print("EMAIL CONFIGURATION TEST")
    print("=" * 50)

    init_db()
    db = DatabaseManager()
    alert_mgr = AlertManager(db_manager=db)

    if not alert_mgr.email_enabled:
        print("\n❌ ENABLE_EMAIL is not 'true' in your .env file.")
        print("   Set ENABLE_EMAIL=true and fill in the email fields, then rerun this.")
        return

    missing = [name for name, val in [
        ('EMAIL_FROM', alert_mgr.email_from),
        ('EMAIL_PASSWORD', alert_mgr.email_password),
        ('ALERT_EMAIL_TO', alert_mgr.alert_email_to),
    ] if not val]

    if missing:
        print(f"\n❌ Missing required .env values: {', '.join(missing)}")
        return

    print(f"\nSending a test alert email...")
    print(f"  From:   {alert_mgr.email_from}")
    print(f"  To:     {alert_mgr.alert_email_to}")
    print(f"  Server: {alert_mgr.smtp_server}:{alert_mgr.smtp_port}")

    success = alert_mgr.send_email_alert(
        detection_id=0,
        waste_count=3,
        confidence=0.87,
        image_path=None
    )

    print()
    if success:
        print("✅ Email sent successfully! Check your inbox (and spam folder).")
    else:
        print("❌ Email failed to send. Common causes:")
        print("   - Using your normal Gmail password instead of an App Password")
        print("     (generate one at https://myaccount.google.com/apppasswords)")
        print("   - 2-Step Verification not enabled on the Gmail account")
        print("     (required before App Passwords can be created)")
        print("   - Typo in EMAIL_FROM, EMAIL_PASSWORD, or ALERT_EMAIL_TO in .env")
        print("   - Firewall/antivirus blocking outbound port 587")


if __name__ == "__main__":
    main()
