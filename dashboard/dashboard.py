"""
Waste Detection Dashboard (Streamlit)

Run with:
    streamlit run dashboard/dashboard.py

Reads real detection data from the project's SQLite database via
backend.database.DatabaseManager, instead of hardcoded sample rows.
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Make sure "backend" is importable regardless of where streamlit is launched from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # so backend.database's relative sqlite path resolves correctly

from backend.database import DatabaseManager, init_db  # noqa: E402

st.set_page_config(page_title="Waste Detection Dashboard", layout="wide")

st.title("🌊 Waste Detection Dashboard")
st.markdown("Real-time monitoring and analytics for the plastic waste detection system")

# Make sure tables exist even on a fresh checkout
init_db()
db = DatabaseManager()


@st.cache_data(ttl=15)
def load_detections(limit=200):
    rows = db.get_recent_detections(limit=limit)
    if not rows:
        return pd.DataFrame(columns=[
            'id', 'timestamp', 'camera_name', 'location',
            'waste_count', 'confidence_avg', 'fps', 'image_path', 'alert_sent'
        ])
    return pd.DataFrame([{
        'id': r.id,
        'timestamp': r.timestamp,
        'camera_name': r.camera_name,
        'location': r.location,
        'waste_count': r.waste_count,
        'confidence_avg': r.confidence_avg,
        'fps': r.fps,
        'image_path': r.image_path,
        'alert_sent': r.alert_sent,
    } for r in rows])


df = load_detections()

if df.empty:
    st.info(
        "No detections logged yet. Run `python run_complete_system.py` to start "
        "detecting — new detections will show up here automatically."
    )
    st.stop()

df = df.sort_values('timestamp')

# ==================== TOP METRICS ====================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Detections", len(df))
col2.metric("Total Items Seen", int(df['waste_count'].sum()))
col3.metric("Avg Confidence", f"{df['confidence_avg'].mean():.1%}")
col4.metric("Alerts Sent", int(df['alert_sent'].sum()))

# ==================== CHARTS ====================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Waste Detected Over Time")
    fig = px.line(df, x='timestamp', y='waste_count', markers=True,
                  labels={'timestamp': 'Time', 'waste_count': 'Items Detected'})
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Confidence Level Over Time")
    fig2 = px.bar(df, x='timestamp', y='confidence_avg',
                  labels={'timestamp': 'Time', 'confidence_avg': 'Avg Confidence'})
    st.plotly_chart(fig2, use_container_width=True)

# ==================== TABLE ====================
st.subheader("Recent Detections")
st.dataframe(
    df[['timestamp', 'camera_name', 'location', 'waste_count', 'confidence_avg', 'alert_sent']]
    .sort_values('timestamp', ascending=False),
    use_container_width=True
)

# ==================== DETECTION IMAGES ====================
st.subheader("Detected Images")
images_per_row = 3
recent_with_images = df[df['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
recent_with_images = recent_with_images.sort_values('timestamp', ascending=False).head(9)

if recent_with_images.empty:
    st.caption("No saved detection images found on disk yet.")
else:
    rows = recent_with_images.to_dict('records')
    for i in range(0, len(rows), images_per_row):
        cols = st.columns(images_per_row)
        for col, row in zip(cols, rows[i:i + images_per_row]):
            col.image(row['image_path'], caption=f"{row['waste_count']} items @ {row['confidence_avg']:.0%}",
                       use_container_width=True)

# ==================== SIDEBAR ====================
st.sidebar.title("Info")
st.sidebar.info(f"Showing latest {len(df)} detections")
st.sidebar.dataframe(df.tail(10)[['timestamp', 'waste_count']])
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit — refreshes automatically every 15s.")
