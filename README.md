# 🌊 Floating Plastic Waste Detection System

AI-powered system that watches a live camera feed (an Android phone acting as
an IP webcam), detects floating plastic waste with a custom-trained YOLO
model, tracks it over time to avoid false-positive flicker, logs every
detection to a database, sends email alerts, and shows everything on a live
Streamlit dashboard.

This README documents the **cleaned-up** version of the project. See
["What was fixed"](#what-was-fixed-during-cleanup) at the bottom for exactly
what changed from the version you uploaded.

## 📂 Project Structure

```
waste-detection/
├── .env.example              # Template for your local config — copy to .env
├── .gitignore
├── requirements.txt
├── run_complete_system.py    # LOCAL entry point — opens an OpenCV window (single machine only)
├── run_web_server.py         # WEB entry point — serves the dashboard to any device on your network
├── backend/
│   ├── __init__.py
│   ├── database.py            # SQLAlchemy models + DatabaseManager
│   ├── detector.py             # Reusable WasteDetector class (YOLO + tracking)
│   ├── alerts.py                # AlertManager — email alerts + location info
│   ├── water_enhancement.py    # Image enhancement tuned for water surfaces
│   ├── service.py               # Background detection loop (used by the web server)
│   └── api.py                    # FastAPI app — REST endpoints + live video stream
├── frontend/
│   ├── index.html              # The web dashboard page
│   ├── style.css
│   └── app.js
├── dashboard/
│   └── dashboard.py            # Streamlit dashboard (alternative, history-only view)
├── models/
│   └── best.pt                  # Your trained YOLO weights
├── data/
│   └── waste_detection.db      # SQLite database (53 existing detections kept)
├── detections/                  # Saved annotated detection images land here
├── docs/
│   ├── architecture.dot         # System diagram source (Graphviz)
│   └── architecture.png
└── notebooks/                    # Original exploratory Jupyter notebooks
```

**Which entry point should you use?**
- `run_web_server.py` — the "public product" version. One server, viewable from any browser on your network (including phones), no local display required, no separate OpenCV window. **Use this one going forward.**
- `run_complete_system.py` — the original local-only version with a popup OpenCV window. Still works, kept for quick single-machine testing.
- `dashboard/dashboard.py` — a separate Streamlit view of just the historical data/charts (no live video). You don't need this if you're using the web server, since its dashboard already includes history — but it's there if you prefer it.



## 🚀 Setup

### 1. Clone / open the project and create a virtual environment

```bash
python -m venv venv

# Activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate            # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then open `.env` and fill in at least:
- `PHONE_IP` — the IP address shown by the **IP Webcam** app on your Android phone
- Email settings (`ENABLE_EMAIL`, `EMAIL_FROM`, `EMAIL_PASSWORD`, `ALERT_EMAIL_TO`) — only needed if you want email alerts

Every setting is documented with a comment in `.env.example`.

### 4. Set up your camera phone

1. Install the **IP Webcam** app (Android) from the Play Store.
2. Open it, scroll down, tap **Start server**.
3. Note the IP address it shows (e.g. `http://192.168.0.105:8080`).
4. Make sure your phone and computer are on the **same Wi-Fi network**.
5. Put that IP (without `http://` or `:8080`) into `PHONE_IP` in `.env`.

### 5. Verify the model and database are in place

```bash
ls models/best.pt        # your trained weights
ls data/waste_detection.db   # existing detection history (53 records)
```

Both are already included in this folder — no extra download needed.

## ▶️ Running the system (web version — recommended)

This is the "public product" version: one server, viewable from any browser on your network.

```bash
python run_web_server.py
```

You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
[service] Loading model: models/best.pt
[service] Model and services ready
```

Then open **http://localhost:8000** in your browser. To view it from your phone or another computer on the same Wi-Fi, find this computer's LAN IP and use that instead:

```bash
# Windows
ipconfig
# look for "IPv4 Address" e.g. 192.168.0.42

# Mac/Linux
ifconfig | grep "inet "
```

Then visit `http://192.168.0.42:8000` (using your actual IP) from any device on the same network.

**Using the dashboard:**
1. Type your phone's IP Webcam address into the "Phone IP" field (or leave blank if `PHONE_IP` is already set in `.env`).
2. Click **Start Detection** — the live annotated video feed appears.
3. Stats update automatically every 2 seconds; detection history updates every 5 seconds.
4. Click **Stop Detection** to release the camera.

To stop the server itself, go back to the terminal and press `Ctrl+C`.

### Deploying beyond your local network

The setup above works great on a home/office Wi-Fi. If you want it reachable from anywhere on the internet (e.g. hosted for a municipal authority to check remotely), you have two common options:
- **Simplest:** deploy to a cloud VM (e.g. AWS EC2, DigitalOcean) with a GPU if you want real-time speed, and put it behind a reverse proxy (nginx) with HTTPS (e.g. via Let's Encrypt/Certbot).
- **Quick tunnel for testing:** tools like `ngrok` or `cloudflared` can expose your local server temporarily without any cloud setup — good for demos, not for production.

Either way, you'd also want to add authentication (the current version has none — anyone who can reach the URL can start/stop detection), since right now it's designed for a trusted local network. Ask if you'd like help adding a login step.

## ▶️ Running the system (local version — original)

If you just want the original single-machine version with a popup OpenCV window instead:

```bash
python run_complete_system.py
```

- If `PHONE_IP` is set in `.env`, it connects automatically. Otherwise it will ask you to type the IP.
- A window opens showing the camera feed with detection boxes, confidence, and a live info panel.
- **Keyboard controls** while the window is focused:
  - `q` — quit
  - `s` — save a screenshot
  - `r` — reset tracking/calibration
  - `p` — pause/resume
- Every stable detection gets saved to `detections/`, logged to `data/waste_detection.db`, and (if enabled) triggers an email alert.

### Run the dashboard

In a **separate terminal** (you can run this at the same time as detection):

```bash
streamlit run dashboard/dashboard.py
```

This opens in your browser automatically (usually `http://localhost:8501`). It reads live from `data/waste_detection.db`, so anything `run_complete_system.py` detects will show up here (refreshes every ~15s).

## 🧠 How everything fits together

```
   Phone camera (IP Webcam app)
              │  MJPEG stream over Wi-Fi
              ▼
   backend/service.py  (background thread, started by clicking "Start Detection")
              │
              ├─► backend/water_enhancement.py   → brightens/sharpens frame for water conditions
              ├─► YOLO model (models/best.pt)     → detects plastic, ByteTrack tracks it across frames
              ├─► temporal smoothing/stability     → waits for a consistent count before "confirming" a detection
              ├─► backend/database.py              → saves each confirmed detection to SQLite
              └─► backend/alerts.py                 → emails you if count ≥ ALERT_THRESHOLD

   backend/api.py (FastAPI)
              ├─► GET  /api/video_feed        → streams the live annotated frame from service.py
              ├─► GET  /api/stats              → live count/confidence/fps/status
              ├─► GET  /api/detections         → history from data/waste_detection.db
              ├─► POST /api/start / /api/stop  → controls the detection thread
              └─► serves frontend/ (index.html, style.css, app.js) at "/"

   Your browser (or anyone's, on the same network)
              └─► http://<server-ip>:8000  ──polls/streams from──►  backend/api.py
```

- **`backend/service.py`** — the detection loop, refactored out of `run_complete_system.py` to run headlessly (no `cv2.imshow`) in a background thread, so a server with no display can run it. Holds the "latest frame" and "latest stats" that the API reads.
- **`backend/api.py`** — the FastAPI app. Thin — it mostly just exposes what `service.py` already tracks, plus serves the `frontend/` folder as static files.
- **`frontend/`** — plain HTML/CSS/JS (no React, no build step). `app.js` polls `/api/stats` every 2s and `/api/detections` every 5s, and points an `<img>` tag at `/api/video_feed` for the live stream.
- **`backend/detector.py`** contains a clean, reusable `WasteDetector` class with similar detection logic, meant for use from other scripts. `service.py` (like `run_complete_system.py` before it) has its own inline copy of the loop, plus the water-enhancement extras — a further refactor to share one implementation is a reasonable next step, but not required for things to work.
- **`backend/water_enhancement.py`** — `WaterEnhancer` (contrast/sharpen/color boost for water scenes), `MultiScaleDetector` (runs YOLO at 640px and 1280px, keeps whichever finds more), `AdaptiveThreshold` (loosens the confidence threshold if detections are frequent but low-confidence — a sign it's water glare, not a false positive), `SmartBoxFilter` (final filtering pass).
- **`backend/alerts.py`** — decides whether an alert is due (`should_send_alert`), builds and sends the email (`send_email_alert`), and logs it to the database (`trigger_alert`). Capped at 50 emails/day so a stuck alert can't spam you.
- **`backend/database.py`** — five tables: `detections`, `alerts`, `daily_summary`, `system_metrics`, `camera_config`, wrapped by `DatabaseManager` with simple methods like `save_detection()` and `get_recent_detections()`.

## 📡 API Reference

All endpoints are served by `backend/api.py` at `http://<server>:8000`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Body: `{"phone_ip": "192.168.0.105"}` (optional if set in `.env`). Connects to the camera and starts detection. |
| `POST` | `/api/stop` | Stops detection and releases the camera. |
| `GET` | `/api/stats` | Current live stats: count, confidence, fps, status, error (if any). |
| `GET` | `/api/detections?limit=50` | Recent detection history as JSON. |
| `GET` | `/api/detections/{id}/image` | The saved annotated image for one detection. |
| `GET` | `/api/video_feed` | Live MJPEG stream — point an `<img>` tag at this. |

You can call these directly (e.g. with `curl` or from another app) if you ever want to build a different frontend on top later.

## 🛠️ Technologies Used
Python, YOLO (Ultralytics), OpenCV, ByteTrack, FastAPI + Uvicorn, SQLAlchemy, HTML/CSS/JS, Streamlit + Plotly (alternative dashboard), SMTP email, geopy.

## 🔮 Future Improvements
- Add authentication to the web dashboard before exposing it beyond a trusted local network.
- Wire `backend/service.py` to use the `WasteDetector` class from `backend/detector.py` instead of its own inline copy of the detection loop.
- Cloud deployment with a real domain + HTTPS, live CCTV integration, GPS-based waste mapping across multiple cameras.

---

## What was fixed during cleanup

Your upload was functional in parts but had accumulated a lot of copy-pasted
Jupyter export cruft. Here's exactly what changed:

1. **`backend/detector.py` was 1286 lines** — a clean `WasteDetector` class followed by **three duplicate copies** of old notebook experiments pasted in as module-level code. One of those copies would `os.chdir()` to a hardcoded Windows path and try to open a webcam in an infinite loop the moment the file was imported. Nothing currently imports this file, so it was silently dead — but it was a landmine. **Fixed:** stripped down to just the 280-line reusable class.
2. **`backend/database.py` was 614 lines** with the same five ORM model classes **defined twice**. **Fixed:** one definition, updated to the SQLAlchemy 2.0-style import (`declarative_base` from `sqlalchemy.orm`).
3. **Bug:** `alerts.py` called `self.db.log_alert(...)`, a method that didn't exist on `DatabaseManager` (it only had `save_alert`), so every alert attempt silently failed to log. **Fixed:** added a `log_alert()` convenience method to `DatabaseManager` and corrected the call.
4. **`dashboard/dashboard.py` displayed fake hardcoded sample rows**, not your real data. **Fixed:** rewritten to query `backend.database.DatabaseManager` directly, with live metrics, charts (via Plotly, already in your requirements), and real saved detection images.
5. **Two copies of the database existed** (`data/waste_detection.db` with your real 53 detections, and an empty duplicate at `backend/data/waste_detection.db`). **Fixed:** kept only the real one, at the path the code actually expects (`data/waste_detection.db` from the project root).
6. **`run_complete_system.py` hardcoded `PROJECT_DIR = r'C:\Users\WIN10\Desktop\...'`** — would only run on that one machine. **Fixed:** now resolves its own directory automatically (`Path(__file__).resolve().parent`), and `PHONE_IP` can come from `.env` instead of always prompting.
7. **No `.env` or `.env.example` existed**, despite nearly every module reading config from environment variables. **Fixed:** added `.env.example` documenting all 19 variables actually used in the code.
8. **Notebooks, `.ipynb_checkpoints`, and `__pycache__` were scattered through every folder.** **Fixed:** notebooks moved to `notebooks/`, checkpoints and cache removed (and `.gitignore` prevents them from coming back).
9. The architecture diagram was a raw Graphviz `.dot` file with no extension. **Fixed:** renamed to `docs/architecture.dot` and rendered to `docs/architecture.png` for quick viewing.

Everything else (the actual detection/tracking/email logic) was already
working code and was left as-is.

## What was added for the web version

You asked for this to become a real public-facing product instead of a
local script with a popup window. Added:

- **`backend/service.py`** — the detection loop, made headless (no `cv2.imshow`) so it can run on a server with no display, in a background thread that a web request can start/stop.
- **`backend/api.py`** — a FastAPI server exposing that loop over HTTP: live video stream, live stats, detection history, start/stop controls.
- **`frontend/`** — a plain HTML/CSS/JS dashboard (no React/Node/build step) served directly by the FastAPI app, so there's exactly one server to run.
- **`run_web_server.py`** — the one command to start it all: `python run_web_server.py`.

Nothing in your existing `run_complete_system.py`, `dashboard/dashboard.py`, or the core detection logic was removed — they still work exactly as before, as a fallback.

## 🩹 Troubleshooting — errors you might hit

**`ModuleNotFoundError: No module named 'fastapi'` (or `cv2`, `ultralytics`, etc.)**
Your virtual environment isn't active, or dependencies weren't installed into it.
```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
In VS Code, also re-select the interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the one under `venv`.

**Browser shows "Camera not connected" placeholder forever after clicking Start**
- Check the phone and computer are on the *same* Wi-Fi network (not phone on mobile data).
- Open the IP Webcam app on the phone and confirm it says "Streaming" — if it's asleep/locked, some phones pause the camera.
- Double-check the IP typed in matches exactly what the IP Webcam app shows (just the numbers, e.g. `192.168.0.105` — no `http://` and no `:8080`, the app adds those internally).
- Try opening `http://<phone-ip>:8080` directly in a browser — you should see the IP Webcam app's own preview page. If that doesn't load, it's a network issue, not this project.

**Page loads but video never appears / stays blank**
- Refresh the page once (the `<img>` tag sometimes needs a reload after the stream starts).
- Check the terminal running `run_web_server.py` for a Python traceback — that's the real error, and the page won't always show it.

**"Address already in use" when running `run_web_server.py`**
Another process (maybe a previous run that didn't close) is already using port 8000.
```bash
# Mac/Linux — find and stop it:
lsof -i :8000
kill <PID>

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```
Or edit `run_web_server.py` and change `port=8000` to something else, e.g. `port=8001`.

**Nothing happens when you click "Start Detection" (no error, no video)**
Check the terminal — if you see `[service] Loading model:` repeating or a traceback about `best.pt`, the model file didn't load. Confirm `models/best.pt` exists and `MODEL_PATH` in `.env` (if set) points to the right path.

**`[service] Detection error: No module named 'lap'` repeating over and over, with `Access is denied` install errors**
Object tracking needs the `lap` package, and Ultralytics is trying (and failing) to auto-install it into a protected system folder. This usually means you're running the global Python install instead of your virtual environment — check your terminal prompt for `(venv)` at the start. Fix:
```bash
python -m venv venv
venv\Scripts\activate      # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
pip install lap
python run_web_server.py
```
If you don't want to deal with this at all, the service now falls back to detection without tracking automatically after the first failure (counts still work fine, you just lose per-object track IDs) — so it's safe to ignore if you'd rather not install `lap`.

**Other devices on the network can't reach `http://<your-ip>:8000`**
Your computer's firewall may be blocking incoming connections on port 8000. On Windows, allow Python through the firewall when prompted (or add a rule for port 8000). On Mac, check System Settings → Network → Firewall.

**Email alerts never arrive even with `ENABLE_EMAIL=true`**
Gmail (and most providers) block plain passwords for SMTP. Generate an **App Password** specifically for this at https://myaccount.google.com/apppasswords and use that 16-character code as `EMAIL_PASSWORD`, not your normal login password.
