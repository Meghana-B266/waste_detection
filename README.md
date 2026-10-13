# Floating Plastic Detection

Detect floating plastic in a live camera feed with a custom-trained **YOLO11n** model, count it with a smoothing rule, log every detection to a database, show it on a web dashboard, and send **e-mail alerts** when the count reaches a threshold.

> **Status: working prototype.** It detects one class (`plastic`), was graded on a held-out water exam from a public dataset (not on a specific deployment site), and runs on a laptop CPU. See [Limitations](#limitations) before relying on it.

![Live system](docs/images/live_system.png)

## Contents

- [Overview](#overview)
- [Key results](#key-results)
- [Features](#features)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Training and evaluation pipeline](#training-and-evaluation-pipeline)
- [Datasets and credits](#datasets-and-credits)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Security and privacy](#security-and-privacy)
- [License](#license)

## Overview

The project has two halves:

1. **The application** (`backend/`, `frontend/`, `dashboard/`): reads a live video stream from an Android phone (IP Webcam app), runs the detector, keeps a stable count, stores detections in SQLite, serves a web dashboard, and sends alert e-mails.
2. **The training and evaluation pipeline** (`training/`): merges three public datasets into one single-class dataset, checks it for leakage and label errors, trains YOLO11n (on Google Colab or a local GPU), grades it on a held-out water exam, and installs the chosen model into the application.

## Key results

All numbers are for a held-out **water exam of 111 images (314 labelled boxes)** at **640 px**. The confidence threshold was chosen on separate validation data, never on the exam.

| Model | Threshold | Precision | Recall | F1 | mAP50 | False alarms |
|---|---|---|---|---|---|---|
| **A** (shipped): Marine litter + water data | 0.40 | 0.88 | 0.67 | **0.76** | 0.75 | 29 |
| B: A + household plastics | 0.30 | 0.84 | 0.69 | **0.76** | 0.75 | 42 |
| Previous model (before this work) | 0.60 | 0.40 | 0.21 | 0.28 | 0.17 | 100 |

What the experiments showed:

- **A and B tie** (F1 0.762 vs 0.755 is within the noise of a 111-image exam), so the simpler model A was kept. Extra household-plastic data did not help on water.
- **The image-enhancement step hurt every model.** With it, F1 dropped from 0.76 to 0.54 (A), 0.76 to 0.49 (B) and 0.28 to 0.08 (previous model). The recommended setting is `USE_WATER_ENHANCEMENT=false`.
- **640 px was slightly better than 1280 px** for model A (F1 0.76 vs 0.73) and about **6x faster**: the model forward pass took about 65 ms vs 390 ms per image on an Intel Core i5-1235U CPU. This is model time only; it excludes video reading, tracking, drawing and streaming, so end-to-end frame rate is lower.

**How to read these numbers honestly**

- The exam comes from a public floating-bottle dataset. It is mostly bottles, and it includes some augmented pictures (collages, grayscale, colour shifts). It does **not** measure performance at your own site.
- A small exam (111 images) means differences under about 0.03 F1 should be treated as noise.
- On a secondary pile of 52 beach and household pictures, both new models scored poorly (F1 about 0.18). The cause was not established, so this repository makes no claim about beach performance.
- Scores come from a custom evaluation script (`training/07_evaluate.py`) and may differ slightly from Ultralytics' own validator.

## Features

- Live detection and tracking of floating plastic from an IP phone camera
- **Stable counting**: the count is the most common value over the last 10 frames and is accepted only if at least 65% of them agree, so one noisy frame cannot change it
- SQLite logging of every confirmed detection, with saved annotated images
- Web dashboard (live video, count, FPS, history) served by FastAPI, plus an optional Streamlit history view
- E-mail alerts when the count reaches `ALERT_THRESHOLD` (capped at 50 per day)
- Optional water-specific extras: image enhancement, multi-scale inference, adaptive threshold, small-box filter (all switchable in `.env`)
- A reproducible, leak-aware training pipeline with automated dataset checks

## How it works

```
Phone camera (IP Webcam)  --MJPEG over Wi-Fi-->  backend/service.py (background thread)
                                                   |-- water_enhancement.py  (optional enhancement)
                                                   |-- YOLO11n + ByteTrack    (models/best.pt)
                                                   |-- box filter + stable count (10-frame vote)
                                                   |-- database.py            (SQLite)
                                                   '-- alerts.py              (e-mail if count >= ALERT_THRESHOLD)

backend/api.py (FastAPI)  <-->  frontend/ (HTML, CSS, JS)  <-->  your browser
```

- `backend/service.py`: the headless detection loop; holds the latest frame and stats that the API reads.
- `backend/api.py`: the FastAPI app (REST endpoints, live MJPEG stream, static frontend).
- `backend/water_enhancement.py`: `WaterEnhancer` (contrast, sharpness, colour), `MultiScaleDetector` (runs 640 px and 1280 px and keeps whichever finds more), `AdaptiveThreshold` (can lower the confidence threshold by 0.05 to 0.10, never below 0.25), `SmartBoxFilter` (drops boxes below `MIN_BOX_AREA`).
- `backend/alerts.py`: decides whether an alert is due and sends the e-mail. An alert needs `ENABLE_EMAIL=true` and a **stable count >= `ALERT_THRESHOLD`**; confidence does not matter.
- `backend/database.py`: tables `detections`, `alerts`, `daily_summary`, `system_metrics`, `camera_config`.
- `backend/detector.py`: a reusable `WasteDetector` class for use from other scripts.

## Repository structure

```
.
├── backend/                 # detection service, API, alerts, database, water enhancement
├── frontend/                # index.html, style.css, app.js (no build step)
├── dashboard/dashboard.py   # optional Streamlit history dashboard
├── models/                  # best.pt goes here (not tracked in Git, see Quick start)
├── data/                    # SQLite database (not tracked)
├── detections/              # saved annotated detection images
├── training/                # dataset building, checks, training, evaluation, install
├── docs/                    # architecture diagrams, screenshots, images used in this README
├── notebooks/               # original exploratory notebooks
├── run_web_server.py        # web entry point (recommended)
├── run_complete_system.py   # local OpenCV-window entry point
├── test_email.py            # sends a test e-mail using your .env settings
├── requirements.txt         # application dependencies
├── requirements-train.txt   # extra packages for training
└── .env.example             # configuration template (copy to .env)
```

Created locally and **not** committed: `.env`, `venv/`, `work/` (datasets, runs, reports), `dataset/` (downloaded dataset zips).

## Quick start

**Requirements:** Python (developed on 3.12), an Android phone with the **IP Webcam** app, and the phone and computer on the same Wi-Fi network.

### 1. Install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Get the model weights

`models/best.pt` is **not stored in Git** (`models/*.pt` is ignored). Either:

- **Download the released model:** `<add the link to your GitHub Release here>` and save it as `models/best.pt`, or
- **Train it yourself** with the [training pipeline](#training-and-evaluation-pipeline).

### 3. Configure

```bash
cp .env.example .env            # Windows: copy .env.example .env
```

Edit **`.env`** (the app reads `.env`, not `.env.example`). Settings that matter first:

- `PHONE_IP`: the IP shown in IP Webcam (numbers only, no `http://` and no `:8080`)
- `ENABLE_EMAIL=true` plus `SMTP_SERVER`, `EMAIL_FROM`, `EMAIL_PASSWORD`, `ALERT_EMAIL_TO` if you want alerts. For Gmail use an **App Password**, not your normal password.
- The recommended detection settings in the next section

`.env` is read only at start-up, so **restart the server after every change**.

### 4. Set up the phone camera

1. Install **IP Webcam** (Android) and tap **Start server**.
2. Check that `http://<phone-ip>:8080` opens in your computer's browser.
3. Put the phone IP into `PHONE_IP`.

### 5. Run (web version, recommended)

```bash
python run_web_server.py
```

Open **http://localhost:8000** in Chrome or Edge. Type the address yourself: `http://0.0.0.0:8000`, which the server prints, is not a page you can browse. Click **Start Detection**. To view it from another device on the same network, use `http://<this-computer-ip>:8000`.

### Other entry points

```bash
python run_complete_system.py          # local OpenCV window (keys: q quit, s screenshot, r reset, p pause)
streamlit run dashboard/dashboard.py   # history charts at http://localhost:8501
python test_email.py                   # test your e-mail settings without a camera
```

## Configuration

Defaults below are the values in `.env.example`; the last column is what our evaluation recommends for the shipped model.

| Variable | Default | Meaning | Recommended |
|---|---|---|---|
| `PHONE_IP` | (blank) | IP Webcam address | your phone's IP |
| `MODEL_PATH` | `models/best.pt` | Model weights | `models/best.pt` |
| `CONFIDENCE_THRESHOLD` | `0.35` | Base confidence threshold | `0.40` (chosen on validation data at 640 px) |
| `IOU_THRESHOLD` | `0.35` | Overlap threshold for removing duplicate boxes | keep |
| `MIN_BOX_AREA` | `800` | Ignore boxes smaller than this many pixels | keep; lower (for example 300) if tiny bottles are missed |
| `USE_WATER_ENHANCEMENT` | `true` | Contrast/sharpness/colour step | `false` |
| `USE_MULTISCALE` | `true` | Run 640 px and 1280 px on every frame (slow on CPU) | `false` |
| `USE_ADAPTIVE_THRESHOLD` | `true` | Let the app lower the threshold automatically | `false` (so the graded threshold is the one used) |
| `ALERT_THRESHOLD` | `5` | Stable count needed to send an alert e-mail | choose for your use case |
| `ENABLE_EMAIL` | `false` | Must be `true` for any e-mail to be sent | `true` if you want alerts |
| `SMTP_SERVER`, `SMTP_PORT` | `smtp.gmail.com`, `587` | SMTP server | as needed |
| `EMAIL_FROM`, `EMAIL_PASSWORD`, `ALERT_EMAIL_TO` | placeholders | Sender, App Password, recipient | your own (never commit them) |
| `DATABASE_URL` | `sqlite:///./data/waste_detection.db` | Database location | keep |
| `USE_PHONE_GPS`, `CAMERA_LATITUDE`, `CAMERA_LONGITUDE`, `CAMERA_LOCATION` | see file | Location shown in alerts | your camera's position |
| `IMAGE_SIZE` | not in the default code | Model input size in pixels | `640` (see below) |

**Input size.** By default `backend/service.py` runs the model at 1280 px (`self.image_size = 1280`). The shipped model was trained and graded at 640 px, where it scored slightly higher and ran about 6x faster. To make it configurable, change that line to:

```python
self.image_size = int(os.getenv('IMAGE_SIZE', 1280))
```

and set `IMAGE_SIZE=640` in `.env`. Keep `USE_MULTISCALE=false`, because multi-scale mode always uses 640 and 1280 regardless of this setting.

## API reference

Served by `backend/api.py` at `http://<server>:8000`. Interactive docs: `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Body `{"phone_ip": "192.168.0.105"}` (optional if set in `.env`). Starts detection. |
| `POST` | `/api/stop` | Stops detection and releases the camera. |
| `GET` | `/api/stats` | Live count, confidence, FPS, status, error, e-mails sent today. |
| `GET` | `/api/detections?limit=50` | Recent detection history as JSON. |
| `GET` | `/api/detections/{id}/image` | Saved annotated image for one detection. |
| `GET` | `/api/video_feed` | Live MJPEG stream (point an `<img>` tag at it). |

## Training and evaluation pipeline

![Pipeline](docs/images/pipeline.png)

The `training/` scripts rebuild the dataset, train, grade and install a model. A GPU is strongly recommended for training (a free Google Colab T4 trained model A in about 44 minutes; a laptop CPU would take many hours or more).

```bash
pip install -r requirements-train.txt
```

### 1. Get the data

Download the three public datasets from Roboflow Universe in **YOLO format** (see [Datasets and credits](#datasets-and-credits)) and put the zips in `dataset/`. The water dataset's file name **must start with `water`** (for example `water_flow_img.zip`).

### 2. Build, check and test

```bash
python training/00_check_setup.py                 # environment check
python training/01_unpack_datasets.py             # unpack into work/raw/
python training/02_inspect_datasets.py            # counts, charts, sample pictures

# Dataset A = beach litter + water; Dataset B = A + household plastics
python training/03_build_dataset.py --name A --include marine water
python training/03_build_dataset.py --name B --include marine plastics water

python training/04_check_dataset.py --name A      # must say PASS before training
python training/05_smoke_test.py --name A         # one-minute practice run
```

### 3. Train

```bash
python training/06_train.py --name A --workers 2
```

On Colab, add `--backup-dir <Drive folder>` so `last.pt`/`best.pt` are copied to Google Drive after every epoch. If a session dies, restore the run folder and use `--resume <run>/weights/last.pt`. The script refuses to resume a run that has already finished instead of silently starting a different training.

### 4. Evaluate

```bash
# model(s) to grade; add your previous model as a baseline
python training/07_evaluate.py --weights work/runs/A_yolo11n/weights/best.pt work/runs/B_yolo11n/weights/best.pt

# also grade at the app's default size
python training/07_evaluate.py --imgsz 1280 --weights work/runs/A_yolo11n/weights/best.pt

# if you trained on a separate copy of the dataset, check the exam was not studied
python training/09_check_overlap.py --name A --trained-groups trained_A_groups.json
```

This writes `work/reports/evaluation_report.md` and mistake pictures (green = correct, red = false alarm, yellow = missed plastic). Use `--gallery-focus fp --gallery-n 6 --gallery-cols 2 --gallery-tile 800 600` for a large look at the false alarms.

### 5. Install into the app

```bash
python training/08_install_model.py --weights work/runs/A_yolo11n/weights/best.pt --conf 0.4
```

This backs up the old `models/best.pt`, installs the new one (slimming it if it still carries training state), and writes `models/model_card.json`. The model card's `recommended_confidence` is only a note; the value the app uses is `CONFIDENCE_THRESHOLD` in `.env`.

### Script reference

| Script | Purpose |
|---|---|
| `00_check_setup.py` | Checks Python, packages, GPU, project files and disk space |
| `01_unpack_datasets.py` | Unpacks dataset zips into `work/raw/` |
| `02_inspect_datasets.py` | Class counts, image sizes, sample pictures with boxes |
| `03_build_dataset.py` | Merges to one class, removes duplicates and unclear labels, splits without leakage |
| `04_check_dataset.py` | Validates labels; finds duplicates and overlap between piles |
| `05_smoke_test.py` | One-epoch test that the training setup works |
| `06_train.py` | Trains YOLO11n; supports Drive backups and a safe resume |
| `07_evaluate.py` | Grades models with precision, recall, F1, mAP50, mAP50-95; threshold sweep; mistake pictures |
| `08_install_model.py` | Installs a model into the app with a backup |
| `09_check_overlap.py` | Compares the trained-on dataset copy with the graded copy |
| `collect_frames.py`, `import_photos.py` | Turn your own videos or photos into labelled-ready sessions |

### Design decisions

- **One class.** The app treats every box as plastic, and the source datasets use 5, 19 and 1 different label sets. Glass, wood and rubber objects become background.
- **No leakage.** Roboflow stores about three augmented copies of each photo, so splits are made **by original photo**. Frame-like water data is cut into **consecutive blocks with gaps** between them. Exact duplicates are removed and automated checks run before training.
- **Honest evaluation.** The threshold is chosen on validation data and then applied unchanged to the exams. Two models and a baseline are always graded on the same exam.
- **Reproducible.** Fixed seed (42); every build writes a report to `work/reports/`.

## Datasets and credits

The datasets are **not redistributed** in this repository. Check each dataset's licence before using it.

| Dataset | Content | Used for |
|---|---|---|
| Marine litter (v1) | Beach and shoreline litter, 5 classes | Training (only the `plastic` class kept) |
| plastics (v1) | Household and studio plastic items, 19 classes | Dataset B only |
| FloW-Img (Roboflow copy, [universe.roboflow.com/flow-g9yqk](https://universe.roboflow.com/flow-g9yqk)) | Bottles floating on inland water | Water training, validation and the exam |

- **FloW:** Cheng, Y. et al., *FloW: A Dataset and Benchmark for Floating Waste Detection in Inland Waters*, ICCV 2021 (original repository: ORCA-Uboat/FloW-Dataset).
- **Detector:** [Ultralytics YOLO11](https://docs.ultralytics.com/).
- **Platform:** [Roboflow Universe](https://universe.roboflow.com/) for dataset hosting and YOLO-format export.

## Limitations

- One class only: it cannot tell bottles, bags and wrappers apart.
- The water training data is mostly bottles, so other plastic is under-represented.
- The exam is a public dataset (with some augmented pictures), not your site. Night, rain and heavy glare are untested.
- Both new models scored poorly on the beach and household test pile (F1 about 0.18); the cause is unknown.
- Speed: at 640 px the model forward pass is about 65 ms on one laptop CPU; the whole application is slower, and 1280 px or multi-scale mode is much slower. Measure the end-to-end frame rate on your hardware.
- The web dashboard has **no authentication**. Do not expose it beyond a trusted network.
- Prototype quality: not production-ready.

## Roadmap

- Collect and label pictures at the real deployment site and build a site-specific exam from untouched originals
- Add classes (bottle, bag, wrapper) so alerts can say what was found
- Investigate the weak beach and household result; improve small-object detection (larger input or tiling)
- Train and run with the same image processing, or keep enhancement off
- Try a GPU or an optimised export (ONNX, OpenVINO) for higher frame rates (not tested here)
- Add authentication; HTTPS and a reverse proxy for deployment beyond a local network
- Let `service.py` reuse `WasteDetector` instead of its own copy of the loop

## Troubleshooting

**The page is blank at `http://0.0.0.0:8000`.** Open `http://localhost:8000` instead. `0.0.0.0` is a listening address, not a page. In VS Code, Ctrl+clicking the printed link opens it in a built-in browser that may not load it; type the address into Chrome or Edge.

**No e-mail arrives.**
1. `ENABLE_EMAIL` must be `true`.
2. An alert needs a **stable count >= `ALERT_THRESHOLD`** (default 5). Two items will not trigger it, however confident the model is. Alerts are checked when the stable count changes.
3. `.env` is read at start-up: restart the server after editing it.
4. Run `python test_email.py` to test the e-mail path alone.
5. For Gmail you need an **App Password** (https://myaccount.google.com/apppasswords), not your normal password.
6. Watch the terminal: it prints `Email sent`, `Email auth failed` or `Email error`.

**The app is slow.** Use `USE_MULTISCALE=false` and a 640 px input size (see [Configuration](#configuration)); the model runs at about a sixth of the cost of 1280 px. A GPU helps most.

**`ModuleNotFoundError` (`fastapi`, `cv2`, `ultralytics`).** Activate the virtual environment (`(venv)` in the prompt) and run `pip install -r requirements.txt`. In VS Code, select the interpreter under `venv`.

**"Camera not connected" or a blank video.** Check that the phone and computer are on the same Wi-Fi; that IP Webcam shows it is streaming; that `PHONE_IP` has numbers only; and that `http://<phone-ip>:8080` opens in a browser. Read the terminal for a traceback.

**`No module named 'lap'` repeating.** Tracking needs `lap`. You are probably outside the virtual environment. Activate it and run `pip install lap`. The service falls back to detection without track IDs after the first failure; counts still work.

**`Address already in use`.** Another process holds port 8000. On Windows: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`. On macOS/Linux: `lsof -i :8000` then `kill <PID>`. Or change the port in `run_web_server.py`.

**Other devices cannot reach the server.** Allow Python through the firewall for port 8000 and use the computer's LAN IP.

**Colab training froze or reset.** Training may still be running in the background; check `results.csv` before starting anything. A finished run cannot be resumed. Use `--backup-dir` so progress survives a reset (see [Train](#3-train)).

## Security and privacy

- **Never commit `.env`.** It holds your e-mail password. `.env` is in `.gitignore`; if it was ever pushed, change the App Password. Keep `.env.example` free of real values.
- The dashboard has no login: anyone who can reach the URL can start and stop detection.
- A camera can capture people and private property. Point it at water, get permission where needed, and do not publish images that show faces or vehicle plates.

