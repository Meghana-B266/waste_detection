"""
DAY 1 - Setup checker.
Run:  python training/00_check_setup.py
It checks your computer and tells you exactly what (if anything) to fix.
"""
import importlib
import sys

from common import ROOT, ZIP_DIR, ensure_dirs, free_gb, say

problems = []
warnings = []


def ok(msg):
    say(f"  [OK]   {msg}")


def bad(msg, fix):
    say(f"  [FIX]  {msg}")
    say(f"         -> {fix}")
    problems.append(msg)


def warn(msg):
    say(f"  [WARN] {msg}")
    warnings.append(msg)


say("=" * 60)
say("WASTE-DETECTION TRAINING - SETUP CHECK")
say("=" * 60)

# 1. Python
v = sys.version_info
if v < (3, 9):
    bad(f"Python {v.major}.{v.minor} is too old", "Install Python 3.10, 3.11 or 3.12 from python.org")
elif v >= (3, 14):
    warn(f"Python {v.major}.{v.minor} is very new; if pip install fails, use Python 3.11 or 3.12")
else:
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

# 2. Packages
say("\nPackages:")
need = [
    ("numpy", "numpy"), ("PIL", "pillow"), ("yaml", "pyyaml"), ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"), ("dotenv", "python-dotenv"),
]
for mod, pipname in need:
    try:
        m = importlib.import_module(mod)
        ok(f"{pipname} {getattr(m, '__version__', '')}")
    except Exception:
        bad(f"{pipname} is missing", f"pip install {pipname}")

torch = None
try:
    import torch  # noqa
    ok(f"torch {torch.__version__}")
except Exception:
    bad("torch is missing", "pip install torch torchvision   (or: pip install -U ultralytics, it brings torch)")

try:
    import ultralytics
    ver = ultralytics.__version__
    ok(f"ultralytics {ver}")
    try:
        parts = [int(p) for p in ver.split(".")[:2]]
        if parts < [8, 3]:
            warn("ultralytics is older than 8.3 - YOLO11 models need 8.3 or newer")
            say("         -> pip install -U ultralytics")
    except Exception:
        pass
except Exception:
    bad("ultralytics is missing", "pip install -U ultralytics")

# 3. GPU
say("\nGraphics card (GPU):")
if torch is not None:
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            ok(f"NVIDIA GPU found: {name} ({mem:.1f} GB)  -> you can train on this computer")
            if mem < 4:
                warn("Less than 4 GB GPU memory: use --batch 8 (or 4) when training")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            ok("Apple GPU (MPS) found -> training works, a bit slower than NVIDIA")
        else:
            warn("No GPU found. Training on CPU takes MANY hours.")
            say("         -> Use free Google Colab for Days 4-5 (the guide explains how).")
    except Exception as e:
        warn(f"Could not check GPU: {e}")

# 4. Folders / files
say("\nProject files:")
ensure_dirs()
ok(f"Project folder: {ROOT}")
if len(str(ROOT)) > 60:
    warn("Your project path is long. On Windows, long paths can break things. "
         "Consider moving it to something short like C:\\waste-detection")
for f in ("backend/water_enhancement.py", "models/best.pt", "requirements.txt"):
    if (ROOT / f).exists():
        ok(f)
    else:
        bad(f"{f} not found", "Run this script from inside the waste-detection folder")

zips = sorted(ZIP_DIR.glob("*.zip")) if ZIP_DIR.exists() else []
if zips:
    for z in zips:
        ok(f"dataset zip: {z.name} ({z.stat().st_size / 1024 ** 2:.0f} MB)")
else:
    bad("No dataset zips in the 'dataset' folder", "Put your 3 dataset .zip files into waste-detection/dataset/")

# 5. Disk
gb = free_gb(ROOT)
if gb < 5:
    bad(f"Only {gb:.1f} GB free disk space", "Free up at least 5 GB")
else:
    ok(f"{gb:.0f} GB free disk space")

say("\n" + "=" * 60)
if problems:
    say(f"RESULT: {len(problems)} thing(s) to fix (marked [FIX] above). Fix them, then run this again.")
    sys.exit(1)
say("RESULT: ALL GOOD! You are ready for Day 2." + (f"  ({len(warnings)} warning(s) - read them.)" if warnings else ""))
