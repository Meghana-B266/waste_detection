"""
common.py - small helpers shared by every script in training/.
You never run this file yourself.
"""
import io
import json
import os
import shutil
import sys
from pathlib import Path

# Make printing safe on Windows consoles (no crash on odd characters).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]      # the waste-detection/ folder
ZIP_DIR = ROOT / "dataset"                       # where your dataset .zip files live
WORK = ROOT / "work"                             # EVERYTHING we generate goes here
RAW = WORK / "raw"                               # unpacked datasets
REPORTS = WORK / "reports"                       # charts, tables, reports
RUNS = WORK / "runs"                             # training runs (weights live here)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAME = "plastic"                           # our ONE class (the app treats every box as plastic)


def ensure_dirs():
    for d in (WORK, RAW, REPORTS, RUNS):
        d.mkdir(parents=True, exist_ok=True)


def say(msg=""):
    print(msg, flush=True)


def die(msg, code=1):
    """Stop the script with a friendly message instead of a scary traceback."""
    print("\n[STOP] " + msg + "\n", flush=True)
    sys.exit(code)


def read_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def class_names(data_yaml_dict):
    """Roboflow writes names as a list, Ultralytics sometimes as a dict. Return a list."""
    names = data_yaml_dict.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def find_yolo_roots(base=None):
    """Find every folder under work/raw that has a data.yaml (= one unpacked dataset)."""
    base = Path(base) if base else RAW
    roots = []
    if not base.exists():
        return roots
    for y in sorted(base.rglob("data.yaml")):
        if len(y.relative_to(base).parts) <= 3:
            roots.append(y.parent)
    return roots


def split_dirs(root):
    """Return {'train': images_dir, 'valid': images_dir, ...} for a Roboflow-style dataset."""
    out = {}
    for name in ("train", "valid", "val", "test"):
        d = Path(root) / name / "images"
        if d.is_dir():
            out[name] = d
    return out


def label_path_for(img_path):
    """
    Find the label file of a picture by swapping the LAST folder called 'images' for 'labels'.
      Roboflow layout:  .../train/images/a.jpg  ->  .../train/labels/a.txt
      YOLO layout:      .../images/train/a.jpg  ->  .../labels/train/a.txt
    """
    img_path = Path(img_path)
    parts = list(img_path.parent.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts) / (img_path.stem + ".txt")


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def dataset_dir(name):
    return WORK / f"dataset_{name}"


def write_abs_yaml(name):
    """
    Ultralytics can get confused by relative paths (especially between your PC and Colab).
    So right before training/evaluating we write small yaml files with the ABSOLUTE path.
    Returns (main_yaml_path, water_yaml_path_or_None).
    """
    d = dataset_dir(name)
    if not (d / "images" / "train").is_dir():
        die(f"I can't find the dataset '{name}' at:\n  {d}\n"
            f"Did you run  python training/03_build_dataset.py --name {name}  yet?")
    main = {
        "path": str(d.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: CLASS_NAME},
    }
    main_path = d / "data_abs.yaml"
    write_yaml(main_path, main)
    water_path = None
    if (d / "water_test" / "images").is_dir() and list_images(d / "water_test" / "images"):
        w = {
            "path": str(d.resolve()),
            "train": "water_test/images",
            "val": "water_test/images",
            "test": "water_test/images",
            "nc": 1,
            "names": {0: CLASS_NAME},
        }
        water_path = d / "water_abs.yaml"
        write_yaml(water_path, w)
    return main_path, water_path


def pick_device(requested="auto"):
    """'auto' -> first GPU if there is one, else 'cpu'."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_workers():
    return 2 if os.name == "nt" else 4


def import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:  # noqa
        die("I could not import 'ultralytics'.\n"
            "Fix:  pip install -U ultralytics\n"
            f"(technical detail: {e})")


def free_gb(path):
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def newest_file(paths):
    paths = [Path(p) for p in paths if Path(p).exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None
