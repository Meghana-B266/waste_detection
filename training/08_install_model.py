"""
DAY 7 - Put the winning model into your app (safely, with a backup of the old one).

Run:  python training/08_install_model.py --weights work/runs/A_yolo11n/weights/best.pt --conf 0.3
Then: python run_web_server.py
"""
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from common import (REPORTS, ROOT, dataset_dir, die, import_ultralytics, list_images, pick_device, say,
                    write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=None, help="threshold recommended by 07_evaluate.py")
    ap.add_argument("--name", default="A", help="dataset used for the quick sanity picture")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        die(f"I can't find {w}")
    YOLO = import_ultralytics()
    try:
        model = YOLO(str(w))
    except Exception as e:  # noqa
        die(f"This model file could not be loaded: {e}")
    names = dict(getattr(model, "names", {}))
    say(f"Model loaded. Classes: {names}")
    if len(names) != 1:
        say("[WARNING] The app expects ONE class (plastic). Your model has more. "
            "It will still run (every box is labeled 'Plastic') but double-check this is the right file.")

    # sanity: really make a prediction
    pics = list_images(dataset_dir(args.name) / "water_test" / "images") or \
        list_images(dataset_dir(args.name) / "images" / "val")
    if pics:
        res = model.predict(str(pics[0]), conf=0.25, device=pick_device(), verbose=False)
        say(f"Sanity check OK: looked at {pics[0].name}, found {len(res[0].boxes)} box(es).")
    else:
        say("(no picture available for the sanity check - skipped)")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    target = models_dir / "best.pt"
    if target.exists():
        if target.resolve() == w.resolve():
            die("That file already IS models/best.pt. Nothing to do.")
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = models_dir / f"best_backup_{stamp}.pt"
        shutil.copy2(target, backup)
        say(f"Old model backed up -> {backup.name}")
    shutil.copy2(w, target)
    say(f"New model installed -> {target}")

    card = {"installed": dt.datetime.now().isoformat(timespec="seconds"), "source_weights": str(w),
            "classes": names, "recommended_confidence": args.conf}
    try:
        import ultralytics
        card["ultralytics_version"] = ultralytics.__version__
    except Exception:
        pass
    summ = w.parent.parent / "run_summary.json"
    if summ.exists():
        card["training_summary"] = json.loads(summ.read_text(encoding="utf-8"))
    res_json = REPORTS / "evaluation_results.json"
    if res_json.exists():
        rows = json.loads(res_json.read_text(encoding="utf-8"))
        label = w.parent.parent.name if w.parent.name == "weights" else w.stem
        card["evaluation"] = [r for r in rows if r.get("model") == label and r.get("exam") != "val"]
    write_json(models_dir / "model_card.json", card)

    say("\nIn your .env file set (or check) these lines:")
    say(f"  MODEL_PATH=models/best.pt")
    if args.conf:
        say(f"  CONFIDENCE_THRESHOLD={args.conf}")
    say("  USE_WATER_ENHANCEMENT=true   # or false, if 07_evaluate.py showed enhancement HURT")
    say("\nThen start the app:  python run_web_server.py   and open http://localhost:8000")
    say("To undo everything: copy the best_backup_*.pt file back to best.pt")


if __name__ == "__main__":
    main()
