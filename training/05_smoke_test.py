"""
DAY 4 - SMOKE TEST: a tiny 1-minute practice run.
If this passes, the big training WILL start. (Like checking the car starts before a long trip.)
Run:  python training/05_smoke_test.py --name A
"""
import argparse
import time
import traceback
from pathlib import Path

from common import (RUNS, dataset_dir, die, ensure_dirs, import_ultralytics, list_images, pick_device,
                    say, write_abs_yaml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="A")
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    ensure_dirs()
    YOLO = import_ultralytics()
    data_yaml, _ = write_abs_yaml(args.name)
    device = pick_device()
    say(f"Device: {device}   (0 = your NVIDIA GPU, cpu = slow computer brain)")
    say("Starting a tiny practice training (this downloads the starter model the first time) ...\n")
    t0 = time.time()
    try:
        model = YOLO(args.model)
        model.train(data=str(data_yaml), epochs=1, imgsz=320, batch=4, fraction=0.05, workers=0,
                    device=device, project=str(RUNS), name="smoke", exist_ok=True, plots=False,
                    seed=0, verbose=False)
    except Exception as e:  # noqa
        traceback.print_exc()
        die("The practice run failed. Read the last lines above. Common fixes:\n"
            "  - 'No module named ...'          -> pip install -U ultralytics\n"
            "  - download/connection error      -> check internet, then run again\n"
            "  - 'out of memory'                -> this is only a tiny test, close other programs\n"
            "  - 'dataset not found'            -> run 03_build_dataset.py then 04_check_dataset.py")

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():
        die(f"Training ended but I can't find weights in {save_dir / 'weights'}")

    say("\nTesting that the new model can look at a picture ...")
    m2 = YOLO(str(best))
    imgs = list_images(dataset_dir(args.name) / "images" / "val")
    if not imgs:
        die("The val pile has no pictures.")
    res = m2.predict(str(imgs[0]), imgsz=320, conf=0.01, device=device, verbose=False)
    say(f"It looked at {imgs[0].name} and found {len(res[0].boxes)} possible plastic box(es) "
        f"(a number near 0 or random is NORMAL after 1 tiny epoch).")
    say(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s.  Your setup works.")
    say("Next: python training/06_train.py --name " + args.name)


if __name__ == "__main__":
    main()
