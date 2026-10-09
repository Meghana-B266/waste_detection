"""
DAY 5 - TRAIN THE MODEL (the long, important step).

Examples:
  python training/06_train.py --name A                       # model A data, small fast model
  python training/06_train.py --name B                       # model B data
  python training/06_train.py --name A --model yolo11s.pt --imgsz 960 --batch 8   # bigger/better, slower
  python training/06_train.py --resume work/runs/A_yolo11n/weights/last.pt         # continue after a crash

Output: work/runs/<run name>/weights/best.pt   <- the model you keep
        work/runs/<run name>/results.png       <- learning curves
"""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

from common import (RUNS, default_workers, die, ensure_dirs, import_ultralytics, pick_device, say,
                    write_abs_yaml, write_json)


def unique_run_name(base):
    name, i = base, 1
    while (RUNS / name).exists():
        i += 1
        name = f"{base}_{i}"
    return name


def summarize(save_dir):
    """Read results.csv and report the best epoch (same rule YOLO uses for best.pt)."""
    f = Path(save_dir) / "results.csv"
    if not f.exists():
        return None
    with open(f, newline="", encoding="utf-8") as fh:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    best, best_fit = None, -1
    for r in rows:
        try:
            m50, m5095 = float(r["metrics/mAP50(B)"]), float(r["metrics/mAP50-95(B)"])
        except (KeyError, ValueError):
            continue
        fit = 0.1 * m50 + 0.9 * m5095
        if fit > best_fit:
            best_fit, best = fit, r
    if not best:
        return None
    return {"epochs_run": len(rows), "best_epoch": int(float(best["epoch"])),
            "precision": float(best["metrics/precision(B)"]), "recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]), "mAP50-95": float(best["metrics/mAP50-95(B)"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="A", help="which built dataset to train on (A or B)")
    ap.add_argument("--model", default="yolo11n.pt", help="starter model: yolo11n.pt (small) / yolo11s.pt (bigger)")
    ap.add_argument("--epochs", type=int, default=100, help="how many times to study the whole pile")
    ap.add_argument("--imgsz", type=int, default=640, help="picture size; 960 helps tiny far-away plastic")
    ap.add_argument("--batch", type=int, default=16, help="pictures at once; lower it if you get 'out of memory'")
    ap.add_argument("--patience", type=int, default=25, help="stop early if no improvement for this many epochs")
    ap.add_argument("--run", default=None, help="name for this training run")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None, help="stop after this many hours (good for Colab)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="path to a last.pt to continue an interrupted run")
    args = ap.parse_args()

    ensure_dirs()
    YOLO = import_ultralytics()

    if args.resume:
        p = Path(args.resume)
        if not p.exists():
            die(f"I can't find {p}")
        say(f"Resuming from {p} ...")
        model = YOLO(str(p))
        model.train(resume=True)
        say("\nFinished resuming. Weights are next to your last.pt.")
        return

    data_yaml, _ = write_abs_yaml(args.name)
    device = pick_device(args.device)
    if device == "cpu":
        say("[WARNING] No GPU: training on CPU can take MANY hours. Consider Google Colab (see the guide).\n")
    run = unique_run_name(args.run or f"{args.name}_{Path(args.model).stem}")
    workers = args.workers if args.workers is not None else default_workers()
    say(f"Run name : {run}\nData     : {data_yaml}\nModel    : {args.model}\nDevice   : {device}\n"
        f"Epochs   : {args.epochs} (stops early after {args.patience} flat epochs)\n"
        f"Picture  : {args.imgsz}px, batch {args.batch}\n")

    kwargs = dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                  patience=args.patience, device=device, workers=workers, project=str(RUNS), name=run,
                  seed=args.seed, plots=True, exist_ok=False)
    if args.hours:
        kwargs["time"] = args.hours
    t0 = time.time()
    try:
        model = YOLO(args.model)
        model.train(**kwargs)
    except KeyboardInterrupt:
        say("\nStopped by you. You can continue later with --resume (see top of this file).")
        raise SystemExit(0)
    except Exception as e:  # noqa
        traceback.print_exc()
        msg = str(e).lower()
        hint = "Read the last lines above."
        if "out of memory" in msg:
            hint = f"GPU memory is full. Run again with a smaller batch, e.g.  --batch {max(2, args.batch // 2)}"
        elif "paging file" in msg or "shared memory" in msg:
            hint = "Windows memory problem. Run again with  --workers 0"
        die("Training stopped with an error.\n" + hint)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    summary = summarize(save_dir) or {}
    summary.update({"run": run, "dataset": args.name, "model": args.model, "imgsz": args.imgsz,
                    "minutes": round((time.time() - t0) / 60, 1), "best_weights": str(best)})
    write_json(save_dir / "run_summary.json", summary)
    say("\n" + "=" * 60)
    say("TRAINING FINISHED")
    for k, v in summary.items():
        say(f"  {k}: {v}")
    say("=" * 60)
    say(f"Your model: {best}")
    say(f"Learning curves: {save_dir / 'results.png'}")
    say("\nNext (Day 6): python training/07_evaluate.py --weights " + str(best))


if __name__ == "__main__":
    main()
