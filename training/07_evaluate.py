"""
DAY 6 - EVALUATION: give the model(s) an exam they have never seen, and grade it.

Examples:
  python training/07_evaluate.py --weights work/runs/A_yolo11n/weights/best.pt
  python training/07_evaluate.py --weights work/runs/A_yolo11n/weights/best.pt work/runs/B_yolo11n/weights/best.pt
  python training/07_evaluate.py --weights models/best.pt          # grade your OLD model too!

Exams used (all from dataset A so every model gets the SAME questions):
  val         -> only used to pick the best 'confidence threshold'
  test        -> public photos never used for studying
  water_test  -> YOUR water sessions never used for studying  (the one that matters!)
  water_test + app enhancement -> same, after your app's WaterEnhancer (that's what the live app does)

Grades:  Precision = of everything it called plastic, how much really was plastic.
         Recall    = of all real plastic, how much it found.
         F1        = one number mixing both.   mAP50 / mAP50-95 = standard detector scores (higher=better).
Output:  work/reports/evaluation_report.md  +  error pictures in work/reports/eval/
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from common import (REPORTS, ROOT, WORK, dataset_dir, die, ensure_dirs, import_ultralytics, label_path_for,
                    list_images, pick_device, say, write_json)

IOU_THRS = [0.5 + 0.05 * i for i in range(10)]
SWEEP = [round(0.05 * i, 2) for i in range(1, 20)]


# ------------------------------------------------------------------ scoring maths (no YOLO needed)
def read_gt(img_path):
    """Ground-truth boxes as normalised x1,y1,x2,y2."""
    lbl = label_path_for(img_path)
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                _, cx, cy, w, h = [float(v) for v in p]
                out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match(pred_boxes, pred_conf, gt, thr):
    """Greedy matching, highest confidence first. Returns (tp_flags aligned with pred order, matched_gt_flags)."""
    n = len(pred_boxes)
    tp = np.zeros(n, dtype=bool)
    gt_hit = np.zeros(len(gt), dtype=bool)
    if n == 0 or len(gt) == 0:
        return tp, gt_hit
    ious = iou_matrix(pred_boxes, gt)
    for i in np.argsort(-pred_conf):
        cand = np.where(~gt_hit & (ious[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            gt_hit[j] = True
            tp[i] = True
    return tp, gt_hit


def average_precision(confs, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs))
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map_scores(preds):
    """preds: list of dict(boxes, conf, gt). Returns (AP50, mAP50-95)."""
    n_gt = sum(len(p["gt"]) for p in preds)
    if n_gt == 0:
        return float("nan"), float("nan")
    aps = []
    for thr in IOU_THRS:
        confs, tps = [], []
        for p in preds:
            tp, _ = match(p["boxes"], p["conf"], p["gt"], thr)
            confs.extend(p["conf"].tolist())
            tps.extend(tp.tolist())
        aps.append(average_precision(confs, tps, n_gt))
    return aps[0], float(np.nanmean(aps))


def prf_at(preds, t, thr=0.5):
    tp = fp = fn = 0
    for p in preds:
        keep = p["conf"] >= t
        b, c = p["boxes"][keep], p["conf"][keep]
        flags, _ = match(b, c, p["gt"], thr)
        tp += int(flags.sum())
        fp += int((~flags).sum())
        fn += len(p["gt"]) - int(flags.sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


def best_threshold(preds, default=0.35):
    if sum(len(p["gt"]) for p in preds) == 0:
        return default
    best_t, best_f1 = default, -1.0
    for t in SWEEP:
        f1 = prf_at(preds, t)[2]
        if f1 > best_f1 + 1e-9:
            best_t, best_f1 = t, f1
    return best_t


# ------------------------------------------------------------------ using the model
def tidy_boxes(b):
    """Make sure x1<=x2 and y1<=y2 and everything is inside 0..1 (protects the maths and the drawing)."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    out = np.stack([np.minimum(b[:, 0], b[:, 2]), np.minimum(b[:, 1], b[:, 3]),
                    np.maximum(b[:, 0], b[:, 2]), np.maximum(b[:, 1], b[:, 3])], axis=1)
    return np.clip(out, 0.0, 1.0)


def predict_all(model, images, imgsz, device):
    by_name = {p.name: p for p in images}
    out, speeds = {}, []
    gen = model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=0.001, iou=0.6, max_det=300,
                        device=device, verbose=False, stream=True)
    for r in gen:
        name = Path(r.path).name
        if name not in by_name:
            continue
        b = r.boxes
        if b is None or len(b) == 0:
            boxes, conf = np.zeros((0, 4)), np.zeros((0,))
        else:
            boxes = tidy_boxes(b.xyxyn.cpu().numpy().astype(np.float64))
            conf = b.conf.cpu().numpy().astype(np.float64)
        out[name] = {"boxes": boxes, "conf": conf, "gt": read_gt(by_name[name]), "path": by_name[name]}
        try:
            speeds.append(float(r.speed.get("inference", 0.0)))
        except Exception:
            pass
    return [out[n] for n in sorted(out)], (float(np.mean(speeds)) if speeds else float("nan"))


def error_gallery(preds, t, out_path, n=12, tile=(400, 300)):
    scored = []
    for p in preds:
        keep = p["conf"] >= t
        b, c = p["boxes"][keep], p["conf"][keep]
        tp, hit = match(b, c, p["gt"], 0.5)
        score = int((~tp).sum()) + int((~hit).sum())
        if score:
            scored.append((score, p, b, tp, hit))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return False
    picks = scored[:n]
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile[0], rows * tile[1]), (25, 25, 25))
    for i, (_, p, b, tp, hit) in enumerate(picks):
        im = Image.open(p["path"]).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        lw = max(2, W // 250)
        for g, h in zip(p["gt"], hit):
            if not h:                                   # missed plastic = YELLOW
                dr.rectangle([g[0] * W, g[1] * H, g[2] * W, g[3] * H], outline=(255, 215, 0), width=lw)
        for bb, ok in zip(b, tp):                       # correct = GREEN, false alarm = RED
            dr.rectangle([bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H],
                         outline=(60, 220, 90) if ok else (255, 60, 60), width=lw)
        im = ImageOps.contain(im, tile)
        sheet.paste(im, ((i % cols) * tile[0], (i // cols) * tile[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return True


def make_enhanced_copy(src_images_dir, dst_root):
    """Run the app's own WaterEnhancer over every picture (what the live app does before detecting)."""
    import cv2
    sys.path.insert(0, str(ROOT))
    try:
        from backend.water_enhancement import WaterEnhancer
    except Exception as e:  # noqa
        say(f"  (cannot load backend.water_enhancement: {e}) - skipping enhanced exam")
        return None
    enh = WaterEnhancer()
    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images").mkdir(parents=True)
    (dst_root / "labels").mkdir(parents=True)
    for p in list_images(src_images_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(dst_root / "images" / (p.stem + ".jpg")), enh.auto_enhance(img),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl = label_path_for(p)
        if lbl.exists():
            shutil.copy2(lbl, dst_root / "labels" / lbl.name)
    return dst_root / "images"


def run_label(w):
    w = Path(w)
    return w.parent.parent.name if w.parent.name == "weights" else w.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="one or more .pt model files to grade")
    ap.add_argument("--name", default="A", help="dataset whose exams to use (default A)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-enhance", action="store_true", help="skip the 'with app enhancement' exam")
    args = ap.parse_args()

    ensure_dirs()
    YOLO = import_ultralytics()
    d = dataset_dir(args.name)
    if not (d / "images" / "val").is_dir():
        die(f"Dataset '{args.name}' not found. Build it first (Day 4).")
    device = pick_device(args.device)

    exams = {"val": list_images(d / "images" / "val"), "test": list_images(d / "images" / "test")}
    water_dir = d / "water_test" / "images"
    if water_dir.is_dir() and list_images(water_dir):
        exams["water_test"] = list_images(water_dir)
        if not args.no_enhance:
            enh_dir = make_enhanced_copy(water_dir, WORK / "tmp_enhanced" / f"{args.name}_water_test")
            if enh_dir:
                exams["water_test+enhance"] = list_images(enh_dir)
    else:
        say("[WARNING] No water_test exam in this dataset. The most important grade will be missing!")

    results, labels_seen = [], set()
    for w in args.weights:
        wp = Path(w)
        if not wp.exists():
            die(f"I can't find the model file: {wp}")
        label = run_label(wp)
        while label in labels_seen:
            label += "_"
        labels_seen.add(label)
        say(f"\n=== Grading: {label}  ({wp})")
        try:
            model = YOLO(str(wp))
        except Exception as e:  # noqa
            die(f"Could not load {wp}: {e}\nTip: use the same ultralytics version you trained with.")
        names = getattr(model, "names", {})
        if len(names) != 1:
            say(f"  [note] this model has {len(names)} classes {names}; the exam counts ANY box as 'plastic'.")

        cache, speeds = {}, []
        for ename, imgs in exams.items():
            cache[ename], sp = predict_all(model, imgs, args.imgsz, device)
            speeds.append(sp)
        t_star = best_threshold(cache["val"])
        say(f"  best confidence threshold (chosen on 'val'): {t_star}")
        for ename, preds in cache.items():
            ap50, map5095 = map_scores(preds)
            p_, r_, f1, tp, fp, fn = prf_at(preds, t_star)
            row = {"model": label, "exam": ename, "images": len(preds), "mAP50": ap50, "mAP50-95": map5095,
                   "precision": p_, "recall": r_, "f1": f1, "threshold": t_star, "tp": tp, "fp": fp, "fn": fn,
                   "ms_per_image": float(np.nanmean(speeds)) if speeds else float("nan")}
            results.append(row)
            say(f"  {ename:20s} mAP50={ap50:.3f}  P={p_:.3f}  R={r_:.3f}  F1={f1:.3f}   "
                f"(found {tp}, false alarms {fp}, missed {fn})")
            if ename in ("test", "water_test", "water_test+enhance"):
                pic = REPORTS / "eval" / label / f"{ename.replace('+', '_')}_mistakes.jpg"
                if error_gallery(preds, t_star, pic):
                    say(f"      mistakes picture: {pic}")

    # ---------------------------------------------------------------- report
    lines = ["# Evaluation report", "",
             "Green box = correct, RED = false alarm, YELLOW = plastic the model missed.", "",
             "| model | exam | pictures | mAP50 | mAP50-95 | precision | recall | F1 | threshold | found | false alarms | missed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["exam"] == "val":
            continue
        lines.append(f"| {r['model']} | {r['exam']} | {r['images']} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['threshold']} | "
                     f"{r['tp']} | {r['fp']} | {r['fn']} |")
    lines.append("")
    key = "water_test" if "water_test" in exams else "test"
    cand = [r for r in results if r["exam"] == key]
    if cand:
        win = max(cand, key=lambda r: (r["f1"], r["mAP50"]))
        lines += [f"**Winner on `{key}` (by F1): {win['model']}**  -  F1 {win['f1']:.3f}, "
                  f"recommended CONFIDENCE_THRESHOLD = {win['threshold']}", ""]
        say(f"\nWINNER on '{key}': {win['model']}  (F1 {win['f1']:.3f}, use CONFIDENCE_THRESHOLD={win['threshold']})")
    if "water_test+enhance" in exams:
        lines.append("Compare `water_test` vs `water_test+enhance`: if enhancement gives a LOWER F1, "
                     "set USE_WATER_ENHANCEMENT=false in your .env.")
    lines += ["", "Small exams (few pictures) give shaky grades. Treat differences under ~0.03 as a tie."]
    rep = REPORTS / "evaluation_report.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORTS / "evaluation_results.json", results)
    say(f"\nReport saved: {rep}")
    say("Next (Day 7): python training/08_install_model.py --weights <best model> --conf <threshold>")


if __name__ == "__main__":
    main()
