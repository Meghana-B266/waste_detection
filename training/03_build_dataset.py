"""
DAY 4 - PREPROCESSING: turn the raw datasets into ONE clean, leak-free training dataset.

Run:
  python training/03_build_dataset.py --name A --include marine own
  python training/03_build_dataset.py --name B --include marine plastics own

What it does (in kid words):
  1. Reads every raw dataset and keeps ONE class: "plastic" (that's all your app needs).
  2. Throws away broken images, duplicate images and confusing labels.
  3. Puts photos into train / val / test PILES by ORIGINAL PHOTO, so the same photo (or its
     Roboflow copies) can never be in the study pile AND the exam pile.
  4. Keeps your own water sessions together; some whole sessions become a secret
     'water_test' exam that the model never sees.
  5. Writes a report with every number, and zips the dataset (handy for Google Colab).
"""
import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import (CLASS_NAME, WORK, REPORTS, class_names, die, ensure_dirs, find_yolo_roots,
                    label_path_for, list_images, read_yaml, say, split_dirs, dataset_dir,
                    write_json, write_yaml)

# Classes in the 'plastics' dataset we do NOT trust. Any photo containing one is dropped entirely.
AMBIGUOUS_PLASTICS = {"undefined", "rdf", "non bio-degradable"}


def kind_of(root):
    n = root.name.lower()
    if n.startswith("own"):
        return "own"
    if "marine" in n:
        return "marine"
    if "plastics" in n:
        return "plastics"
    return None


def class_policy(kind, name):
    """Return 'plastic' (keep as plastic), 'ignore' (becomes background) or 'exclude' (drop the photo)."""
    n = str(name).strip().lower()
    if kind == "own":
        return "plastic"
    if kind == "marine":
        return "plastic" if n == "plastic" else "ignore"
    if kind == "plastics":
        return "exclude" if n in AMBIGUOUS_PLASTICS else "plastic"
    return "ignore"


def parse_label(path, names, kind):
    """-> (status, boxes). status: ok | empty | excluded | nolabel. boxes = [(cx,cy,w,h)] normalised."""
    if not path.exists():
        return "nolabel", []
    boxes, bad_lines = [], 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0]))
            vals = [float(x) for x in p[1:]]
        except ValueError:
            bad_lines += 1
            continue
        cname = names[c] if 0 <= c < len(names) else None
        if cname is None:
            bad_lines += 1
            continue
        pol = class_policy(kind, cname)
        if pol == "exclude":
            return "excluded", []
        if pol == "ignore":
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(vals) >= 6 and len(vals) % 2 == 0:       # polygon -> bounding box
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            bad_lines += 1
            continue
        x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or w * h < 1e-5:
            bad_lines += 1
            continue
        boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return ("ok" if boxes else "empty"), boxes


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of(kind, img_path):
    stem = img_path.name.split(".rf.")[0]
    if kind == "own":
        stem = stem.split("__")[0]          # whole SESSION is one group
    return f"{kind}:{stem}"


def collect_records(root, kind, stats):
    names = class_names(read_yaml(root / "data.yaml"))
    recs = []
    for _, d in split_dirs(root).items():
        for img in list_images(d):
            status, boxes = parse_label(label_path_for(img), names, kind)
            if status == "nolabel":
                stats[f"{kind}: skipped (no label file)"] += 1
                continue
            if status == "excluded":
                stats[f"{kind}: skipped (contains an unclear class)"] += 1
                continue
            recs.append({"img": img, "boxes": boxes, "status": status, "kind": kind,
                         "group": group_of(kind, img)})
    return recs


def dedupe(recs, stats):
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: str(r["img"])):
        try:
            h = md5(r["img"])
        except OSError:
            stats[f"{r['kind']}: skipped (unreadable file)"] += 1
            continue
        if h in seen:
            stats[f"{r['kind']}: skipped (exact duplicate image)"] += 1
            continue
        seen.add(h)
        out.append(r)
    return out


def split_public(recs, kind, seed, val_frac, test_frac):
    groups = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    keys = sorted(groups)
    random.Random(f"{seed}-{kind}").shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * test_frac)) if n >= 10 else 0
    n_val = max(1, round(n * val_frac)) if n >= 10 else 0
    out = {"train": [], "val": [], "test": []}
    for i, k in enumerate(keys):
        members = sorted(groups[k], key=lambda r: r["img"].name)
        if i < n_test:
            out["test"].append(members[0])          # ONE photo per original = honest exam
        elif i < n_test + n_val:
            out["val"].append(members[0])
        else:
            out["train"].extend(members)
    return out


def split_own(recs, seed):
    """Whole sessions -> train / val / water_test."""
    groups = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    keys = sorted(groups)
    out = {"train": [], "val": [], "water_test": []}
    if len(keys) < 3:
        out["train"] = list(recs)
        return out, False
    random.Random(f"{seed}-own").shuffle(keys)
    total = len(recs)
    for k in keys:
        if len(out["water_test"]) < 0.25 * total:
            out["water_test"].extend(groups[k])
        elif len(out["val"]) < 0.15 * total:
            out["val"].extend(groups[k])
        else:
            out["train"].extend(groups[k])
    return out, True


def cap_negatives(train, ratio, seed, kind):
    pos = [r for r in train if r["status"] == "ok"]
    neg = [r for r in train if r["status"] == "empty"]
    keep = math.ceil(ratio / (1 - ratio) * max(len(pos), 1))
    dropped = 0
    if len(neg) > keep:
        random.Random(f"{seed}-neg-{kind}").shuffle(neg)
        dropped = len(neg) - keep
        neg = neg[:keep]
    return pos + neg, dropped


def write_record(rec, split_dir, idx, max_side, stats):
    """Copy (or shrink) the image and write its label. Returns the new filename or None."""
    img = rec["img"]
    try:
        with Image.open(img) as im:
            im.load()
            w, h = im.size
            if max(w, h) > max_side:
                s = max_side / max(w, h)
                im2 = im.convert("RGB").resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
                fname = f"{rec['kind']}_{idx:05d}.jpg"
                im2.save(split_dir / "images" / fname, quality=95)
            else:
                fname = f"{rec['kind']}_{idx:05d}{img.suffix.lower()}"
                shutil.copy2(img, split_dir / "images" / fname)
    except Exception:
        stats[f"{rec['kind']}: skipped (corrupt image)"] += 1
        return None
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in rec["boxes"]]
    (split_dir / "labels" / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                     encoding="utf-8")
    return fname


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="A", help="name of the dataset to build (A, B, ...)")
    ap.add_argument("--include", nargs="+", default=["marine", "own"], choices=["marine", "plastics", "own"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280, help="shrink bigger pictures to this size")
    ap.add_argument("--neg-ratio", type=float, default=0.15,
                    help="max share of 'no plastic here' pictures kept in each public source's train pile")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--include-marine-v2", action="store_true", help="also use the near-duplicate Marine v2")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    roots = find_yolo_roots()
    if not roots:
        die("No unpacked datasets. Run  python training/01_unpack_datasets.py  first.")

    by_kind = defaultdict(list)
    for r in roots:
        k = kind_of(r)
        if k:
            by_kind[k].append(r)
        else:
            say(f"[note] ignoring unknown dataset folder: {r.name}")
    if "marine" in by_kind and not args.include_marine_v2 and len(by_kind["marine"]) > 1:
        v1 = [r for r in by_kind["marine"] if "v1" in r.name.lower()]
        chosen = v1[0] if v1 else by_kind["marine"][0]
        say(f"[note] Marine litter has several versions; using only '{chosen.name}' "
            f"(the others contain the SAME photos again).")
        by_kind["marine"] = [chosen]

    stats = Counter()
    if "own" in args.include and "own" not in by_kind:
        say("\n[WARNING] You asked for 'own' water photos but I found no labeled own dataset.\n"
            "          Put your Roboflow export zip (name must start with 'own', e.g. own_water_labeled.zip)\n"
            "          into the 'dataset' folder, run 01_unpack_datasets.py, then run this again.\n"
            "          Continuing WITHOUT your own photos - the final water test will not be possible.\n")

    say(f"Building dataset '{args.name}' from: {', '.join(args.include)}")
    final = {"train": [], "val": [], "test": [], "water_test": []}
    water_ok = False
    for kind in args.include:
        for root in by_kind.get(kind, []):
            say(f"  reading {root.name} ...")
            recs = collect_records(root, kind, stats)
            if kind != "own":
                recs = dedupe(recs, stats)
            if not recs:
                say("    (nothing usable)")
                continue
            if kind == "own":
                parts, water_ok = split_own(recs, args.seed)
                if not water_ok:
                    say("    [WARNING] fewer than 3 sessions -> NO water_test can be made. "
                        "Collect more sessions (Day 2).")
            else:
                parts = split_public(recs, kind, args.seed, args.val_frac, args.test_frac)
                parts["train"], dropped = cap_negatives(parts["train"], args.neg_ratio, args.seed, kind)
                if dropped:
                    stats[f"{kind}: dropped extra 'no plastic' training pictures"] += dropped
            for s, lst in parts.items():
                final[s].extend(lst)

    if not any(r["status"] == "ok" for r in final["train"]):
        die("The training pile has no plastic boxes at all. Check your datasets.")

    out = dataset_dir(args.name)
    if out.exists():
        if out.parent != WORK or not out.name.startswith("dataset_"):
            die("Safety stop: refusing to delete an unexpected folder.")
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (out / sub / s).mkdir(parents=True)
    for sub in ("images", "labels"):
        if final["water_test"]:
            (out / "water_test" / sub).mkdir(parents=True)

    group_map, counts = {}, defaultdict(lambda: Counter())
    for split in ("train", "val", "test", "water_test"):
        recs = sorted(final[split], key=lambda r: (r["kind"], str(r["img"])))
        if not recs:
            continue
        counters = Counter()
        if split == "water_test":
            sub_dir = _SplitDir({"images": out / "water_test" / "images", "labels": out / "water_test" / "labels"})
        else:
            sub_dir = _SplitDir({"images": out / "images" / split, "labels": out / "labels" / split})
        for r in recs:
            counters[r["kind"]] += 1
            fname = write_record(r, sub_dir, counters[r["kind"]], args.max_side, stats)
            if fname is None:
                continue
            group_map[f"{split}/{fname}"] = r["group"]
            c = counts[split]
            c["images"] += 1
            c["boxes"] += len(r["boxes"])
            c["no_plastic_images"] += int(not r["boxes"])
            c[f"from_{r['kind']}"] += 1

    write_yaml(out / "data.yaml", {"path": ".", "train": "images/train", "val": "images/val",
                                   "test": "images/test", "nc": 1, "names": {0: CLASS_NAME}})
    write_json(out / "groups.json", group_map)
    report = {"name": args.name, "include": args.include, "seed": args.seed,
              "counts": {k: dict(v) for k, v in counts.items()}, "notes": dict(stats),
              "water_test_available": bool(final["water_test"])}
    write_json(out / "build_report.json", report)

    lines = [f"# Dataset {args.name} build report", "", f"Built from: {', '.join(args.include)}  (seed {args.seed})", "",
             "| pile | images | plastic boxes | pictures with NO plastic | from |", "|---|---|---|---|---|"]
    for s in ("train", "val", "test", "water_test"):
        if s in counts:
            c = counts[s]
            frm = ", ".join(f"{k[5:]}={v}" for k, v in c.items() if k.startswith("from_"))
            lines.append(f"| {s} | {c['images']} | {c['boxes']} | {c['no_plastic_images']} | {frm} |")
    lines += ["", "## Things I skipped or changed", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(stats.items())] or ["- nothing"]
    if not final["water_test"]:
        lines += ["", "**NO water_test pile.** Your final real-world exam is missing. Add more own water sessions."]
    (out / "build_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / f"build_report_{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    say("\n" + "\n".join(lines))
    if not args.no_zip:
        say("\nZipping for Google Colab (this can take a minute) ...")
        z = shutil.make_archive(str(WORK / f"dataset_{args.name}"), "zip", root_dir=str(WORK),
                                base_dir=f"dataset_{args.name}")
        say(f"Zip saved: {z}")
    say(f"\nDataset ready: {out}\nNext: python training/04_check_dataset.py --name {args.name}")


class _SplitDir:
    """Tiny helper so write_record can use  split_dir / 'images'  for any pile."""
    def __init__(self, mapping):
        self.m = mapping

    def __truediv__(self, key):
        return self.m[key]


if __name__ == "__main__":
    main()
