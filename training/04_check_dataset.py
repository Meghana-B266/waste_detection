"""
DAY 4 - Safety inspector. Checks the built dataset for mistakes BEFORE you waste hours training.
Run:  python training/04_check_dataset.py --name A
Prints PASS or FAIL. Also saves sample pictures with boxes drawn: work/reports/check_<name>_<pile>.jpg
"""
import argparse
import hashlib
import json
import random

from PIL import Image, ImageDraw

from common import REPORTS, dataset_dir, die, ensure_dirs, label_path_for, list_images, say

random.seed(1)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def piles(d):
    out = {}
    for s in ("train", "val", "test"):
        out[s] = d / "images" / s
    if (d / "water_test" / "images").is_dir():
        out["water_test"] = d / "water_test" / "images"
    return out


def draw_sample(imgs, out_path, n=8):
    picks = random.sample(imgs, min(n, len(imgs)))
    if not picks:
        return
    tile, cols = 320, 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), (30, 30, 30))
    for i, p in enumerate(picks):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        lbl = label_path_for(p)
        for line in lbl.read_text(encoding="utf-8").splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            dr.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(255, 60, 60),
                         width=max(2, W // 200))
        sheet.paste(im.resize((tile, tile)), ((i % cols) * tile, (i // cols) * tile))
    sheet.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="A")
    args = ap.parse_args()
    ensure_dirs()
    d = dataset_dir(args.name)
    if not d.exists():
        die(f"Dataset '{args.name}' not found. Run  python training/03_build_dataset.py --name {args.name}")

    errors, warnings = [], []
    hashes = {}
    split_of_group = {}
    groups = {}
    gpath = d / "groups.json"
    if gpath.exists():
        groups = json.loads(gpath.read_text(encoding="utf-8"))
    else:
        warnings.append("groups.json missing - cannot check photo-level leakage")

    say(f"Checking dataset '{args.name}' ...\n")
    say(f"{'pile':12s} {'images':>7s} {'boxes':>7s} {'empty':>6s}")
    totals = {}
    for pile, folder in piles(d).items():
        imgs = list_images(folder)
        boxes = empty = 0
        for p in imgs:
            lbl = label_path_for(p)
            if not lbl.exists():
                errors.append(f"{pile}: no label file for {p.name}")
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                errors.append(f"{pile}: broken image {p.name}")
                continue
            lines = [l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                empty += 1
            for l in lines:
                parts = l.split()
                try:
                    ok = len(parts) == 5 and int(parts[0]) == 0 and all(0 <= float(v) <= 1 for v in parts[1:])
                    ok = ok and float(parts[3]) > 0 and float(parts[4]) > 0
                except ValueError:
                    ok = False
                if not ok:
                    errors.append(f"{pile}: bad label line in {lbl.name}: '{l}'")
                else:
                    boxes += 1
            h = md5(p)
            if h in hashes and hashes[h][0] != pile:
                errors.append(f"LEAK: identical picture in '{pile}' and '{hashes[h][0]}': {p.name}")
            hashes.setdefault(h, (pile, p.name))
            g = groups.get(f"{pile}/{p.name}")
            if g:
                if g in split_of_group and split_of_group[g] != pile:
                    errors.append(f"LEAK: original photo/session '{g}' is in both "
                                  f"'{split_of_group[g]}' and '{pile}'")
                split_of_group.setdefault(g, pile)
        totals[pile] = (len(imgs), boxes, empty)
        say(f"{pile:12s} {len(imgs):7d} {boxes:7d} {empty:6d}")
        if imgs:
            draw_sample(imgs, REPORTS / f"check_{args.name}_{pile}.jpg")

    n_train, b_train, _ = totals.get("train", (0, 0, 0))
    if n_train < 200:
        warnings.append(f"Only {n_train} training pictures - the model will be weak.")
    if b_train == 0:
        errors.append("No plastic boxes in the training pile!")
    for p in ("val", "test"):
        if totals.get(p, (0,))[0] == 0:
            errors.append(f"The '{p}' pile is empty.")
    if "water_test" not in totals:
        warnings.append("No water_test pile: you will not be able to run the real-world water exam. "
                        "Collect >=3 own water sessions and rebuild.")
    else:
        wi, wb, _ = totals["water_test"]
        if wi < 30:
            warnings.append(f"water_test has only {wi} pictures - results will be very rough. Aim for 50+.")
        if wb == 0:
            errors.append("water_test has no plastic boxes!")

    say("\nSample pictures with boxes: " + str(REPORTS) + f"/check_{args.name}_*.jpg  <- LOOK AT THEM!")
    for w in warnings:
        say(f"[WARN] {w}")
    if errors:
        say(f"\nRESULT: FAIL  ({len(errors)} problem(s))")
        for e in errors[:25]:
            say(f"  - {e}")
        if len(errors) > 25:
            say(f"  ... and {len(errors) - 25} more")
        raise SystemExit(1)
    say("\nRESULT: PASS - labels valid, no leaks between piles. Safe to train.")


if __name__ == "__main__":
    main()
