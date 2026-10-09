"""
DAY 2 - Look inside the datasets (like opening a toy box and counting the toys).
Run:  python training/02_inspect_datasets.py
Makes:  work/reports/dataset_overview.md   (numbers)
        work/reports/dataset_overview.png  (chart)
        work/reports/samples_<dataset>.jpg (photos with boxes drawn on)
"""
import collections
import random

from PIL import Image, ImageDraw

from common import (REPORTS, class_names, die, ensure_dirs, find_yolo_roots, label_path_for,
                    list_images, read_yaml, say, split_dirs)

random.seed(0)
COLORS = [(230, 57, 70), (29, 143, 255), (46, 204, 113), (255, 165, 0), (155, 89, 182),
          (241, 196, 15), (26, 188, 156), (233, 30, 99)]


def read_boxes(lbl):
    rows = []
    if not lbl.exists():
        return rows
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                rows.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
            except ValueError:
                pass
    return rows


def draw_grid(root, names, out_path, n=8):
    imgs = []
    for d in split_dirs(root).values():
        imgs += list_images(d)
    if not imgs:
        return
    picks = random.sample(imgs, min(n, len(imgs)))
    tile = 320
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), (30, 30, 30))
    for i, p in enumerate(picks):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for c, cx, cy, w, h in read_boxes(label_path_for(p)):
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            col = COLORS[c % len(COLORS)]
            dr.rectangle([x1, y1, x2, y2], outline=col, width=max(2, W // 200))
            dr.text((x1 + 3, y1 + 3), names[c] if c < len(names) else str(c), fill=col)
        im = im.resize((tile, tile))
        sheet.paste(im, ((i % cols) * tile, (i // cols) * tile))
    sheet.save(out_path, quality=90)


def main():
    ensure_dirs()
    roots = find_yolo_roots()
    if not roots:
        die("No unpacked datasets found. Run  python training/01_unpack_datasets.py  first.")

    lines = ["# Dataset overview", ""]
    all_stats = {}
    for root in roots:
        cfg = read_yaml(root / "data.yaml")
        names = class_names(cfg)
        say(f"\n=== {root.name}")
        say(f"    classes ({len(names)}): {names}")
        lines += [f"## {root.name}", f"Classes ({len(names)}): {', '.join(map(str, names))}", "",
                  "| split | images | boxes | empty-label images |", "|---|---|---|---|"]
        cls_count = collections.Counter()
        areas, sizes = [], collections.Counter()
        groups = set()
        for split, d in split_dirs(root).items():
            imgs = list_images(d)
            boxes = empty = 0
            for i, p in enumerate(imgs):
                rows = read_boxes(label_path_for(p))
                if not rows:
                    empty += 1
                for c, cx, cy, w, h in rows:
                    boxes += 1
                    cls_count[names[c] if c < len(names) else str(c)] += 1
                    areas.append(w * h)
                groups.add(p.name.split(".rf.")[0])
                if i < 40:
                    try:
                        with Image.open(p) as im:
                            sizes[im.size] += 1
                    except Exception:
                        pass
            say(f"    {split:6s}: {len(imgs):5d} images, {boxes:5d} boxes, {empty:4d} empty")
            lines.append(f"| {split} | {len(imgs)} | {boxes} | {empty} |")
        n_img = sum(len(list_images(d)) for d in split_dirs(root).values())
        lines += ["", f"- Distinct original photos (before Roboflow copies): about {len(groups)} "
                      f"(total files: {n_img})"]
        if len(groups) < n_img * 0.6:
            lines.append("- NOTE: Roboflow saved several augmented copies of each photo, "
                         "so we must split by photo, not by file (the build script does this).")
        if areas:
            areas.sort()
            small = sum(a < 0.01 for a in areas) / len(areas) * 100
            lines.append(f"- Median box covers {areas[len(areas) // 2] * 100:.2f}% of the image; "
                         f"{small:.0f}% of boxes are SMALL (<1% of the image)")
        if sizes:
            lines.append("- Common image sizes: " + ", ".join(f"{w}x{h} ({n}x)" for (w, h), n in sizes.most_common(3)))
        lines += ["", "Boxes per class: " + ", ".join(f"{k}={v}" for k, v in cls_count.most_common()), ""]
        all_stats[root.name] = cls_count
        grid = REPORTS / f"samples_{root.name}.jpg"
        draw_grid(root, names, grid)
        say(f"    sample photos saved: {grid}")
        lines.append(f"Sample photos: `{grid.name}`\n")

    lines += ["## What to notice (read this!)", "",
              "- Look at the sample pictures. Are they photos of **plastic floating on water**? "
              "Beach sand, studio tables and people holding bottles are NOT water.",
              "- Your app watches WATER. So you must add your own water photos (Day 2 step 3) "
              "and use them for the final test.", ""]
    (REPORTS / "dataset_overview.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(all_stats), figsize=(5.5 * len(all_stats), 4.5), squeeze=False)
        for ax, (name, cnt) in zip(axes[0], all_stats.items()):
            items = cnt.most_common(12)
            ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#1d8fff")
            ax.set_title(name[:32], fontsize=9)
            ax.set_xlabel("number of boxes")
        fig.tight_layout()
        fig.savefig(REPORTS / "dataset_overview.png", dpi=110)
        plt.close(fig)
        say(f"\nChart saved: {REPORTS / 'dataset_overview.png'}")
    except Exception as e:
        say(f"\n(chart skipped: {e})")
    say(f"Report saved: {REPORTS / 'dataset_overview.md'}")
    say("\nNext: collect your own water photos ->  python training/collect_frames.py --help")


if __name__ == "__main__":
    main()
