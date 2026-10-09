"""
DAY 2 - Unpack your dataset zip files.
Run:  python training/01_unpack_datasets.py
Takes every .zip inside waste-detection/dataset/ and opens it into work/raw/<name>/
Safe to run again: already-unpacked datasets are skipped (use --force to redo).
"""
import argparse
import re
import shutil
import zipfile

from common import RAW, ZIP_DIR, die, ensure_dirs, free_gb, say


def slug(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="unpack again even if it already exists")
    args = ap.parse_args()

    ensure_dirs()
    zips = sorted(ZIP_DIR.glob("*.zip")) if ZIP_DIR.exists() else []
    if not zips:
        die(f"No .zip files found in {ZIP_DIR}\nPut your dataset zips there and run again.")

    need_gb = sum(z.stat().st_size for z in zips) * 1.5 / 1024 ** 3
    if free_gb(RAW) < need_gb + 1:
        die(f"Not enough disk space. Need about {need_gb + 1:.1f} GB free.")

    for z in zips:
        target = RAW / slug(z.stem)
        if target.exists() and (target / "data.yaml").exists() and not args.force:
            say(f"[skip]   {z.name}  (already unpacked -> {target.name})")
            continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        say(f"[unpack] {z.name}  ->  work/raw/{target.name}/ ...")
        try:
            with zipfile.ZipFile(z) as zf:
                bad = zf.testzip()
                if bad:
                    die(f"The zip '{z.name}' is damaged (bad file: {bad}). Download it again.")
                zf.extractall(target)
        except zipfile.BadZipFile:
            die(f"'{z.name}' is not a valid zip file. Download it again.")
        if not (target / "data.yaml").exists():
            # some zips have one extra folder inside; move its contents up
            subs = [p for p in target.iterdir() if p.is_dir()]
            if len(subs) == 1 and (subs[0] / "data.yaml").exists():
                for item in subs[0].iterdir():
                    shutil.move(str(item), str(target / item.name))
                subs[0].rmdir()
        if not (target / "data.yaml").exists():
            say(f"  [WARN] no data.yaml inside {z.name} - is it a YOLO-format export?")
        else:
            say("  done.")

    say("\nAll good. Next:  python training/02_inspect_datasets.py")


if __name__ == "__main__":
    main()
