"""
DAY 2 - Collect YOUR OWN water photos from a phone camera, a webcam or a video file.

Examples:
  python training/collect_frames.py --source phone --session lake-morning
  python training/collect_frames.py --source video.mp4 --session river-video1 --every 2
  python training/collect_frames.py --source 0 --session webcam-test

--source phone   uses PHONE_IP from your .env  (IP Webcam app: http://PHONE_IP:8080/video)
--session NAME   one name per visit/video (lake-morning, lake-evening ...). IMPORTANT:
                 later we keep each session together, so it can't leak into the test set.
Press Ctrl+C any time to stop; the photos saved so far are kept.
"""
import argparse
import os
import re
import time

import cv2
import numpy as np

from common import ROOT, WORK, die, ensure_dirs, say

OUT_DIR = WORK / "raw" / "own_water_frames"
MAX_SIDE = 1280


def resolve_source(src):
    if src.lower() == "phone":
        ip = os.getenv("PHONE_IP", "")
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            ip = os.getenv("PHONE_IP", ip)
        except Exception:
            pass
        if not ip:
            die("PHONE_IP is not set. Put it in your .env file (copy .env.example to .env), "
                "or run with  --source http://YOUR_PHONE_IP:8080/video")
        return f"http://{ip}:8080/video", True
    if src.isdigit():
        return int(src), True
    if src.startswith(("http://", "https://", "rtsp://")):
        return src, True
    if not os.path.exists(src):
        die(f"I can't find the video file: {src}")
    return src, False


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def small_gray(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="phone | webcam number | video file | stream URL")
    ap.add_argument("--session", required=True, help="a short name for this visit, e.g. lake-morning")
    ap.add_argument("--every", type=float, default=2.0, help="save one picture every N seconds (default 2)")
    ap.add_argument("--max", type=int, default=150, help="stop after this many saved pictures (default 150)")
    ap.add_argument("--min-sharp", type=float, default=40.0, help="skip pictures blurrier than this")
    args = ap.parse_args()

    session = re.sub(r"[^A-Za-z0-9]+", "-", args.session).strip("-").lower()
    if not session:
        die("Please give --session a name using letters/numbers, e.g. lake-morning")

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src, is_live = resolve_source(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        die("I could not open the camera/video.\n"
            "For the phone: is the IP Webcam app running, and are phone + computer on the same Wi-Fi?\n"
            "Try opening  http://PHONE_IP:8080  in your computer's browser first.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(fps * args.every)))
    existing = len(list(OUT_DIR.glob(f"{session}__*.jpg")))
    saved = skipped_blur = skipped_dup = frame_no = fails = 0
    last_small = None
    last_time = 0.0
    say(f"Collecting into {OUT_DIR}")
    say(f"Session '{session}': one picture every {args.every}s, up to {args.max}. Ctrl+C to stop.\n")

    try:
        while saved < args.max:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if not is_live or fails > 20:
                    break
                time.sleep(0.2)
                continue
            fails = 0
            frame_no += 1
            if is_live:
                now = time.time()
                if now - last_time < args.every:
                    continue
                last_time = now
            elif frame_no % step_frames != 0:
                continue

            if sharpness(frame) < args.min_sharp:
                skipped_blur += 1
                continue
            sm = small_gray(frame)
            if last_small is not None and float(np.abs(sm - last_small).mean()) < 4.0:
                skipped_dup += 1
                continue
            last_small = sm

            h, w = frame.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            name = f"{session}__{existing + saved + 1:04d}.jpg"
            if not cv2.imwrite(str(OUT_DIR / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                die(f"Could not write {OUT_DIR / name} (disk full?)")
            saved += 1
            if saved % 10 == 0 or saved == 1:
                say(f"  saved {saved} pictures ...")
    except KeyboardInterrupt:
        say("\nStopped by you (Ctrl+C).")
    finally:
        cap.release()

    say(f"\nDONE. Saved {saved} new pictures for session '{session}'.")
    say(f"Skipped: {skipped_blur} too blurry, {skipped_dup} too similar to the last one.")
    total = len(list(OUT_DIR.glob("*.jpg")))
    sessions = sorted({p.name.split("__")[0] for p in OUT_DIR.glob("*.jpg")})
    say(f"Folder now has {total} pictures from {len(sessions)} session(s): {', '.join(sessions)}")
    if saved == 0:
        say("Nothing was saved. Is the picture too blurry/dark/still? Try  --min-sharp 10")
    if len(sessions) < 4:
        say("TIP: aim for at least 4 different sessions (different time of day / place / weather).")


if __name__ == "__main__":
    main()
