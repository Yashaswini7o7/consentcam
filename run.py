"""Run ConsentCam on a webcam, a video file or a single image.

    python run.py --source 0 --whitelist whitelist.npz --show
    python run.py --source walk.mp4 --whitelist whitelist.npz --out walk_safe.mp4
    python run.py --source samples/crowd.jpg --out crowd_safe.jpg

Keys in the preview window: q quits, d toggles debug boxes.
"""
import argparse
import json
from pathlib import Path

import cv2

from consentcam.pipeline import ConsentCam, Settings

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--whitelist")
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-text", action="store_true", help="faces only")
    ap.add_argument("--style", default="pixelate", choices=["blur", "pixelate", "solid"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--width", type=int, default=1280, help="resize input to this width")
    a = ap.parse_args()

    cam = ConsentCam(Settings(whitelist=a.whitelist, redact_text=not a.no_text, style=a.style, debug=a.debug))

    if Path(a.source).suffix.lower() in IMG:
        cam.s.text_every = 1
        out = cam.process(cv2.imread(a.source))
        cv2.imwrite(a.out or "out.jpg", out)
        print(json.dumps(cam.stats.summary(), indent=2))
        return

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if a.width and frame.shape[1] > a.width:
            frame = cv2.resize(frame, (a.width, int(frame.shape[0] * a.width / frame.shape[1])))
        out = cam.process(frame)
        if a.out:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, out.shape[1::-1])
            writer.write(out)
        if a.show:
            cv2.imshow("ConsentCam", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("d"):
                cam.s.debug = not cam.s.debug
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(json.dumps(cam.stats.summary(), indent=2))


if __name__ == "__main__":
    main()
