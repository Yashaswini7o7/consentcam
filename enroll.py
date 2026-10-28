"""Enroll the people who agreed to be on camera.

    python enroll.py --name yashaswini --images me1.jpg me2.jpg me3.jpg
    python enroll.py --name yashaswini --webcam          # grabs 10 frames

Only the 128-number SFace embedding is saved, never the photo. Re-running with
the same --out file appends to it. Delete the .npz and consent is gone.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from consentcam.detectors import FaceDetector, FaceMatcher


def largest(faces):
    return max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--out", default="whitelist.npz")
    a = ap.parse_args()

    det, rec = FaceDetector(), FaceMatcher()
    frames = [cv2.imread(p) for p in a.images]
    if a.webcam:
        cap = cv2.VideoCapture(0)
        for _ in range(40):
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        frames = frames[::4]

    feats = []
    for f in frames:
        if f is None:
            continue
        face = largest(det(f))
        if face is not None:
            feats.append(rec.embed(f, face))
    if not feats:
        raise SystemExit("no face found in the given input")

    names, bank = [], np.zeros((0, 128), np.float32)
    if Path(a.out).exists():
        old = np.load(a.out)
        names, bank = [str(n) for n in old["names"]], old["feats"]
    names += [a.name] * len(feats)
    bank = np.vstack([bank, np.stack(feats)])
    np.savez(a.out, names=np.array(names), feats=bank)
    print(f"enrolled {len(feats)} sample(s) for '{a.name}' -> {a.out} ({len(set(names))} people total)")


if __name__ == "__main__":
    main()
