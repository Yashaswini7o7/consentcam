"""Leak audit: could a face-recognition model still tell who this is?

Our first version of this script just re-ran a face detector on the blurred
output and counted what it found. Every blurred face still got detected (a
blurred face is still a face-shaped blob), so the number was useless.

What actually matters is re-identification. So for every face the auditor
finds in the ORIGINAL frame (more sensitive settings than the live pipeline,
image upscaled 1.5x so smaller faces show up too), we take the SFace embedding
of the original face and of the same spot in the redacted frame. If the two
are still more similar than SFace's own "same person" cut-off (0.363 cosine),
we count it as a leak. Faces of enrolled people are skipped, they're meant
to stay visible.

    python audit.py --source group.png
    python audit.py --source walk.mp4 --whitelist whitelist.npz --every 5
    python audit.py --source group.png --style pixelate
"""
import argparse
import json
from pathlib import Path

import cv2

from consentcam.detectors import FaceDetector, FaceMatcher
from consentcam.pipeline import ConsentCam, Settings


SAME_PERSON = 0.363  # SFace published cosine threshold


def leaks(det, matcher, original, redacted, enrolled):
    """Returns (bystander faces in original, how many are still re-identifiable)."""
    a = cv2.resize(original, None, fx=1.5, fy=1.5)
    b = cv2.resize(redacted, None, fx=1.5, fy=1.5)
    total = leaked = 0
    for f in det(a):
        ea = matcher.embed(a, f)
        if enrolled and matcher.identify(ea)[0]:
            continue
        total += 1
        if float(ea @ matcher.embed(b, f)) >= SAME_PERSON:
            leaked += 1
    return total, leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--whitelist")
    ap.add_argument("--every", type=int, default=5, help="audit every Nth frame of a video")
    ap.add_argument("--style", default="pixelate", choices=["blur", "pixelate", "solid"])
    a = ap.parse_args()

    cam = ConsentCam(Settings(whitelist=a.whitelist, style=a.style))
    auditor = FaceDetector(score=0.45)
    m = FaceMatcher()
    if a.whitelist:
        m.load(a.whitelist)

    before = after = audited = 0
    if Path(a.source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
        cam.s.text_every = 1
        frames = [cv2.imread(a.source)]
    else:
        cap = cv2.VideoCapture(a.source)

        def gen():
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield fr

        frames = gen()

    for i, f in enumerate(frames):
        if f is None:
            break
        out = cam.process(f)
        if i % a.every:
            continue
        audited += 1
        t, l = leaks(auditor, m, f, out, bool(a.whitelist))
        before += t
        after += l

    report = {
        "frames_audited": audited,
        "bystander_faces": before,
        "still_identifiable": after,
        "leak_rate": round(after / before, 4) if before else 0.0,
        **cam.stats.summary(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
