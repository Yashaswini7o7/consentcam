"""Turn a still group photo into a shaky 'handheld walk' clip.

Not a substitute for real footage, but it lets us test the tracker, the
hold-on-miss logic and frame rate on any machine without filming anyone.
Every ~17 frames we smear two frames horizontally to mimic motion blur,
which is exactly when the face detector tends to drop faces.

    python scripts/make_test_clip.py data/group_photos/class57.png walk.mp4
"""
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
img = cv2.imread(src)
H, W = img.shape[:2]
OW, OH = 960, 540
vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), 25, (OW, OH))
for t in range(150):
    s = 0.85 + 0.1 * np.sin(t / 40)
    cw, ch = min(W, int(OW * s)), min(H, int(OH * s))
    x = int((W - cw) * (0.5 + 0.45 * np.sin(t / 30)))
    y = int((H - ch) * (0.5 + 0.4 * np.cos(t / 45)))
    fr = cv2.resize(img[y:y + ch, x:x + cw], (OW, OH))
    if t % 17 in (0, 1):
        fr = cv2.blur(fr, (9, 1))
    vw.write(fr)
vw.release()
print("wrote", dst)
