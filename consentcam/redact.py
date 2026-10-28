import cv2
import numpy as np


def _pad(box, frac, w, h):
    x, y, bw, bh = box
    px, py = int(bw * frac), int(bh * frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    return x0, y0, x1, y1


def apply(frame, box, style="blur", pad=0.2):
    """Redact one region in place. Faces get extra padding because YuNet's box
    is tight around the face and hair/ears are still identifying."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _pad(box, pad, w, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    if style == "pixelate":
        # fixed number of blocks across the face, not a fixed block size,
        # so a big close-up face is as unreadable as a small distant one
        n = 7
        small = cv2.resize(roi, (n, max(1, round(n * (y1 - y0) / (x1 - x0)))), interpolation=cv2.INTER_AREA)
        frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    elif style == "solid":
        frame[y0:y1, x0:x1] = roi.mean(axis=(0, 1)).astype(np.uint8)
    else:
        # kernel scales with the region, otherwise big faces stay recognisable
        k = max(15, ((x1 - x0) // 3) | 1)
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def outline(frame, box, color, label=None):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
