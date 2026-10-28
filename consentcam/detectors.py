"""Model wrappers.

Everything here talks to OpenCV's DNN module for now. On the UGen300 the same
three networks get compiled to .hef files and run through HailoRT instead
(see docs/hailo_port.md). We kept the interfaces small on purpose so that
swap is one class per model and nothing above this file has to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    raw: np.ndarray  # yunet row (15 values), needed by SFace for alignment


class FaceDetector:
    """YuNet. Tiny (~230 KB), fast, and good at small faces in crowds."""

    def __init__(self, model=MODEL_DIR / "yunet.onnx", score=0.5, nms=0.3):
        self.net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score, nms, 5000)
        self._size = None

    def __call__(self, frame) -> list[Face]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.net.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self.net.detect(frame)
        if rows is None:
            return []
        out = []
        for r in rows:
            x, y, bw, bh = (int(v) for v in r[:4])
            out.append(Face((x, y, bw, bh), float(r[14]), r))
        return out


class FaceMatcher:
    """SFace embeddings + cosine similarity against the enrolled list.

    0.363 is the cosine threshold the SFace authors publish for LFW. We use a
    slightly stricter default because a false "this is the creator" means a
    stranger goes un-blurred, which is the one mistake we really can't make.
    """

    def __init__(self, model=MODEL_DIR / "sface.onnx", threshold=0.42):
        self.rec = cv2.FaceRecognizerSF.create(str(model), "")
        self.threshold = threshold
        self.names: list[str] = []
        self.bank: np.ndarray | None = None  # (N, 128), L2-normalised

    def embed(self, frame, face: Face) -> np.ndarray:
        crop = self.rec.alignCrop(frame, face.raw)
        f = self.rec.feature(crop).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def load(self, path):
        data = np.load(path, allow_pickle=False)
        self.names = [str(n) for n in data["names"]]
        self.bank = data["feats"].astype(np.float32)

    def identify(self, emb) -> tuple[str | None, float]:
        if self.bank is None or len(self.bank) == 0:
            return None, 0.0
        sims = self.bank @ emb
        i = int(np.argmax(sims))
        s = float(sims[i])
        return (self.names[i], s) if s >= self.threshold else (None, s)


class TextDetector:
    """PP-OCRv3 DB text detector. Catches signboards, screens, documents and,
    as a side effect, number plates (a plate is just short text on a box).
    We only need *where* text is, never what it says, so there's no OCR stage.
    """

    def __init__(self, model=MODEL_DIR / "db.onnx", size=(480, 480)):
        m = cv2.dnn_TextDetectionModel_DB(str(model))
        m.setInputParams(1.0 / 255.0, size, (122.67891434, 116.66876762, 104.00698793))
        m.setBinaryThreshold(0.3)
        m.setPolygonThreshold(0.5)
        m.setMaxCandidates(200)
        m.setUnclipRatio(2.0)
        self.m = m
        self.size = size

    def __call__(self, frame) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self.size)
        polys, _ = self.m.detect(resized)
        sx, sy = w / self.size[0], h / self.size[1]
        boxes = []
        for p in polys:
            p = np.array(p, dtype=np.float32)
            p[:, 0] *= sx
            p[:, 1] *= sy
            x, y, bw, bh = cv2.boundingRect(p.astype(np.int32))
            if bw * bh > 60:  # ignore specks
                boxes.append((x, y, bw, bh))
        return boxes
