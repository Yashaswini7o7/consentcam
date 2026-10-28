"""Per-frame logic. Read this file first if you're new to the repo.

frame
  -> YuNet faces ------> IoU tracker (with hold) --> who is enrolled? (SFace, every N frames)
  -> DB text regions (every K frames) --> region memory
  -> blur every face that is NOT confirmed + every text region
  -> out
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import redact
from .detectors import FaceDetector, FaceMatcher, TextDetector
from .tracker import IoUTracker, RegionMemory


@dataclass
class Settings:
    whitelist: str | None = None
    redact_text: bool = True
    text_every: int = 5        # run the text detector on every 5th frame
    recheck_every: int = 10    # re-verify identity of allowed faces every 10 frames
    confirm: int = 2           # consecutive matches needed before a face is un-blurred
    embed_budget: int = 4      # max face-recognition calls per frame (largest faces first)
    hold: int = 8              # frames to keep blurring a face the detector lost
    style: str = "pixelate"   # see README: plain blur leaks identity
    debug: bool = False        # draw boxes (never use for a real stream)


@dataclass
class Stats:
    frames: int = 0
    faces_seen: int = 0
    faces_blurred: int = 0
    faces_allowed: int = 0     # enrolled people shown un-redacted
    held_blurs: int = 0
    text_regions: int = 0
    latency_ms: deque = field(default_factory=lambda: deque(maxlen=500))

    def summary(self):
        lat = np.array(self.latency_ms) if self.latency_ms else np.array([0.0])
        return {
            "frames": self.frames,
            "faces_seen": self.faces_seen,
            "faces_blurred": self.faces_blurred,
            "faces_allowed": self.faces_allowed,
            "held_blurs": self.held_blurs,
            "text_regions": self.text_regions,
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
            "fps_est": round(1000.0 / max(float(np.mean(lat)), 1e-3), 1),
        }


class ConsentCam:
    def __init__(self, s: Settings | None = None):
        self.s = s or Settings()
        self.faces = FaceDetector()
        self.matcher = FaceMatcher()
        if self.s.whitelist:
            self.matcher.load(self.s.whitelist)
        self.text = TextDetector() if self.s.redact_text else None
        self.tracker = IoUTracker(hold=self.s.hold)
        self.regions = RegionMemory(hold=self.s.text_every * 2)
        self.stats = Stats()
        self._n = 0

    def _due(self, track) -> bool:
        if track.allowed:
            wait = self.s.recheck_every
        else:
            # strangers stay blurred no matter what, so there's no rush to keep
            # re-checking them. Back off 1, 2, 4, 8, 16 frames. The creator is
            # usually the biggest face and gets checked first anyway.
            wait = min(2 ** track.fails, 16)
        return self._n - track.last_check >= wait

    def _verify(self, frame, track):
        """Decide whether a tracked face belongs to an enrolled person."""
        track.last_check = self._n
        name, _ = self.matcher.identify(self.matcher.embed(frame, track.payload))
        if name is None:
            # any miss resets trust. Costs the creator a few frames of blur
            # now and then; much better than leaking a stranger.
            track.identity, track.streak, track.pending = None, 0, None
            track.fails += 1
            return
        track.fails = 0
        track.streak = track.streak + 1 if track.pending == name else 1
        track.pending = name
        if track.streak >= self.s.confirm:
            track.identity = name

    def process(self, frame):
        t0 = time.perf_counter()
        self._n += 1
        out = frame.copy()

        dets = self.faces(frame)
        tracks = self.tracker.update([d.box for d in dets], dets)
        if self.matcher.bank is not None:
            due = [t for t in tracks if t.payload is not None and self._due(t)]
            due.sort(key=lambda t: t.box[2] * t.box[3], reverse=True)
            for t in due[: self.s.embed_budget]:
                self._verify(frame, t)

        if self.text is not None:
            if self._n % self.s.text_every == 1 or self.s.text_every == 1:
                found = self.text(frame)
                self.regions.update(found)
                self.stats.text_regions += len(found)
            for b in self.regions.tick():
                redact.apply(out, b, self.s.style, pad=0.1)

        for t in tracks:
            if t.allowed and t.misses == 0:
                self.stats.faces_allowed += 1
                continue
            redact.apply(out, t.box, self.s.style)
            self.stats.faces_blurred += 1
            if t.misses:
                self.stats.held_blurs += 1

        if self.s.debug:
            for t in tracks:
                col = (60, 200, 60) if t.allowed else (40, 40, 230)
                redact.outline(out, t.box, col, t.identity or f"#{t.tid}")

        self.stats.frames += 1
        self.stats.faces_seen += len(dets)
        self.stats.latency_ms.append((time.perf_counter() - t0) * 1000)
        return out
