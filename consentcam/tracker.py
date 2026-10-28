"""A deliberately boring IoU tracker.

Why not ByteTrack / DeepSORT? We tried ByteTrack first. It's better at keeping
IDs through occlusion, but here we care about something different: the blur
must never blink off for even one frame when the detector misses a face
(motion blur, a head turn, someone walking past a lamp). So the important part
below is `hold`: a lost track keeps getting blurred at its last known spot for
a few frames. Fail closed, not open.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    box: tuple
    hits: int = 1
    misses: int = 0
    # identity voting: a face is only treated as "allowed" after it has
    # matched the same enrolled person `confirm` times in a row
    identity: str | None = None
    streak: int = 0
    pending: str | None = None
    fails: int = 0            # failed identity checks, drives the back-off
    last_check: int = -999
    payload: object = field(default=None, repr=False)

    @property
    def allowed(self) -> bool:
        return self.identity is not None


class IoUTracker:
    def __init__(self, iou_thr=0.3, hold=8):
        self.iou_thr = iou_thr
        self.hold = hold
        self.tracks: list[Track] = []
        self._next = 0

    def update(self, boxes, payloads=None):
        payloads = payloads or [None] * len(boxes)
        unmatched = set(range(len(boxes)))
        # greedy matching, best IoU first. Fine for < ~50 faces per frame.
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t = set()
        for score, ti, bi in pairs:
            if score < self.iou_thr:
                break
            if ti in used_t or bi not in unmatched:
                continue
            t = self.tracks[ti]
            t.box, t.payload = boxes[bi], payloads[bi]
            t.hits += 1
            t.misses = 0
            used_t.add(ti)
            unmatched.discard(bi)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
                t.payload = None
        self.tracks = [t for t in self.tracks if t.misses <= self.hold]

        for bi in unmatched:
            self.tracks.append(Track(self._next, boxes[bi], payload=payloads[bi]))
            self._next += 1
        return self.tracks


class RegionMemory:
    """Same idea for text regions, which we only detect every few frames."""

    def __init__(self, hold=6):
        self.hold = hold
        self.regions: list[tuple[tuple, int]] = []

    def update(self, boxes):
        self.regions = [(b, self.hold) for b in boxes]

    def tick(self):
        self.regions = [(b, n - 1) for b, n in self.regions if n > 0]
        return [b for b, _ in self.regions]
