import numpy as np

from consentcam.pipeline import ConsentCam, Settings
from consentcam.redact import apply



def test_pixelate_changes_region_only():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = img.copy()
    apply(out, (40, 40, 20, 20), "pixelate", pad=0)
    assert not np.array_equal(out[40:60, 40:60], img[40:60, 40:60])
    assert np.array_equal(out[:30], img[:30])


def test_blank_frame_runs():
    cam = ConsentCam(Settings())
    out = cam.process(np.zeros((360, 640, 3), np.uint8))
    assert out.shape == (360, 640, 3)
    assert cam.stats.frames == 1
