"""Fetch the three ONNX models from the OpenCV model zoo (Apache-2.0)."""
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    "yunet.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "db.onnx": f"{BASE}/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
}

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dst = out / name
    if dst.exists() and dst.stat().st_size > 100_000:
        print(f"{name}: already there")
        continue
    print(f"{name}: downloading...")
    urllib.request.urlretrieve(url, dst)
    print(f"  {dst.stat().st_size / 1e6:.1f} MB  sha1 {hashlib.sha1(dst.read_bytes()).hexdigest()[:12]}")
