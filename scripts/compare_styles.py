"""Reproduces the blur vs pixelate vs solid table in docs/results.md.

    bash scripts/get_test_images.sh
    python scripts/compare_styles.py data/group_photos
"""
import subprocess
import sys
import json
from pathlib import Path

folder = Path(sys.argv[1])
imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg"})
root = Path(__file__).resolve().parent.parent
print(f"{'style':<10}{'faces':>7}{'re-identifiable':>17}{'leak %':>9}")
for style in ("blur", "pixelate", "solid"):
    tot = leak = 0
    for p in imgs:
        r = subprocess.run([sys.executable, str(root / "audit.py"), "--source", str(p), "--style", style],
                           capture_output=True, text=True, cwd=root)
        d = json.loads(r.stdout)
        tot += d["bystander_faces"]
        leak += d["still_identifiable"]
    print(f"{style:<10}{tot:>7}{leak:>17}{100 * leak / max(tot, 1):>8.1f}%")
