# ConsentCam

Real-time, on-device redaction for people who film in public. You stay visible. Everyone who didn't agree to be in your video gets pixelated before the frame ever leaves your machine, along with number plates, phone screens and other readable text.

Built for the ASUS UGen AI League Hackathon 2026, Creator AI theme, Battlefield Lightning (ASUS UGen300 / Hailo-10H).

## Why

Anyone who vlogs in a market, a metro, a college fest or an office has the same problem. Strangers walk through every shot. Blurring them afterwards means scrubbing through footage frame by frame, so most people don't bother, and bystanders end up on the internet without ever being asked. Live streamers can't fix it afterwards at all.

The existing fixes either run in the cloud (so the raw footage of those strangers gets uploaded somewhere, which is the exact thing we're trying to prevent) or they're editing-software features you apply after the fact. We wanted something that runs live, on the creator's own laptop, and treats "don't leak a stranger's face" as the one rule it can't break.

## What it does

- **Enroll yourself once.** A few photos or 2 seconds of webcam. We store a 128-number face embedding, never the photo. Delete the file and you're out.
- **Everyone else is redacted by default.** A new face starts pixelated and only becomes visible after it matches an enrolled person twice in a row.
- **Text too.** Signboards, screens, documents and number plates are found and pixelated. We only detect where text is. We never read it.
- **Fails closed.** If the detector loses a face for a few frames (motion blur, head turn), the redaction stays on its last position instead of flickering off.
- **Measures itself.** `audit.py` checks whether a face recognition model can still identify anyone in the output.

## Quick start

```bash
git clone <this repo> && cd consentcam
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py

# 1. enroll yourself (and anyone else who agreed)
python enroll.py --name me --webcam
#    or: python enroll.py --name me --images me1.jpg me2.jpg

# 2. live preview from your webcam (q to quit, d to show debug boxes)
python run.py --source 0 --whitelist whitelist.npz --show

# 3. or process a recorded clip
python run.py --source walk.mp4 --whitelist whitelist.npz --out walk_safe.mp4

# 4. check how much leaks
python audit.py --source walk.mp4 --whitelist whitelist.npz
```

No camera? `bash scripts/get_test_images.sh` pulls some public group photos and `python scripts/make_test_clip.py data/group_photos/class57.png walk.mp4` turns one into a moving test clip.

## How it works

```
camera frame
   |
   +--> YuNet face detector --> IoU tracker (keeps lost faces for 8 frames)
   |                                |
   |                                +--> SFace: is this an enrolled person?
   |                                     (max 4 checks per frame, biggest faces first,
   |                                      strangers re-checked with back-off)
   |
   +--> PP-OCR text detector (every 5th frame) --> region memory
   |
   +--> pixelate: every face not confirmed + every text region
   |
   v
safe frame --> preview / file / (Stage II) OBS virtual camera
```

`consentcam/pipeline.py` is the place to start reading. It's about 150 lines.

## The thing that surprised us

We started with Gaussian blur, like most face-blur tools. It looks fine. Then we ran a face-recognition model over the blurred output and **20% of the faces could still be matched to the original person**. Pixelation (a fixed 7 blocks across each face, however big it is) brought that down to 3.8%, so that's the default now. Full numbers and how to reproduce them are in [docs/results.md](docs/results.md).

## Stage I results (2 vCPU laptop-class CPU, no accelerator)

| | |
|---|---|
| re-identifiable faces after redaction, still photos (104 faces) | 3.8% (blur: 20.2%) |
| re-identifiable faces after redaction, moving 57-person clip (822 faces audited) | 0.85% |
| speed with ~28 faces per frame, 960x540 | ~11.5 fps |

Usable on a CPU, not yet smooth. Getting to 1080p30 is what the UGen300 is for. See [docs/hailo_port.md](docs/hailo_port.md).

## Layout

```
consentcam/
  detectors.py   model wrappers (the only file that changes for Hailo)
  tracker.py     IoU tracker with hold-on-miss
  pipeline.py    per-frame logic and stats
  redact.py      pixelate / blur / solid
enroll.py        build the consent list
run.py           webcam, video or image in, redacted out
audit.py         re-identification leak test
scripts/         model download, test images, test clip, style comparison
tests/           tracker and redaction tests
docs/            results, Stage II plan
```

## Known limits

- Faces smaller than about 10 px across, or fully side-on, are often missed. Most of the remaining leaks are these.
- Twins and very similar faces could match an enrolled person. The threshold is set stricter than SFace's default for this reason, but it's still a risk.
- Text detection is the slowest part on CPU, which is why it runs every 5th frame.
- No audio. Voices and names said out loud are out of scope for now.

## Roadmap

1. Port detection and recognition to the UGen300 (USB) via HailoRT.
2. Dedicated number-plate detector.
3. OBS virtual camera output, so it works with any streaming app.
4. "Consent QR": a bystander who wants to be in the video can scan a code to opt in for that session.

## Models and licences

YuNet, SFace and the PP-OCRv3 text detector come from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0). They're downloaded by the script, not stored in this repo. Our code is MIT.
