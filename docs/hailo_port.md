# Stage II plan: moving to the UGen300

Nothing in this file has been run on the real hardware yet. It's the plan, and we'll replace each "planned" with a measured number once the accelerator arrives.

## Which UGen300

We picked the **USB** version. The person we're building for is a creator with a laptop or a mini PC running OBS, not someone who's going to open up a machine to fit an M.2 card. Plug it in, the redaction moves off the CPU, unplug it and it still works (slower) on the CPU path. The code keeps both paths so that fallback is always there.

## Model mapping

| job | Stage I (CPU, OpenCV DNN) | Stage II (Hailo-10H) |
|---|---|---|
| find faces | YuNet | SCRFD (2.5G or 10G), compiled to .hef |
| who is it | SFace, 128-d | ArcFace MobileFaceNet, 512-d |
| find text / plates | PP-OCRv3 DB detector | same DB detector compiled, plus a dedicated plate detector |
| track + decide + redact | Python on CPU | unchanged, stays on CPU |

SCRFD and ArcFace MobileFaceNet are both in Hailo's public model zoo for earlier Hailo chips. We still need to confirm which are already compiled for the 10H and which we have to compile ourselves with the Dataflow Compiler. If the DB text detector gives the compiler trouble we'll drop to a plate-only detector first, since plates matter more than random signs.

## What changes in the code

Only `consentcam/detectors.py`. Each class has one job (`__call__` returns boxes, `embed` returns a vector), so a `HailoFaceDetector` with the same interface slots straight in. The tracker, the identity voting, the back-off and the redaction don't know or care where the boxes came from.

The one real piece of work is making the detector and the recogniser run as a pipeline on the chip (detect on frame N while embedding faces from frame N-1) instead of one after the other. HailoRT supports running several networks; we want to measure whether it beats simple sequential calls before adding the complexity.

## What we'll measure

- leak rate with the same `audit.py`, plus a second recogniser as auditor
- fps and p95 latency at 1080p30, the most common creator setting
- CPU usage of the host laptop with and without the stick
- power: the chip's typical figure is 2.5 W; we want the whole system number, measured at the wall

Target: 1080p at 30 fps with p95 latency under 50 ms, leak rate at or under the Stage I CPU number. If the bigger SCRFD model fits in the time budget, the leak rate should go down, because most of our current leaks are faces the small detector never found.
