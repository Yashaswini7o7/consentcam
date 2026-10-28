# Stage I numbers

All of this ran on a 2 vCPU cloud VM (Intel Xeon @ 2.8 GHz), OpenCV 5.0 DNN on CPU, no GPU, no accelerator. So these are the *worst case* numbers. The point of the UGen300 in Stage II is to move the three networks off the CPU.

## 1. Plain blur is not anonymisation

This was the most useful thing we learned in Stage I, and it changed our default.

We first used a Gaussian blur, like most face-blur tools do. It looks convincing to a human. Then we wrote `audit.py`, which asks a face-recognition model (SFace) whether the redacted face still matches the original. For one in five faces, it did.

Test set: 13 group photos from the CMU/MIT frontal face test set, 104 faces in total.

| redaction style | faces | still re-identifiable | leak |
|---|---|---|---|
| Gaussian blur (kernel = 1/3 of face width) | 104 | 21 | 20.2% |
| pixelate, 7 blocks across the face | 104 | 4 | 3.8% |
| solid fill | 104 | 1 | 1.0% |

Reproduce: `bash scripts/get_test_images.sh && python scripts/compare_styles.py data/group_photos`

Pixelation is the default now. Solid fill is safest but looks bad on video, so it's an option. Most of the remaining pixelate leaks are faces our live detector didn't find at all (tiny, side-on or partly hidden). That's a detector problem, not a redaction one, and it's the first thing we want to fix with the bigger SCRFD model on the Hailo chip.

A small caveat on the method: SFace is our auditor *and* our matcher, so this measures "can this kind of model still recognise you", not "can any model". It's a proxy. We'll add a second, unrelated recogniser (ArcFace) as an auditor in Stage II.

## 2. Moving footage

`scripts/make_test_clip.py` turns a group photo (57 people) into a 6 second, 960x540 handheld-style clip with simulated motion blur every 17 frames. One person was enrolled from a single cropped photo.

| metric | value |
|---|---|
| faces detected per frame (avg) | 28 |
| bystander faces still re-identifiable (every 5th frame audited, 822 faces) | 7 (0.85%) |
| blurs kept alive by the tracker after the detector lost the face | 309 |
| enrolled person shown un-redacted | 93 of the 97 frames where they were recognisable in the raw clip (we need 2 matches in a row before un-redacting anyone, so there is a short delay each time they come back into frame) |
| latency p50 / p95, faces + text | 84 ms / 165 ms |
| throughput, faces + text | ~11.5 fps |
| throughput, faces only | ~12.8 fps |

Those 309 "held" blurs are the tracker doing its job: each one is a frame where the detector missed a face and, without the hold logic, a stranger would have flashed on screen for 40 ms.

## 3. Where the time goes (per frame, same VM)

| step | cost |
|---|---|
| YuNet face detection, 960x540 | ~25 ms |
| SFace, per face | ~13 ms |
| PP-OCR text detection, 480x480 | ~150 ms (run every 5th frame) |

Our first version ran SFace on every unconfirmed face every frame. With 28 faces that's ~360 ms and the pipeline crawled at under 2 fps. Two changes fixed most of it: a budget of 4 recognitions per frame (largest faces first, since the creator is usually closest to the camera), and exponential back-off for faces that already failed to match. Strangers stay blurred regardless, so there's nothing to lose by checking them less often. That took us from 1.9 fps to 11.5 fps.

On the Hailo-10H the detection networks should stop being the bottleneck entirely. We'll publish those numbers in Stage II, measured, not estimated.
