# Camera sync at deploy time — current status and remaining work

> **Status (2026-05-03)** — Code-only instrumentation is done, and
> the first on-rig arrival-time probe has run successfully. The old
> hypothesis "Femto arrives 0.5-2.5 s behind D405 at deploy time" is
> **not supported at the host-arrival level**: the measured Femto/D405
> arrival gap is millisecond-scale. The only remaining blocker before
> closing or escalating this issue is Step 2: visual content-lag
> measurement with an LED blink or fast-motion event.

## Current Verdict

- `bag_align.py` is still the correct offline alignment script for the
  current recording pipeline. It matches Femto frames to scalar wall
  time by `sys_ts_us`, not by Femto `hw_ts_us`.
- The old `hw_ts_us` matching bug is fixed. Matching by `hw_ts_us`
  previously made `femto_aligned.mp4` play about 3.7 s ahead of D405.
- Live deploy now has timing observability in both the MVP and full
  scripts, but it still does not actively pair/drop frames. Given the
  first probe's arrival-gap result, that is currently appropriate.
- Do **not** add deploy ring buffers, D405 delay, or training-time
  lag augmentation unless the visual content-lag test proves a real
  content gap.

## First On-Rig Probe — 2026-05-03

Recorded on the live rig with both cameras connected:

- Femto Mega: Ethernet `192.168.0.15`; ping checked at 1.09 ms.
- D405: USB detected as `Intel(R) RealSense(TM) Depth Camera 405`.

Command:

```bash
PYTHONPATH=src:deployment python3 deployment/scripts/camera_sync_probe.py \
  --duration-s 30 \
  --poll-hz 60 \
  --femto-fps 15 \
  --d405-fps 15 \
  --out /tmp/cam_probe_20260503_003431
```

Results from `/tmp/cam_probe_20260503_003431/summary.txt`:

| Metric | p5 | p50 | p95 | Notes |
|---|---:|---:|---:|---|
| Femto interarrival (ms) | 48.8 | 68.4 | 83.2 | 451 unique frames, about 14.6 fps |
| D405 interarrival (ms) | 66.4 | 66.7 | 67.5 | 451 unique frames, very stable 15 fps |
| Pair gap `D405.arrival - nearest Femto.arrival` (ms) | -17.8 | -1.2 | +13.5 | max absolute gap 64.8 ms |
| Apparent `Femto.sys_ts - Femto.hw_ts` (ms) | 6807.5 | 6817.2 | 6829.0 | narrow offset, mostly clock-domain offset |

Interpretation:

- Host arrival is effectively synchronized. Median arrival gap is
  about 1.2 ms, and p5-p95 sits within roughly +/-18 ms.
- No cached-frame reuse was observed. Both cameras produced 451 unique
  `frame_id`s over 1786 polls.
- The large `sys_ts_us - hw_ts_us` value is not evidence of a 6.8 s
  pipeline buffer. Because `hw_ts_us` is a free-running firmware clock,
  this value mostly reflects clock-domain offset.
- Step 1 therefore weakens the original train-vs-deploy lag hypothesis
  substantially, but does not fully close it because arrival time is
  not the same thing as visual content time.

## Remaining Required Validation

### P0: Measure Visual Content Lag

Run a shared visual event test where both cameras can see the same
event:

- Preferred: LED blink in both views, around 2 Hz for about 30 s.
- Alternative: fast hand wave or motion target visible in both views.

Then run:

```bash
python3 deployment/scripts/camera_visual_lag.py \
  --femto /tmp/led_femto.mp4 \
  --d405 /tmp/led_d405.mp4 \
  --mode brightness \
  --roi <X> <Y> <W> <H> \
  --out /tmp/visual_lag_$(date +%Y%m%d_%H%M%S)
```

Decision rule from `summary.json`:

- `abs(lag_ms_p50) < 50`: cross-camera content gap is not the issue.
  Do not implement sync mitigations for this hypothesis.
- `abs(lag_ms_p50)` in the 200-2000 ms range: there is a real content
  lag large enough to investigate mitigation.
- Very few matched pairs or unstable signs: the visual event/ROI is
  not clean enough; repeat the recording before drawing conclusions.

Tool caveats:

- `camera_visual_lag.py` only measures content lag from the videos you
  give it. If the two MP4s were started independently with arbitrary
  offsets, frame-index timing can be misleading. Prefer simultaneous
  recording, or provide timestamp CSVs that correctly map video frames
  to host time.
- The current `--roi` is one rectangle applied to both videos. If the
  LED appears at different pixel coordinates in Femto and D405, use a
  large/full-frame ROI, crop the videos first, use `--mode motion`, or
  extend the script with separate `--femto-roi` and `--d405-roi`.

## Conditional Fixes After Step 2

Only choose a mitigation after the visual content-lag result is known:

- If visual lag is small and train/deploy match, close this camera-sync
  hypothesis and look elsewhere for policy divergence.
- If live deploy has larger content lag than the sys-ts-aligned
  training data, consider active frame pairing by `arrival_host_ns`,
  dropping stale observations, or reducing Femto pipeline latency.
- If lag is unavoidable and consistently present, consider synthetic
  third-person delay augmentation during training.
- Do not slow D405 by about 1 s to match Femto except as a temporary
  experiment. It deliberately makes the control loop stale.

## What Was Fixed In Code

| Area | Current state |
|---|---|
| `deployment/scripts/bag_align.py` header | Fixed. It now documents `sys_ts_us` matching and warns that `sys_ts_us - hw_ts_us` mixes SDK buffering with clock-domain offset. |
| `deployment/openwam/camera/femto_mega.py` | Fixed for observability. `CameraFrame` carries `arrival_host_ns` and `sys_timestamp_us`; Femto stamps `arrival_host_ns` only when a new SDK frame is acquired. |
| `deployment/openwam/camera/d405.py` | Fixed for observability. D405 stamps `arrival_host_ns` after `wait_for_frames()` returns a fresh frame. |
| `deployment/openwam/camera/sync.py` | Fixed. `CameraGrabThread` skips repeated `frame_id`s and prefers `frame.arrival_host_ns`, so cached Femto reads no longer look fresh. |
| `deployment/scripts/deploy_parallel_stream_fr3_serial.py` and `deployment/scripts/deploy_parallel_stream_fr3_parallel.py` | Fixed for logging. `capture_obs()` records frame metadata, warns on cached-frame reuse and large arrival gaps, and prints periodic `[obs]` lines. |
| `deployment/scripts/camera_sync_probe.py` | Added. Step-1 arrival-time probe; already run successfully on 2026-05-03. |
| `deployment/scripts/camera_visual_lag.py` | Added. Step-2 visual content-lag analysis tool; waiting for an LED/motion recording. |

## Clock Semantics

- `hw_ts_us`: Femto firmware clock. It is not directly comparable with
  host wall time or host monotonic time.
- `sys_ts_us`: Orbbec SDK system timestamp. `bag_align.py` uses this
  host-comparable timestamp to match `scalars["timestamps"]`.
- `arrival_host_ns`: host `time.monotonic_ns()` stamped when the camera
  wrapper receives a new frame. Repeated cached reads of the same
  `frame_id` keep the same `arrival_host_ns`.
- `frame_id`: per-camera counter. Use it for de-dup within one camera;
  never match Femto and D405 by cross-camera `frame_id`.

Do not subtract `arrival_host_ns` from `sys_ts_us` as an absolute
timestamp comparison. They use different host clock bases. Use
`arrival_host_ns` for live interarrival/cross-camera arrival-gap
analysis. Treat `sys_ts_us - hw_ts_us` as an apparent offset diagnostic,
not pure visual latency.

## Bag Align: Correct Usage

The intended offline path remains:

```text
teleop recording
  -> femto.bag + d405.mp4 + scalars.npz
  -> deployment/scripts/bag_align.py
  -> femto_aligned.mp4 + d405_aligned.mp4 + scalars_aligned.npz + bag_align.json
  -> the LeRobot v2.1 converter under deployment/scripts/
```

Use `bag_align.json` as the quality record. Episodes with large
`max_match_delta_us`, large `max_d405_match_delta_us`, missing
`bag_align.json`, or catastrophic edge fill should not be silently
converted.

## Known Non-Solutions

- Do not match Femto and D405 by cross-camera `frame_id`.
- Do not treat `sys_ts_us - hw_ts_us` as pure SDK buffer latency.
- Do not use the old `sync.py` behavior from before `e18c71f` for
  latency claims; it re-stamped repeated cached frames.
- Do not assume `bag_align.py` fixes live deploy. It is an offline
  post-processing script.
