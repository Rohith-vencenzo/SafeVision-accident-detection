# Scoring, thresholds and how to tune them

**Read this before you quote a number from this project.** The thresholds in
`config.yaml` are a *documented starting point* calibrated on scripted
scenarios. They are not scientifically optimal, they are not validated on a
labelled accident dataset, and this document exists so that you replace them
with measurements from your own footage before you present any accuracy figure.

---

## 1. What the score actually is

```
accident_score  = Σ (weight_i · component_i) / Σ weight_i   ×  penalty_factor
physical_score  = the same mean WITHOUT the temporal component
```

| Component | What it measures | Default weight |
|---|---|---|
| `collision` | multi-signal collision evidence (Phase 8) | 0.34 |
| `motion` | sudden stop / turn / acceleration (Phase 6) | 0.16 |
| `trajectory` | path deviation, crossing, convergence (Phase 7) | 0.08 |
| `temporal` | fused multi-frame evidence (Phase 9) | 0.30 |
| `object_evidence` | how many vehicles / people are involved | 0.07 |
| `scene_evidence` | disruption, blockage, instability, duration | 0.05 |

A component that could not be measured in a frame is dropped from *both* the
numerator and the denominator (it is listed in `missing_components`), so a
missing signal never silently pulls the score down.

`physical_score` exists because a crash produces a **burst** of physical
evidence followed by a quiet aftermath. The state machine therefore uses:

* `physical_score` (peak, held with a decay) → *is this event worth verifying?*
* `temporal_score` (fused window) → *is this event confirmed?*

---

## 2. The decision gates

```
NORMAL ──physical ≥ suspicious_score for 2 frames──> SUSPICIOUS
SUSPICIOUS ──physical ≥ verify_entry_score for 3 frames──> VERIFYING
VERIFYING ──ALL of: ──────────────────────────────────> CONFIRMED_ACCIDENT
    • evidence_sufficient        (≥ min_vehicles, ≥ min_signals, no blocking negatives)
    • temporal_score ≥ temporal_confirm_score
    • peak physical evidence ≥ confirm_score
    • ≥ min_verification_seconds in VERIFYING
    • ≥ min_consecutive_frames of continuous temporal support
VERIFYING ──evidence gone, or verify_timeout_seconds──> FALSE_ALARM ──> NORMAL
```

Defaults: `suspicious 0.30`, `verify_entry 0.34`, `confirm 0.42`,
`temporal_confirm 0.40`, `min_vehicles 2`, `min_signals 2`,
`min_verification_seconds 1.0`, `verify_timeout 8.0`, `cooldown 12.0`.

Raise `confirm_score` and `temporal_confirm_score` to reduce false alarms at the
cost of missed events; lower them the other way. `min_vehicles` and
`min_signals` are the structural guards - raising them makes a confirmation
physically impossible in ambiguous cases.

---

## 3. The false-alarm traps the shipped configuration is checked against

`ai_engine/detection/scenarios.py` is the executable specification. Run it after
**any** change to a weight or threshold:

```bat
python scripts\validate_thresholds.py
```

| Scenario | Must | What it models |
|---|---|---|
| `head_on_crash` | ALARM | two vehicles, same lane, impact, both stop |
| `rear_end_crash` | ALARM | follower brakes hard, lead vehicle pushed |
| `t_bone_crash` | ALARM | perpendicular impact |
| `hard_brake` | quiet | sudden stop at a junction, no other vehicle |
| `opposite_lanes` | quiet | vehicles cross the image plane without interacting |
| `following_close` | quiet | tailgating at a constant gap |
| `parked_cars` | quiet | static vehicles, heavily overlapping in the image |
| `pedestrian_crossing` | quiet | person walking across normal traffic |
| `single_frame_ghost` | quiet | one-frame false detection |

Reference run on the development machine:

```
scenario               expect    got    peak   event  severity  veh  sig  result
following_close         quiet  quiet   0.033   0.000         -    0    0  PASS
hard_brake              quiet  quiet   0.284   0.364         -    0    0  PASS
head_on_crash           ALARM  ALARM   0.500   0.554    MEDIUM    2    3  PASS
opposite_lanes          quiet  quiet   0.033   0.000         -    0    0  PASS
parked_cars             quiet  quiet   0.009   0.000         -    0    0  PASS
pedestrian_crossing     quiet  quiet   0.033   0.000         -    0    0  PASS
rear_end_crash          ALARM  ALARM   0.527   0.557    MEDIUM    2    2  PASS
single_frame_ghost      quiet  quiet   0.148   0.000         -    0    0  PASS
t_bone_crash            ALARM  ALARM   0.444   0.521       LOW    2    3  PASS
9/9 scenarios behave as configured
```

The margin to watch: the weakest crash (T-bone, 0.444) versus the strongest
quiet case (`hard_brake`, 0.284 peak / 0.364 held event score). If you change a
weight and that gap disappears, you have made the system worse.

### What this suite does and does not prove

* **Does**: the decision logic (tracking → motion → trajectory → collision →
  temporal → verification) is regression-tested, and the six classic
  false-alarm traps stay quiet.
* **Does not**: prove any detection accuracy. The scenarios feed *scripted
  detections* into the pipeline; they do not test YOLO. A separate real-imagery
  check exists for that:

  ```bat
  python scripts\real_yolo_check.py --photo your_road_photo.jpg
  python scripts\real_yolo_check.py --video --frames 60      # pan/zoom a real photo
  python scripts\e2e_check.py --photo your_road_photo.jpg     # full acceptance run
  ```

### A note on global-motion compensation

`motion.camera_motion_*` implements camera-motion compensation for panning / PTZ
cameras: when at least `camera_motion_min_tracks` (default 3) objects agree on
the same image velocity, that shared motion is estimated and subtracted from
every track before the motion features are computed.

The `min_tracks = 3` rule matters. With only one or two objects, "they agree" is
meaningless: two cars travelling in the same direction on a straight road look
exactly like a panning camera, and compensating for that would erase the real
motion evidence and cause missed accidents. If your footage is a PTZ camera with
few objects, raise `camera_motion_min_tracks` rather than lowering the
agreement threshold, and check the effect with `scripts/real_yolo_check.py`.

---

## 4. Calibrating on your own footage

```bat
:: 1. collect labelled clips
::    calibration/accident/*.mp4     confirmed accidents
::    calibration/normal/*.mp4       ordinary traffic (the hard half)

:: 2. measure
python scripts\calibrate.py --accident-dir calibration\accident \
                            --normal-dir   calibration\normal

:: 3. read the report
```

It prints, per clip, the peak score / peak physical evidence / peak temporal
score, then the distribution per class and the threshold that maximises
`TPR − FPR` on *your* data, plus a list of any false alarms it produced.

```bat
:: 4. apply (review the diff afterwards!)
python scripts\calibrate.py ... --write config.yaml
```

### Clip selection matters more than the threshold

The `normal` folder is where the truth lives. Include:

* parked cars and cars parked in a row, overlapping in the image
* vehicles in adjacent lanes passing at close range
* queueing at a traffic light (everyone stops, then everyone goes)
* motorcycles and bicycles weaving through traffic
* pedestrians on the pavement and at crossings
* night, rain, glare, and camera compression artefacts
* **panning / PTZ cameras** (the engine estimates and compensates global camera
  motion, but it is worth measuring how well it does on your footage)

A calibration set of 20 accident clips and 100 normal clips is far more useful
than 200 accident clips and 10 normal clips.

### Report both numbers

Never report a detection rate alone. The pair that matters is

```
detection rate on accidents  |  false alarms per hour of normal traffic
```

and the false-alarm number should be measured on **hours** of normal footage, not
on clip counts, because that is how the system will actually be operated.

---

## 5. Common adjustments

| Symptom | First thing to try | Why |
|---|---|---|
| Too many false alarms | raise `verification.confirm_score`, `temporal_confirm_score` | stricter final gates |
| Too many false alarms (objects close) | lower `collision.context_weight`, raise `collision.min_signals` | geometric context is the weakest evidence |
| Vehicles in opposite lanes trigger it | raise `collision.time_to_impact_window` | far-away approach is not closing |
| Missed gentle/slow collisions | lower `collision.min_signals` to 1 *and* raise `collision.physical_weight` | trade agreement for sensitivity |
| Missed night/low-light events | lower `model.conf` to 0.2, raise `model.imgsz` to 960 | detector recall, not the fusion |
| Track ids swap near the impact | lower `tracking.recovery_distance`, raise `tracking.match_threshold_low` | more permissive second association stage |
| It reacts to a panning camera | check `motion.camera_motion_agreement`; lower to 0.5 | more aggressive global-motion compensation |
| It reacts to a panning camera | raise `motion.camera_motion_min_tracks` to 4–5 | only compensate when many objects agree |
| Pedestrians alone push the score | lower `motion.person_fallback_scale` | pedestrians alone are weak evidence |
| Every event is "confirmed" but late | lower `verification.min_verification_seconds` | faster decision, slightly higher risk |
| Duplicate reports for one crash | raise `verification.cooldown_seconds` | one incident per event |

---

## 6. Reproducing the shipped calibration

```bat
python scripts\validate_thresholds.py --json calibration_report.json
python -m unittest tests.test_pipeline_e2e -v     # includes the same 9 scenarios
```

Both run offline, without YOLO weights and without a network connection.

---

## 7. A note on units

Every distance is expressed in **frame diagonals** and every speed in **frame
diagonals per second**, never in pixels. That is why the same thresholds work at
640×360 and 1920×1080. If you resize the stream, nothing needs re-tuning; if you
crop instead, re-run the validation, because cropping changes the frame
diagonal and therefore the scale of every normalised measurement.
