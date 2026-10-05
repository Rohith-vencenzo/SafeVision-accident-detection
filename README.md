# SafeVision

### AI-Based Real-Time Accident Detection and Intelligent Emergency Response System

**This repository contains the AI / Computer-Vision subsystem (the 60% part).**
It reads video, decides whether a road accident is happening, and emits a
structured JSON incident. It does **not** contain a backend, a database, a
dashboard, SMS, or any call to an emergency service - that is the other 40%,
which consumes this module's output.

```
CCTV / IP CAMERA  ·  LOCAL VIDEO  ·  LAPTOP WEBCAM
                    │
             VIDEO STREAM  (ai_engine/video)
                    │
             FRAME PROCESSING
                    │
             YOLO DETECTION  (ai_engine/detection)      car · truck · bus · motorcycle · bicycle · person
                    │
             BYTETRACK  (ai_engine/tracking)            stable track ids + history
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 MOTION         TRAJECTORY       COLLISION
    │               │               │
    └───────────────┼───────────────┘
                    │
            TEMPORAL FUSION  (ai_engine/temporal)     3–10 s rolling window
                    │
            FALSE-ALARM PREVENTION + VERIFICATION STATE MACHINE
                    │
                 ACCIDENT SCORE  (ai_engine/scoring)   transparent weighted score
                    │
          ┌─────────┴──────────┐
       NORMAL              POSSIBLE
                             │
                        VERIFYING ──(evidence gone)──► FALSE_ALARM
                             │
                     CONFIRMED_ACCIDENT
                             │
        ┌────────────────────┼────────────────────┐
   SEVERITY              LOCATION              EVIDENCE
   ANALYSIS          (camera-registered)      (before/collision/after)
        └────────────────────┼────────────────────┘
                             │
                    STRUCTURED JSON OUTPUT  (ai_engine/schemas)
                             │
                    AI INCIDENT REPORT  (ai_engine/reporting)
```

---

## 1. Project overview

| | |
|---|---|
| **What it does** | Watches a road camera / video, detects vehicles and people, tracks them, and reports a **confirmed** accident only when several independent kinds of evidence agree across multiple frames. |
| **What it outputs** | A versioned JSON incident (`incident_id`, score, severity, camera-registered location, evidence paths, reasons) plus a human-readable report. |
| **What it does not do** | No emergency calls, no SMS, no backend, no dashboard, no cloud service, no paid API. |
| **Runs on** | A normal laptop, CPU only. GPU is used automatically if present. |
| **Model** | Ultralytics YOLO11 (`yolo11n.pt`, ~5 MB), downloaded once, then fully offline. |

## 2. Problem statement

Road accidents are usually reported by a passer-by or a bystander, which means
the response clock starts *after* the accident. CCTV cameras are everywhere and
record continuously, but nobody watches 24 video feeds. Naive automation makes
this worse: a rule that fires on "two vehicle boxes overlap" or "a vehicle
decelerated" produces a flood of false alarms (two cars passing in opposite
lanes, a car braking for a red light, parked cars overlapping in perspective),
and a system that cries wolf gets switched off.

The interesting problem is therefore **not** "detect a collision in one frame".
It is: **decide, from a stream of noisy detections, whether a real accident is
happening - and prove why.**

## 3. Innovation

1. **Evidence fusion instead of a single trigger.** Six measurable signals
   (proximity, overlap, closing speed, paired deceleration, post-impact
   stopping, displacement spike) plus trajectory convergence must *agree*.
   `collision.min_signals` is the gate; below it the score is suppressed.
2. **Time is a first-class dimension.** A rolling window with recency weighting,
   a time-constant "hold", a peak term and an isolated-spike penalty means one
   bad frame decays in milliseconds while a real event keeps its evidence.
3. **Explicit false-alarm reasoning.** Parked vehicles, stable scenes,
   single-frame events, unstable tracking and low detection confidence are
   collected as *negative* evidence and shown in the report.
4. **A verification state machine** (`NORMAL → SUSPICIOUS → VERIFYING →
   CONFIRMED_ACCIDENT`, or `→ FALSE_ALARM`) so a confirmation is always the
   result of a time-bounded investigation, never of a single frame.
5. **Transparent scoring.** Every score ships with its components, the weights
   used, the penalties applied and human-readable reasons - and every weight
   lives in `config.yaml`.
6. **Honest output.** `AI-estimated accident probability`, `AI-estimated scene
   severity`, `camera-registered location`. No injury detection, no invented GPS,
   no "100% accurate" claims.

## 4. AI pipeline in detail

| Phase | Module | What it contributes |
|---|---|---|
| 3 | `ai_engine/video/source.py` | file / webcam / RTSP, reconnect, resize, frame skipping, FPS bookkeeping |
| 4 | `ai_engine/detection/` | YOLO11 inference → `Detection` objects (class, confidence, box, centre, frame, timestamp) |
| 5 | `ai_engine/tracking/byte_tracker.py` | ByteTrack: Kalman prediction, two-stage association, distance recovery stage, per-track history |
| 6 | `ai_engine/motion/motion_analyzer.py` | speed, displacement, heading, acceleration, **sudden stop**, **sudden turn**, jitter |
| 7 | `ai_engine/trajectory/trajectory_analyzer.py` | straightness, heading/lateral deviation, zigzag, crossing & convergence |
| 8 | `ai_engine/collision/collision_analyzer.py` | the seven multi-signal collision signals + the agreement gate |
| 9 | `ai_engine/temporal/temporal_fusion.py` | rolling-window fusion, persistence, consistency, isolated-spike suppression |
| 10 | `ai_engine/verification/false_alarm_prevention.py` | negative evidence + the state machine |
| 11 | `ai_engine/scoring/accident_scorer.py` | the transparent weighted accident score |
| 13 | `ai_engine/severity/severity_analyzer.py` | AI-estimated **scene** severity |
| 14 | `ai_engine/location/location_manager.py` | camera-registered location metadata |
| 15 | `ai_engine/evidence/evidence_capture.py` | before / collision / after / annotated frames, optional clip, SHA-256 |
| 16 | `ai_engine/evidence/incident_id.py` | `INC-YYYYMMDD-NNNN`, crash-safe and thread-safe |
| 17 | `ai_engine/schemas/incident_result.py` | the versioned JSON contract |
| 18 | `ai_engine/reporting/incident_report.py` | the AI incident report |
| 19 | `ai_engine/visualization/overlay.py` | boxes, track ids, trails, collision links, HUD |
| 20 | `ai_engine/pipeline/demo.py` | scripted presentation mode |

Two packages are not in the original phase list but are needed to make those
phases work: `ai_engine/video/` (Phase 3) and `ai_engine/utils/` (shared
geometry / time / JSON helpers), plus `ai_engine/scoring/`, `ai_engine/visualization/`
and `ai_engine/reporting/` as single-responsibility homes for their phase.

## 5. Installation

```bat
cd Desktop\SafeVision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **This machine used a smaller setup.** Drive C: was nearly full, so the venv
> here was created with `python -m venv --system-site-packages .venv` and
> reuses the already-installed packages instead of duplicating ~2 GB. On a
> normal machine use the plain `python -m venv .venv` + `pip install -r
> requirements.txt` above; the result is identical.

Verify the installation:

```bat
python -c "import cv2, numpy, ultralytics, torch; print('deps OK')"
python main.py --list-cameras
```

## 6. Model setup

```bat
python scripts\download_model.py            :: yolo11n.pt into models\
python scripts\download_model.py --list     :: sizes and trade-offs
python main.py --check-model                :: loads it and prints the class filter
```

Ultralytics also downloads weights automatically on first use. After the first
run everything works offline. Nothing here needs an account, an API key or a
paid service.

**The detector cannot detect "accident" as a class.** A COCO checkpoint knows
`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person` and nothing about
accidents. SafeVision therefore uses the detected objects plus motion,
trajectory, collision and temporal reasoning to *infer* an accident. If you point
`model.weights` at an accident-specific checkpoint that does have such a class,
it is surfaced as a separate signal (`evidence.signals.accident_class_score`)
rather than being treated as proof.

## 7. Running it

```bat
:: local video
python main.py --source test_videos\accident.mp4

:: laptop webcam
python main.py --source 0 --camera-id CAM-001

:: the automatic DEMONSTRATION emergency notification: one call, then one SMS
:: (needs DEMO_MODE=true in .env)
python main.py --source test_videos\accident_demo.mp4 --show --demo-call
python main.py --check-emergency

:: CCTV / IP camera - credentials come from the environment, never the URL
setx SAFEVISION_RTSP_USERNAME admin
setx SAFEVISION_RTSP_PASSWORD <secret>
python main.py --source "rtsp://192.168.1.64:554/stream" --camera-id CAM-002

:: watch the debug overlay
python main.py --source clip.mp4 --show

:: save an annotated copy + stream the incident JSON on stdout
python main.py --source clip.mp4 --save-output --output out\debug.mp4 --json
```

A source must always be chosen: a bare `python main.py` is a usage error rather
than a silent webcam start. Set `video.source` in `config.yaml` if you want a
default.

With `--json`, **stdout carries exactly one JSON object per confirmed incident
and nothing else** (logs go to stderr), so the CLI can be piped straight into a
consumer:

```bat
python main.py --source clip.mp4 --json > incidents.ndjson
```

Quick checks before a demo:

```bat
python main.py --print-config
python main.py --print-schema
python main.py --list-cameras
python main.py --check-source clip.mp4
python main.py --check-model
```

No test videos are shipped. Generate the synthetic demo clips with
`python scripts\make_test_video.py` (drawings, for the pipeline walkthrough -
not for accuracy claims), or drop your own footage into `test_videos/`.

A real-imagery sanity check (downloads a real photo and runs the real detector
over it) is available too:

```bat
python scripts\real_yolo_check.py --photo test_videos\street_photo.jpg
python scripts\real_yolo_check.py --video --frames 60
```

### Presenting the state machine

```bat
python main.py --demo
```

The demo always starts with two **scripted** scenarios (clearly labelled as
synthetic detections) so the state machine is visible even without accident
footage, then plays your video clips:

```
==============================================================================
 SCENARIO 1/5  scripted crash (synthetic detections)    source: scenario:head_on_crash
 NOTE: scripted detections (not YOLO output). This demonstrates the
       decision logic - tracking, motion, collision, temporal fusion,
       verification and the incident JSON - not detector accuracy.
==============================================================================
    frame    0  STATE NORMAL             score 0.00   [physical 0.00, temporal 0.00]
    frame   42  >> NORMAL -> SUSPICIOUS   (physical evidence 0.42 >= 0.30 for 2 frame(s))
    frame   44  >> SUSPICIOUS -> VERIFYING   (physical evidence held >= 0.34 for 3 frame(s))
    frame   59  >> VERIFYING -> CONFIRMED_ACCIDENT   (multi-frame evidence confirmed:
                       peak physical evidence 0.55, temporal 0.48, 3 agreeing signals,
                       1.0s of verification)
    incident INC-20260928-0069  severity MEDIUM  score 0.42  vehicles 2  signals 3

==============================================================================
 SCENARIO 2/5  scripted false-alarm trap (synthetic detections)
==============================================================================
    no confirmed incident (this is the expected result for this scenario)
```

A clip the engine finds nothing in reports *"no confirmed incident (this is a
valid result)"* - the demo never fabricates an outcome.

## 8. Configuration

Everything lives in `config.yaml`; nothing is hard-coded.

| Section | Key settings |
|---|---|
| `model` | `weights`, `imgsz`, `conf`, `iou`, `device`, `classes` |
| `video` | `source`, `frame_stride`, `target_width`, `rtsp_transport`, reconnect limits |
| `tracking` | `backend`, `match_threshold`, `conf_high`/`conf_low`, `max_age`, `recovery_distance` |
| `motion` | stop thresholds, turn threshold, per-term `weights` |
| `trajectory` | window, deviation thresholds, `crossing_weight` (capped - crossing alone is weak) |
| `collision` | `min_signals` ★, signal thresholds, `physical_weight` / `context_weight` |
| `temporal` | `window_seconds`, `hold_tau_seconds`, `min_consecutive_frames`, `isolated_penalty` |
| `scoring` | component `weights` ★, negative-evidence `penalties` |
| `verification` | `suspicious_score`, `verify_entry_score`, `confirm_score` ★, `temporal_confirm_score` ★, `min_vehicles` ★, `min_signals` ★, cooldowns |
| `severity` | level thresholds and weights |
| `evidence` | `enabled`, `root`, `save_clip`, `pre_event_seconds`, `max_evidence_dirs` |
| `location` | `cameras_file`, `default_camera_id`, RTSP credential env var names |
| `visualization` | overlay switches |
| `runtime` | `max_frames`, `save_output_video`, `incidents_file` |

★ = the ones that matter most for false-alarm control.

Three override layers:

```bat
python main.py --source clip.mp4 --set verification.confirm_score=0.5
set SAFEVISION_MODEL=yolo11s.pt
set SAFEVISION_DEVICE=cpu
```

An unknown key, an out-of-range value or an inconsistent combination fails
immediately with a clear message instead of silently doing nothing.

## 9. How false-alarm prevention works

1. **Nothing is decided from one frame.** Every component score is pushed into
   the temporal window (`ai_engine/temporal/`), which weights recent evidence,
   holds a decaying memory with a time constant, keeps the in-window peak and
   penalises isolated spikes.
2. **A collision needs agreement.** The collision analyzer only returns a
   candidate when at least `collision.min_signals` independent signals fired.
   Below the gate the score is multiplied by 0.35.
3. **Proximity alone is deliberately weak.** `physical_weight = 0.75` is owned by
   signals that describe what a collision *does* (closing, deceleration,
   post-impact stopping, displacement). `context_weight = 0.25` covers
   proximity, box overlap and path crossing - on a fixed camera two vehicles
   overlap in the image constantly.
4. **Closing is gated by time-to-impact.** Two vehicles driving towards each
   other 400 px away are not "closing on a collision"; the signal is scaled by
   how soon contact would actually happen.
5. **Negative evidence can block a confirmation.** Parked/static objects, a
   stable scene, a single-frame event, unstable tracking, low detection
   confidence and too little history each contribute a penalty *and* appear in
   `blocking_reasons`. A crash that leaves stopped vehicles behind is explicitly
   exempt from the "parked" penalty, because otherwise every real accident would
   be rejected two seconds after it happened.
6. **Confirmation is gated on the event, not the frame.** A crash produces a
   burst of evidence and then a quiet aftermath, so the state machine confirms
   on the *peak physical evidence of the event* together with the *current*
   fused temporal score, a minimum verification duration, and the number of
   agreeing signals.
7. **One event, one incident.** A cooldown and an "incident open" flag prevent
   the same crash from being reported every frame.
8. **Evidence is separated from the decision.** Every rejected event still keeps
   its `evidence/INC-.../meta.json` marked `FALSE_ALARM`, which is what makes
   the system tunable instead of a black box.

The scripted regression suite in `ai_engine/detection/scenarios.py` is the
executable specification of this behaviour - it contains three crashes and six
false-alarm traps (hard brake, opposite lanes, following too close, parked cars,
pedestrian crossing, single-frame ghost):

```bat
python scripts\validate_thresholds.py
```

```
scenario               expect    got    peak   event  severity  veh  sig  result
--------------------------------------------------------------------------
following_close         quiet  quiet   0.033   0.000         -    0    0  PASS
hard_brake              quiet  quiet   0.284   0.364         -    0    0  PASS
head_on_crash           ALARM  ALARM   0.500   0.554    MEDIUM    2    3  PASS
opposite_lanes          quiet  quiet   0.033   0.000         -    0    0  PASS
parked_cars             quiet  quiet   0.009   0.000         -    0    0  PASS
pedestrian_crossing     quiet  quiet   0.033   0.000         -    0    0  PASS
rear_end_crash          ALARM  ALARM   0.527   0.557    MEDIUM    2    2  PASS
single_frame_ghost      quiet  quiet   0.148   0.000         -    0    0  PASS
t_bone_crash            ALARM  ALARM   0.445   0.522       LOW    2    3  PASS
9/9 scenarios behave as configured
```

## 10. How the accident score is calculated

```
accident_score = Σ (weight_i · component_i) / Σ weight_i        # 1
                 × penalty_factor                                 # 2
```

1. **Weighted mean** of the six components. Weights come from `scoring.weights`
   (defaults: collision 0.34, motion 0.16, trajectory 0.08, temporal 0.30,
   object 0.07, scene 0.05). A component that could not be measured this frame
   is dropped from *both* numerator and denominator rather than counted as
   zero, so a missing signal never silently drags the score down. The
   component list and the weights actually used travel with the score.
2. **Penalty factor** from negative evidence
   (`scoring.penalties`, capped by `penalty_max`). Stored as
   `penalties: {parked_vehicles: 0.35, ...}`.

`accident.physical_score` is the same weighted mean **without** the temporal
component; the state machine uses it to decide whether an event is worth
verifying, and the temporal fusion to decide whether it is confirmed.

Example (a head-on collision, from `evidence` in a real incident):

```json
"accident": { "score": 0.50, "confidence_percent": 50, "severity": "MEDIUM" },
"evidence": {
  "collision_score": 0.65, "motion_score": 0.44, "trajectory_score": 0.13,
  "temporal_score": 0.48, "object_evidence_score": 0.45, "scene_evidence_score": 0.37,
  "agreement_signals": 3,
  "weights": { "collision": 0.34, "motion": 0.16, "trajectory": 0.08,
               "temporal": 0.30, "object_evidence": 0.07, "scene_evidence": 0.05 },
  "reasons": ["Sudden vehicle velocity change", "Collision proximity detected",
              "Post-collision stopping", "Multiple vehicles involved"],
  "negative_evidence": ["Parked vehicles only (no meaningful movement change)"]
}
```

### The seven collision signals

| Signal | What it measures | Weight in the score |
|---|---|---|
| `proximity` | centre distance in object diagonals | context |
| `overlap` | box IoU + centre inside the other box | context |
| `convergence` | paths meet ahead of both objects | context |
| `closing` | radial closing speed, gated by time-to-impact | physical |
| `deceleration` | **both** objects lost speed together | physical |
| `post_stop` | both essentially still *after* a physical event | physical |
| `displacement` | position jump the previous velocity did not predict | physical |

## 11. How severity is calculated

`severity` is an **AI-estimated scene severity**, not a medical or injury
assessment. Components: number of vehicles involved, collision intensity proxy
(peak closing speed before impact), peak deceleration, displacement, people
present (*presence only*), traffic blockage, duration of the abnormal scene, and
an optional external smoke/fire signal (absent by default, reported as
`unavailable_signals`).

```
severity_score = Σ (weight · component) / Σ weight
LOW (< 0.25)  ·  MEDIUM (0.25–0.50)  ·  HIGH (0.50–0.75)  ·  CRITICAL (≥ 0.75)
```

The transient signals are remembered for the whole event, because severity is
assessed at confirmation time - long after the closing speed has vanished.

Every severity result carries the wording *AI-estimated scene severity* and a
`disclaimer`. The system does **not** claim injury detection, triage or damage
estimation.

## 12. Evidence capture

Only for events that reach `VERIFYING` or `CONFIRMED_ACCIDENT`:

```
evidence/INC-20260928-0001/
    before.jpg        frame from before the event (pre-event ring buffer)
    before_mid.jpg    a second "before" frame
    collision.jpg     the frame with the highest collision score
    after.jpg         the confirmation frame
    annotated.jpg     the same frame with boxes, trails and HUD
    meta.json         scores, signals, transitions, SHA-256 of every file
    clip.mp4          optional (evidence.save_clip: true)
```

Writes are atomic, every file is hashed with SHA-256, and old directories are
pruned beyond `evidence.max_evidence_dirs`. A rejected event keeps its directory
with `"outcome": "FALSE_ALARM"` so you can review what nearly fooled the system.

## 13. JSON output and integration

Full incident (written to `incidents.json`, printed with `--json`):

```json
{
  "schema_version": "1.3.0",
  "incident_id": "INC-20260928-0001",
  "status": "CONFIRMED_ACCIDENT",
  "timestamp": "2026-09-28T09:15:00.123456+00:00",
  "camera":   { "camera_id": "CAM-001", "name": "Main Road Camera", "type": "cctv" },
  "location": { "name": "Chennai Main Road", "latitude": 13.0827, "longitude": 80.2707,
                "is_camera_registered": true, "source": "camera_metadata" },
  "accident": { "score": 0.50, "confidence_percent": 50, "severity": "MEDIUM",
                "severity_score": 0.53, "confirmed": true, "frames_of_evidence": 46,
                "verification_seconds": 1.9,
                "confirmation": "multi-frame evidence (temporal fusion + independent signals)" },
  "evidence": { "collision_score": 0.65, "motion_score": 0.44, "trajectory_score": 0.13,
                "temporal_score": 0.48, "reasons": ["..."], "negative_evidence": ["..."],
                "agreement_signals": 3, "weights": { } },
  "objects":  { "vehicles_involved": 2, "persons_detected": 0, "involved_track_ids": [12, 17] },
  "media":    { "evidence_dir": "evidence/INC-20260928-0001",
                "annotated_frame": "evidence/INC-20260928-0001/annotated.jpg",
                "evidence_frames": ["..."], "clip": null, "files": { "before": "<sha256>" } },
  "performance": { "fps": 14.8, "inference_ms": 41.2, "tracking_ms": 1.8, "analysis_ms": 2.4 },
  "demo_emergency_call": { "label": "DEMO EMERGENCY CALL", "attempted": true,
                           "status": "COMPLETED", "trigger": "automatic_on_confirmation",
                           "incident_id": "INC-20260928-0001", "recipient": "+91*****3949",
                           "provider": "simulation", "call_placed": true },
  "warnings": [], "disclaimer": "SafeVision output is an AI-estimated accident probability ..."
}
```

`demo_emergency_call` is present on every incident so a dashboard card renders
identically whether or not demo calling is enabled - check `attempted` to tell
them apart. See section 16.

Three ways to consume it - all shown in `examples/integration_example.py`:

```python
# 1. callback: the engine calls you the moment an incident is confirmed
pipeline = AccidentPipeline(config, on_incident=my_handler)

# 2. file: the engine writes incidents.json; your job reads it
# 3. reader: you own the frame loop
for result in pipeline.stream("clip.mp4"):
    if result.incident:
        save(result.incident)

# the compact alert payload, or the emergency-response interface:
payload = pipeline.generate_emergency_incident(minimum_score=0.5)   # dict, {} when filtered
```

A complete example payload is checked in as
[`sample_incident.json`](sample_incident.json) (regenerate with
`python scripts\example_incident.py --out sample_incident.json`), and the same
incident *with* the demonstration call as
[`sample_incident_demo_call.json`](sample_incident_demo_call.json)
(`--demo-call`). Both are generated by actually running the pipeline, and both
mask the recipient.

Versioning: `MAJOR.MINOR.PATCH` - reject an unknown MAJOR, ignore unknown
fields (a new MINOR only adds). A draft-07 JSON Schema is available as
`INCIDENT_JSON_SCHEMA` and via `python main.py --print-schema`.

## 14. AI incident report

Rendered **from** the structured evidence, never invented: a vehicle-count
sentence appears only when `vehicles_involved >= 2`, the confidence is the score,
the severity string is whatever the analyzer estimated, and evidence paths are
listed only for files present in the payload. It is available as a dict
(`incident["report"]`) and as Markdown (`incident["report"]["markdown"]`).

## 15. Testing

```bat
python -m unittest discover -s tests -t .        :: 327 tests, no weights needed
python -m unittest tests.test_collision -v       :: one module
pytest tests -q                                  :: also works
```

| Module | Covers |
|---|---|
| `test_detection_types.py` | bbox maths, class vocabulary, detection JSON |
| `test_tracking.py` | id stability across an abrupt stop, class gate, expiry, both backends |
| `test_motion.py` | sudden stop / turn, insufficient history, aggregation, camera-motion compensation |
| `test_trajectory.py` | segment geometry, straight vs deviated, crossing cap |
| `test_collision.py` | the agreement gate, overlap of parked cars, TTI gating |
| `test_temporal.py` | single-frame suppression, sustained accumulation, decay |
| `test_false_alarm_prevention.py` | the full state machine, negative evidence |
| `test_scoring_severity.py` | weights, penalties, level thresholds, honest wording |
| `test_schemas.py` | JSON contract, validation, incident ids, location metadata |
| `test_evidence_video.py` | evidence files + hashes, video I/O, source errors |
| `test_evidence_lifecycle.py` | an interrupted run still produces auditable evidence |
| `test_config.py` | loading, overrides, range checks, JSON helpers |
| `test_cli_contract.py` | CLI flags, informational modes, the `--json` stdout contract |
| `test_cli_config_source.py` | a source from `config.yaml` is honoured; a default one is rejected |
| `test_emergency_call.py` | `.env` handling, the call gates, one-call-per-incident, the contract |
| `test_pipeline_e2e.py` | all nine scenarios end to end, evidence on disk, report, demo |

`scripts\check_readme_commands.py` is not a unit test - it executes every command
in section 23 and fails if any of them stops working.

Manual / exploratory tools:

```bat
python scripts\e2e_check.py                              :: full acceptance check
python scripts\e2e_check.py --photo photo.jpg             :: + real detector check
python scripts\manual_test.py test_videos\accident.mp4    :: frame-by-frame trace
python scripts\manual_test.py --list-scenarios
python scripts\inspect_scenario.py head_on_crash 30 45    :: per-frame components
python scripts\real_yolo_check.py --photo photo.jpg       :: real YOLO on real imagery
python scripts\calibrate.py --accident-dir calibration\accident --normal-dir calibration\normal
python examples\integration_example.py --demo-call        :: the demo call + SMS, simulated
python scripts\show_sms_body.py                          :: preview the SMS text
python main.py --check-emergency                         :: demo-call config check
python scripts\check_readme_commands.py                  :: re-verifies section 23
```

**Section 23 has the full step-by-step commands** - setup, demo, your own
footage, CCTV, JSON output, the demo call, verification and troubleshooting.

`scripts/e2e_check.py` is the single command that proves the whole chain works
and prints a PASS/FAIL line per acceptance item.

## 16. Performance

* The model is loaded **once** and kept resident (one throwaway warm-up
  inference at start-up).
* Frames can be downscaled (`video.target_width`) and skipped
  (`video.frame_stride`) *before* inference.
* **Measured on the development machine** (CPU only, no GPU, `yolo11n`,
  `imgsz=640`, 640×360 input): ~95–190 ms of inference per frame, i.e. roughly
  6–9 FPS end to end, of which the rest of the pipeline (tracking + analysis +
  scoring) is about 3–6 ms. A newer or GPU-backed machine will be considerably
  faster - read your own numbers from the HUD rather than quoting these.
* Evidence frames are held in a bounded in-memory ring (30 frames max), never
  written continuously.
* `device: auto` uses CUDA when available, `half: true` enables fp16 (ignored
  with a log line on CPU).

## 17. GPU (optional)

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py --source clip.mp4 --device 0 --set model.half=true
```

`device: auto` (the default) already selects CUDA when it is present.

## 18. Limitations

* **No accuracy claim is made.** The shipped thresholds were calibrated on
  scripted scenarios, not on a labelled accident dataset. Re-tune them on your
  own footage with `scripts/calibrate.py`; the numbers you should quote are the
  ones you measure, with the false-alarm rate reported alongside detection rate.
* **No injury detection.** A person in the scene raises the severity score
  because their *presence* matters operationally, and nothing more.
* **Location is whatever was registered for the camera.** If the camera moves,
  the metadata is wrong - the system cannot tell.
* **Night, rain, glare, heavy traffic and low camera angles** are the hard
  cases; the detector's own accuracy dominates the system.
* **Perspective limits collision analysis.** A fixed camera cannot recover
  depth, so "closing speed" is measured in the image plane; two vehicles
  approaching in *different lanes* are handled by the proximity and
  time-to-impact gates, not by geometry alone. Panning / PTZ cameras are handled
  by the global-motion compensation in `ai_engine/motion/`, which estimates the
  camera's own motion from the tracks and subtracts it - but it is worth
  measuring on your own footage: on the real panning test clip the engine still
  reached `VERIFYING` once and correctly rejected it as a false alarm (0
  incidents, 1 false alarm over 45 frames).
* **Re-identification across long occlusions** is limited to
  `tracking.max_age` frames.
* **Single-process.** Multiple cameras = multiple processes, each writing its
  own incident file; a shared incident-id counter makes that safe.
* **The demonstration call is a demo, not a notification service.** It has one
  recipient, no retry logic, no escalation and no delivery guarantee. It
  reports what the telephony API returned; it does not verify a human answered.
  The voice message says "demonstration call" out loud so the recipient is never
  confused.
* **There is no GPS.** Every location in the payload and in the SMS is the
  camera's *registered* position, labelled as such. SafeVision cannot tell you
  where a vehicle actually is.
* **The SMS is sent only after a call that reached the network.** If the call
  fails at the provider, no SMS is sent - the incident is still in the JSON, so
  nothing is lost, but there is no fallback notification channel.

## 19. Future improvements

* Camera calibration (homography) to measure real metres and closing speeds in
  km/h instead of image units.
* A second model for scene-level accident classification, used as an extra
  independent signal rather than as a replacement for the evidence pipeline.
* Optional smoke/fire and debris detectors wired through
  `severity.optional_signal_provider`.
* Multi-camera correlation: the same event seen by two cameras.
* A learned post-processing head (logistic regression over the component scores)
  trained on labelled footage, keeping the analytic layer as a hard safety gate.
* On-vehicle detection to turn every camera into a notifying node.
* Replay/backfill mode so recorded footage can be re-analysed after a model
  update.

## 20. Project layout

```
SafeVision/
├── ai_engine/
│   ├── config/          settings.py, classes.py - every threshold
│   ├── detection/       yolo_detector.py, detection_types.py, scripted.py, scenarios.py
│   ├── tracking/        byte_tracker.py
│   ├── motion/          motion_analyzer.py
│   ├── trajectory/      trajectory_analyzer.py
│   ├── collision/       collision_analyzer.py
│   ├── temporal/        temporal_fusion.py
│   ├── verification/    false_alarm_prevention.py
│   ├── scoring/         accident_scorer.py
│   ├── severity/        severity_analyzer.py
│   ├── evidence/        evidence_capture.py, incident_id.py
│   ├── emergency/       dispatcher.py, providers.py, call_record.py,
│   │                    recipient.py, envfile.py  (demo call, DEMO_MODE only)
│   ├── location/        location_manager.py
│   ├── video/           source.py
│   ├── visualization/   overlay.py
│   ├── reporting/       incident_report.py
│   ├── schemas/         incident_result.py, enums.py
│   ├── pipeline/        accident_pipeline.py, demo.py
│   └── utils/           geometry, time, json, logging, ring buffer
├── models/              YOLO weights
├── evidence/            per-incident evidence (git-ignored)
├── test_videos/         your clips (git-ignored)
├── tests/               327 unit + end-to-end tests
├── scripts/             e2e_check, validate_thresholds, calibrate, manual_test,
│                        real_yolo_check, download_model, make_test_video,
│                        inspect_scenario, check_readme_commands
├── examples/            integration_example.py
├── docs/                my_contribution.md, integration_contract.md,
│                        scoring_and_tuning.md
├── main.py              CLI
├── config.yaml          all thresholds (never a phone number)
├── cameras.yaml         camera-registered locations
├── .env.example         the committed template for DEMO_MODE + the recipient
├── .env                 your local secrets - git-ignored, never commit
├── sample_incident.json a real incident payload, for reference
└── sample_incident_demo_call.json  the same, with the demo call attached
```

## 21. Demonstration emergency call (DEMO_MODE)

For the project demonstration, SafeVision can automatically place **one
demonstration call** to an authorised recipient when an incident is confirmed.

This is a demo aid, not an emergency service. It is **not** connected to any
ambulance, police or 112 line, and no emergency service is ever notified.

### What triggers it

Nothing to click. The call is automatic, and only when **all** of these hold:

1. `DEMO_MODE=true` (from `.env`, or `--demo-call` for one run).
2. The incident reached `CONFIRMED_ACCIDENT` - i.e. it already passed the whole
   false-alarm-prevention pipeline. A `SUSPICIOUS` or `VERIFYING` frame never
   triggers anything.
3. Exactly once per `incident_id`. Frames after the confirmation, and repeated
   deliveries of the same incident, are counted and ignored.

If any of those fails, the incident payload says so explicitly
(`demo_emergency_call.reason`) rather than silently doing nothing.

### Setup

```bat
copy .env.example .env
```

Then edit `.env`:

```ini
DEMO_MODE=true
EMERGENCY_CONTACT_NUMBER=+91XXXXXXXXXX   ; your authorised demo recipient
```

The number lives **only** in `.env`, which is git-ignored. It is never written
into Python source, `config.yaml`, a log line or the overlay - those show a
masked form such as `+91*****3949`. Real environment variables always win over
`.env`, so `setx DEMO_MODE false` disables calling without editing the file.

One deliberate exception: `emergency.expose_recipient: true` (the default)
adds `recipient_full` to the incident JSON so your dashboard can show the
number while you present. Set it to `false` if you share `incidents.json` or
`demo_report.json` with anyone - then the payload carries the masked form only.
Either way these files are git-ignored, so the number is never committed. A
test enforces this: it scans every committed file and fails if a non-fictional
phone number appears outside `.env`.

### The notification: call first, then SMS

On **one** confirmed incident the engine places **one** call and then sends
**one** SMS. Both live on a single record per `incident_id`, so "exactly once"
is structural rather than a convention.

```
CONFIRMED_ACCIDENT
   |
   +-- CALL   -> INITIATED -> RINGING -> IN_PROGRESS -> COMPLETED
   |
   +-- SMS    -> SENT
```

The SMS follows only a call that actually reached the recipient's network
(`RINGING`, `IN_PROGRESS`, `COMPLETED`, `BUSY`, `NO_ANSWER`). If the call was
skipped, blocked or failed outright, the SMS is recorded as `SKIPPED` with the
reason - it is never sent "just in case".

The SMS body is built from the confirmed incident - nothing hard-coded:

```
🚨 ACCIDENT ALERT
Incident: INC-20261002-0006
Date: 02-10-2026
Time: 13:23:03
Location: Chennai Main Road (13.082700, 80.270700)
Camera: CAM-001
Severity: MEDIUM (AI-estimated scene severity, confidence 42%)

A traffic accident has been confirmed by SafeVision from multi-frame video
evidence. Emergency assistance may be required.

Location source: CAMERA-REGISTERED (not a GPS fix)
This is an AI estimate, not a verified injury assessment. Please confirm before acting.
```

Preview it without running a pipeline:

```bat
python scripts\show_sms_body.py
```

**Location honesty.** SafeVision has no GPS receiver. The location is the
camera's *registered* position, and the SMS says so explicitly
(`Location source: CAMERA-REGISTERED (not a GPS fix)`). For an unregistered
camera it prints `UNREGISTERED CAMERA - no GPS fix available` rather than
inventing coordinates.

### Providers

| Provider | When | Behaviour |
|---|---|---|
| `simulation` (default) | no telephony credentials in `.env` | walks `INITIATED → RINGING → IN_PROGRESS → COMPLETED`, then `SMS → SENT`, with no network access |
| `twilio` | `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` + `TWILIO_FROM_NUMBER` (call) / `TWILIO_PHONE_NUMBER` (SMS) set | places a real call, then sends a real SMS via Twilio's REST API using only the standard library |
| `none` | `emergency.provider: none` | the feature is removed |

`provider: auto` picks Twilio when its credentials are present and the
simulation provider otherwise. **This is the safety property that matters: with
no credentials in `.env`, the code physically cannot place a real call or send a
real SMS.** A Twilio trial account can only call and message numbers you have
verified, so test it before the demo.

Credentials are read only from the environment, are never logged, never appear
in the incident JSON and are never committed.

### Showing the status

```bat
python main.py --check-emergency     :: configuration check, recipient masked
python main.py --source clip.mp4     :: "DEMO EMERGENCY CALL - RINGING | ..."
```

The debug overlay draws a persistent banner with the same wording, and the status
travels in the incident JSON (`demo_emergency_call` **and**
`demo_emergency_sms`) plus a `demo_calls.jsonl` audit log. The dashboard should
render `label` + `status` from those blocks; there is no button to wire up.

### Required log lines

For a genuine confirmed accident you will see exactly:

```
STATE NORMAL -> SUSPICIOUS (physical evidence 0.40 >= 0.30 for 2 frame(s))
STATE SUSPICIOUS -> VERIFYING (physical evidence held >= 0.34 for 3 frame(s))
STATE VERIFYING -> CONFIRMED_ACCIDENT (multi-frame evidence confirmed: ...)
DEMO CALL -> INITIATED | +91*****3949 | simulation | incident INC-...
DEMO CALL -> RINGING | +91*****3949 | simulation | incident INC-...
DEMO CALL -> IN_PROGRESS | +91*****3949 | simulation | incident INC-...
DEMO CALL -> COMPLETED | +91*****3949 | simulation | incident INC-...
DEMO SMS  -> SENT | +91*****3949 | simulation
```

Without `DEMO_MODE` the prefixes are `CALL ->` and `SMS ->`.

## 22. Safety and ethics

This is a **decision-support prototype for a college innovation project**. It
produces an AI-estimated probability from video evidence and a camera-registered
location. Outside `DEMO_MODE` it contacts nobody, and a human should always
confirm before any emergency action is taken. The evidence directory exists so
that decision can be audited.

The demonstration notification is gated behind `DEMO_MODE`, fires only on a
*confirmed* incident, sends exactly one call and one SMS to a single number you
supply, and is labelled `DEMO EMERGENCY CALL` / `DEMO SMS` everywhere it
appears. Keep it that way: a demo that can ring an arbitrary phone number is a
hazard, and one that reaches an ambulance line is worse.

See [`docs/my_contribution.md`](docs/my_contribution.md) for the 60% breakdown,
[`docs/integration_contract.md`](docs/integration_contract.md) for the handoff to
the backend, and [`docs/scoring_and_tuning.md`](docs/scoring_and_tuning.md) for
how to re-calibrate the thresholds on your own footage.

---

## 23. Run commands — step by step

Copy-paste blocks. Run every command from the project folder
(`Desktop\SafeVision`). `python` below means the venv interpreter
(`.venv\Scripts\python.exe`); if you activated the venv in step 2 you can type
`python` directly.

Every command in steps 2-9 is re-verified by
`python scripts\check_readme_commands.py`, so this section cannot drift away
from what actually runs.

### Step 1 - create the environment

```bat
cd Desktop\SafeVision
python -m venv .venv
.venv\Scripts\activate
```

If `pip install` in the next step fails with *"No space left on device"*,
create the venv so it reuses the packages you already have:

```bat
python -m venv --system-site-packages .venv
```

### Step 2 - install the dependencies

```bat
pip install -r requirements.txt
```

Needs internet once. Verify:

```bat
python -c "import cv2, numpy, ultralytics, torch; print('deps OK')"
```

Expected: `deps OK`.

### Step 3 - get the model and the test clips

```bat
python scripts\download_model.py
python scripts\make_test_video.py
python main.py --check-model
python main.py --check-source test_videos\street_photo.jpg
```

Expected: the model loads (`model 'yolo11n' loaded successfully`) and the source
prints `source is usable`.

### Step 4 - set up `.env` (once)

```bat
copy .env.example .env
notepad .env
```

Set the demo recipient and switch:

```ini
DEMO_MODE=true
EMERGENCY_CONTACT_NUMBER=+91XXXXXXXXXX   ; your authorised demo recipient
```

Leave the three `TWILIO_*` lines **empty** - then the call is fully simulated
and nothing is ever dialled. Check what the engine will do, without calling:

```bat
python main.py --check-emergency
```

### Step 5 - watch the decision logic (the demo to present)

```bat
python main.py --demo
```

This is the one command to show a judge. It runs two clearly-labelled scripted
scenarios first, so the state machine is visible even without accident footage,
then your clips. Look for:

```
NORMAL -> SUSPICIOUS -> VERIFYING -> CONFIRMED_ACCIDENT
```

and, if `DEMO_MODE=true`, the call line:

```
>> DEMO EMERGENCY CALL - INITIATED | +91*****3949 | simulation
>> DEMO EMERGENCY CALL - RINGING | +91*****3949 | simulation
>> DEMO EMERGENCY CALL - IN_PROGRESS | +91*****3949 | simulation
>> DEMO EMERGENCY CALL - COMPLETED | +91*****3949 | simulation
```

Only one call appears even though many frames follow the confirmation.

### Step 6 - run it on your own footage

```bat
:: a video file (drop it in test_videos\ first)
python main.py --source test_videos\my_clip.mp4

:: laptop webcam, with the debug overlay
python main.py --source 0 --camera-id CAM-001 --show

:: save an annotated copy and the incident JSON
python main.py --source test_videos\my_clip.mp4 --save-output --output out\debug.mp4

:: stop early / cap the cost
python main.py --source test_videos\my_clip.mp4 --max-frames 200
```

> The shipped `test_videos\*.mp4` are **synthetic drawings** used to exercise the
> pipeline; YOLO finds no real objects in them, so a run over them correctly
> reports 0 incidents. Use real footage (or step 5) to see detections.

### Step 7 - CCTV / IP camera

```bat
setx SAFEVISION_RTSP_USERNAME admin
setx SAFEVISION_RTSP_PASSWORD <secret>
python main.py --source "rtsp://192.168.1.64:554/stream" --camera-id CAM-002
```

`setx` takes effect in a **new** terminal. Credentials stay in the environment,
never in the URL, a log file or the incident JSON.

### Step 8 - hand the JSON to your teammate

```bat
:: one JSON object per incident on stdout, nothing else (logs go to stderr)
python main.py --source test_videos\my_clip.mp4 --json > incidents.ndjson

:: or write incidents.json / evidence\ / the demo-call audit trail
python main.py --source test_videos\my_clip.mp4
```

Print the contract instead:

```bat
python main.py --print-schema
python main.py --print-config
python main.py --list-cameras
```

### Step 9 - the demonstration emergency call

```bat
:: configuration check (places no call, recipient masked)
python main.py --check-emergency

:: force it on / off for one run, whatever .env says
python main.py --source test_videos\my_clip.mp4 --demo-call
python main.py --source test_videos\my_clip.mp4 --no-demo-call

:: the full lifecycle (call then SMS), simulation-pinned so it cannot dial
python examples\integration_example.py --demo-call

:: preview the exact SMS text
python scripts\show_sms_body.py
```

The notification is automatic: no button, exactly one call and one SMS per
**confirmed** incident. Turn it off entirely with either

```bat
python main.py --source clip.mp4 --no-demo-call     :: for one run
setx DEMO_MODE false                                 :: for good (new terminal)
```

### Step 10 - prove it still works

```bat
python -m unittest discover -s tests -t .       :: 327 tests
python scripts\e2e_check.py                     :: 73/73 acceptance checks
python scripts\e2e_check.py --real-footage       :: + real-footage false-alarm check
python scripts\validate_thresholds.py           :: 9/9 scenarios
python scripts\check_readme_commands.py         :: every command in this section
```

Expected output:

```
Ran 327 tests ... OK
  73/73 checks passed
  77/77 checks passed
9/9 scenarios behave as configured
  14/14 documented commands verified
```

### Step 11 - tuning and calibration

```bat
python main.py --source clip.mp4 --set verification.confirm_score=0.5
python scripts\inspect_scenario.py head_on_crash 30 45
python scripts\manual_test.py test_videos\my_clip.mp4
python scripts\calibrate.py --accident-dir calibration\accident --normal-dir calibration\normal
python scripts\real_yolo_check.py --photo test_videos\street_photo.jpg
```

### Quick reference

| I want to... | Command |
|---|---|
| see it work | `python main.py --demo` |
| run my video | `python main.py --source test_videos\my_clip.mp4` |
| use the webcam | `python main.py --source 0 --show` |
| get the incident JSON | `python main.py --source clip.mp4 --json` |
| see the demo call + SMS | `python examples\integration_example.py --demo-call` |
| stop the demo call | `python main.py --source clip.mp4 --no-demo-call` |
| check the setup | `python main.py --check-emergency` |
| check everything | `python scripts\e2e_check.py` |
| print the contract | `python main.py --print-schema` |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `no video source: pass --source ...` | always pass `--source`, or set `video.source` in `config.yaml` |
| `MODEL NOT AVAILABLE` | `python scripts\download_model.py` (needs internet once) |
| `SOURCE NOT USABLE` | `python main.py --check-source <path>` to see why |
| `no telephony credentials in .env` | expected - with `DEMO_MODE=true` the simulation provider is used and no call is placed. With `DEMO_MODE=false` the engine reports `NOT_CONFIGURED` and still places nothing |
| demo call did not happen | run `--check-emergency`; the `action` line tells you exactly which gate refused it |
| `No space left on device` | recreate the venv with `--system-site-packages` (step 1) |
| a real call is wanted | add the three `TWILIO_*` values to `.env`; `provider: auto` then picks the live provider |
---

## 24. Notification reliability, routing and deployment (Phases 4–5)

Three additions, each documented in full:

* [`docs/notification_state_machine.md`](docs/notification_state_machine.md) —
  the call/SMS lifecycle, the `SMS_ON_CALL_FAILURE` policy, retry classes,
  idempotency and callback requirements.
* [`docs/deployment.md`](docs/deployment.md) — modes, pre-flight checklist,
  secrets, retention, the rollback/stop procedure, and an "an alert did not
  arrive" runbook.
* [`docs/location_provenance.md`](docs/location_provenance.md) — how a camera
  position is kept distinct from a live GPS fix.

### What changed in behaviour

| Before | Now |
|---|---|
| `provider: auto` with no credentials silently simulated and reported `COMPLETED` / `SENT` | reports **`NOT_CONFIGURED`**; simulation happens only with `DEMO_MODE=true` |
| SMS-on-call-failure was hard-coded | `emergency.sms_on_call_failure`, explicit and validated |
| exactly-once lived in an in-memory dict, lost on restart | SQLite outbox, `UNIQUE(incident_id, channel)`, crash-safe leases |
| no retries; polling blocked up to 60 s | bounded retry with jitter + circuit breaker; optional signed callbacks |
| routing was implicit | `dispatch/recipients.json` + `routing:` config, with a decision log |

`NOT_CONFIGURED` is deliberately distinct from both `COMPLETED` and `FAILED`:
it means *there was no provider to ask*, so nobody was told.

### Four switches stand between a fresh checkout and a real alert

All default to off:

```yaml
routing:
  mode: simulation          # 1. decide-and-record only
emergency:
  allow_real_call: false    # 2. permit voice
  allow_real_sms: false     # 3. permit SMS
  # plus real TWILIO_* credentials, resolved from .env
```

Any one of them off means nothing is sent.

### Emergency services

SafeVision has **no** public-emergency integration, no jurisdiction and no
authorisation. `emergency.allow_emergency_service_routing` and
`routing.allow_emergency_service` are **refused by configuration validation** —
setting either to `true` fails startup rather than being ignored. An
emergency-service contact listed in the recipient directory is logged and
excluded on every routing decision.

`both_channels_failed_policy: escalate` records the failure and stops. It does
not contact anyone else.

### Commands

```bat
:: routing decisions (sends nothing, in any mode)
python scripts\routing_report.py
python scripts\routing_report.py --mode live

:: evaluation (accuracy is UNESTABLISHED until a labelled set exists)
python scripts\evaluate.py --describe-only

:: latency (RECORDED vs LIVE is always labelled; never a live response claim)
python scripts\latency_bench.py
python scripts\live_latency_bench.py --live-replay

:: the live provider test - refuses unless all eight gates pass
python scripts\live_provider_test.py --help
python scripts\live_provider_test.py --live --channel sms
:: also required when the live-test recipient is your own number:
python scripts\live_provider_test.py --live --channel sms --acknowledge-real-number
```

### Honest limits

* **No real call or SMS has ever been sent.** No delivery is claimed.
* `COMPLETED` can mean an IVR or voicemail answered — it is not proof a person
  was reached.
* **Accuracy is unestablished**: one real clip, one event, and the harness
  refuses to publish a rate from it.
* **No live GPS**: coordinates are camera-registered and labelled as such.
* **No encryption at rest** and no access control — filesystem ACLs only.
* This build sustains ~6 fps on CPU, so it cannot keep up with a 30 fps camera
  without dropping frames.