# My contribution to SafeVision (60%)

**Team member:** the author of this repository
**Scope:** the complete AI / Computer-Vision subsystem
**Not in scope (the other 40%, implemented independently by my teammate):**
backend, database, dashboard, notifications, emergency-response application,
police/ambulance dispatch simulation, authentication, admin tooling.

This document explains exactly what I built, why each part exists, and where the
boundary between my 60% and my teammate's 40% is. It is written so the split can
be evaluated without reading the whole codebase.

---

## 1. The one-paragraph version

I built the part of SafeVision that *decides*. It takes a video stream from a
CCTV camera, a video file or a laptop webcam; detects and tracks every vehicle
and person; measures what the objects are physically doing (motion, trajectories,
collision kinematics); fuses that evidence over several seconds; refuses to
report anything that a single frame, a single signal or a parked car could
trigger; and then emits a versioned JSON incident with a score, an AI-estimated
scene severity, the camera's registered location, captured evidence frames and a
written explanation of why it decided what it decided. My teammate's part takes
that JSON and does something useful with it.

---

## 2. What I delivered, phase by phase

### 2.1 Video input — `ai_engine/video/source.py`

One `VideoReader` class for all three sources (local file, webcam index, RTSP
URL), with the things that actually break in practice handled: source
classification and validation *before* opening (a typo fails in milliseconds
instead of 30 seconds), RTSP credentials injected from environment variables
(never hard-coded, never logged), reconnect with a bounded number of attempts,
empty/corrupt read counting, downscales and frame skipping applied before the
model ever sees a frame, live FPS reporting, deterministic release via
`with`/`close()`.

### 2.2 Object detection — `ai_engine/detection/`

Ultralytics YOLO11 integration (`yolo11n.pt` by default: ~5 MB, CPU-friendly,
no API key, no cloud). I wrote the wrapper that:

* loads the model **once** and keeps it resident, with a throwaway warm-up
  inference;
* restricts inference to the road-scene classes (person, bicycle, car,
  motorcycle, bus, truck) so a person in the background cannot flood the
  tracker;
* normalises alternative vocabularies (`auto`→`car`, `lorry`→`truck`,
  `pedestrian`→`person`, …) so a different public checkpoint still works;
* converts the output into typed `Detection` objects carrying class id, class
  name, confidence, box, centre point, frame index and timestamp;
* surfaces an `accident`-class score **separately** if the configured checkpoint
  has such a class, instead of pretending object detection alone proves an
  accident.

### 2.3 Object tracking — `ai_engine/tracking/byte_tracker.py`

A self-contained ByteTrack implementation: constant-velocity Kalman prediction on
the box centre, **two-stage association** (high-confidence detections first, then
low-confidence detections against the leftovers, which is what keeps a blurred or
partially occluded vehicle alive), Hungarian assignment with IoU and class-group
gating, track history with both smoothed and instantaneous velocity, and a
**distance-based recovery stage** that I added after observing that a plain
ByteTrack emits a brand-new track id exactly when a vehicle stops abruptly - i.e.
precisely during a crash. Track ids are stable, expire after `max_age`, and an
optional second backend adopts ids from Ultralytics' own `model.track()`.

### 2.4 Motion analysis — `ai_engine/motion/motion_analyzer.py`

Per track: position, displacement, velocity, heading, acceleration, **sudden
stop**, **sudden direction change** and positional jitter, EMA-smoothed across
frames and combined into a normalised score with reasons. The abrupt-change
features deliberately use the *unsmoothed* per-frame measurements, because
smoothing is what hides a crash. I also added **global-motion compensation**:
when at least three tracks agree on the same image velocity (a panning or PTZ
camera), that shared motion is estimated and subtracted before the features are
computed, with a minimum-track rule so that two cars driving in the same
direction on a straight road are never mistaken for a moving camera.

### 2.5 Trajectory analysis — `ai_engine/trajectory/trajectory_analyzer.py`

Per track: straightness, heading deviation, lateral deviation from the
established path, zigzag/oscillation, sudden stop. Per pair: path crossing and
convergence with a time-to-closest-approach estimate. The crossing term is
**capped** (`trajectory.crossing_weight = 0.30`) because on a fixed camera two
vehicles in different lanes cross in the image plane constantly.

### 2.6 Collision analysis — `ai_engine/collision/collision_analyzer.py`

Seven independent signals per object pair: proximity, bounding-box overlap,
radial closing speed (gated by time-to-impact), paired deceleration,
post-impact stopping, unexplained displacement spike, and trajectory
convergence. Signals are split into *physical* (what a collision does to a
vehicle, 75% of the score) and *context* (proximity/overlap/crossing, 25%), and
a candidate is only returned when at least `collision.min_signals` agree.
Below the gate the score is suppressed to 35%.

### 2.7 Temporal fusion — `ai_engine/temporal/temporal_fusion.py`

A rolling 3–10 s window with a recency-weighted mean, a decaying accumulator
whose time constant is expressed in *seconds* (so the behaviour does not change
with the processing FPS), an in-window peak hold, a persistence term, a
consistency term and an isolated-spike penalty.

### 2.8 False-alarm prevention and the state machine — `ai_engine/verification/`

The component I consider the heart of the project. It collects **negative
evidence** (parked/static objects, stable scene, no meaningful movement change,
single-frame event, unstable tracking, low detection confidence, insufficient
history) and runs the state machine
`NORMAL → SUSPICIOUS → VERIFYING → CONFIRMED_ACCIDENT`, with
`VERIFYING → FALSE_ALARM → NORMAL` when the evidence disappears. Confirmation
requires agreement (vehicles, independent signals, sustained temporal support,
a minimum verification duration) and is judged on the *event* rather than on one
frame, because a crash produces a burst of evidence followed by a quiet
aftermath. Every transition carries a written reason, and every rejected event
keeps its evidence with `"outcome": "FALSE_ALARM"`.

### 2.9 Transparent accident score — `ai_engine/scoring/accident_scorer.py`

A weighted mean of six components with weights renormalised over the components
that were actually measurable, multiplied by a capped negative-evidence penalty.
The result carries its components, the weights used, the penalties applied and
the reasons - and every weight lives in `config.yaml`.

### 2.10 Severity — `ai_engine/severity/severity_analyzer.py`

LOW/MEDIUM/HIGH/CRITICAL from explainable scene signals: vehicles involved,
collision intensity proxy (peak closing speed), peak deceleration, displacement,
people present (presence only), traffic blockage, duration of the abnormal
scene, plus an optional external smoke/fire signal that is reported as
*unavailable* when no such model is configured. The wording is always
"AI-estimated scene severity" and the result carries a disclaimer; there is no
injury or triage claim.

### 2.11 Camera-registered location — `ai_engine/location/location_manager.py`

A `cameras.yaml` registry (camera id, name, latitude, longitude, road, address,
zone) with range validation, runtime registration for the teammate's database,
and honest behaviour for an unregistered camera: the incident is still reported,
with `is_camera_registered: false` and a warning, rather than invented
coordinates or a silently dropped event.

### 2.12 Evidence capture and incident ids — `ai_engine/evidence/`

Evidence directories named by incident id, holding `before`, `collision`,
`after` and `annotated` frames (plus an optional short clip), each file written
atomically and hashed with SHA-256, with `meta.json` recording the scores,
signals and state transitions. The id generator produces
`INC-YYYYMMDD-NNNN`, is thread-safe, persists its counter atomically and also
respects ids already present on disk.

### 2.13 The output contract — `ai_engine/schemas/`

A versioned JSON incident (`schema_version = 1.3.0`) with a full payload, a
compact alert payload for a dashboard, a draft-07 JSON Schema, a validator that
returns problems instead of raising, and a documented
`MAJOR.MINOR.PATCH` versioning rule. Every payload is plain JSON - no numpy
types, no NaN, no custom objects.

### 2.14 The AI incident report — `ai_engine/reporting/`

Generates the report *from* the structured evidence. It never invents a fact: a
vehicle-count sentence only appears when the count supports it, evidence paths
are listed only for files present in the payload, and missing fields cause the
sentence to be omitted rather than guessed.

### 2.15 Visual debug mode, demo mode and the CLI

The debug overlay draws boxes, class names, confidences, track ids, fading
trajectories, the collision link and a HUD with camera, state, accident score,
all component scores, verification progress, the "holding" reasons, negative
evidence and per-stage timings. Demo mode runs a scripted set of clips and
prints every state transition as it happens - the false-alarm mechanism is
*shown*, not described. `main.py` wraps both in a documented CLI.

### 2.16 The demonstration emergency call — `ai_engine/emergency/`

An automatic **demonstration** call placed once per *confirmed* incident, built
for the project demo and off by default. The interesting part is not the dialling
but the gating:

* the recipient lives only in `.env` (`EMERGENCY_CONTACT_NUMBER`), never in
  Python source or `config.yaml`, and is masked (`+91*****3949`) in every log
  line, in the overlay and in the call log;
* the notification fires only after the false-alarm-prevention state machine
  reaches `CONFIRMED_ACCIDENT` - a `SUSPICIOUS` or `VERIFYING` frame triggers
  nothing;
* **exactly one call and exactly one SMS per `incident_id`**: both channels hang
  off a single record registered *before* the call is placed, so every subsequent
  frame - and every repeated delivery of the same incident - is counted and
  ignored. There is no manual trigger anywhere in the package, which is asserted
  by a test that scans its public API;
* **call first, then SMS.** The SMS is sent only after the call reached the
  recipient's network; if the call was skipped, blocked or failed, the SMS is
  recorded as `SKIPPED` with the reason rather than sent "just in case";
* the SMS body is composed from the confirmed incident (id, date, time,
  camera-registered location, camera id, severity, score) and states the location
  source explicitly - `CAMERA-REGISTERED (not a GPS fix)` - because the engine
  has no GPS receiver and an unregistered camera reports no location at all;
* the transports are pluggable: simulation providers walk the full
  `INITIATED → RINGING → IN_PROGRESS → COMPLETED` + `SENT` lifecycle with no
  network access, and the Twilio providers place a real call and send a real SMS
  through the REST API using only `urllib`, so no dependency was added.
  `provider: auto` selects Twilio only when its credentials are present - **with
  no credentials in `.env` the code physically cannot contact the network**;
* refusals are reported rather than hidden: `SKIPPED` (demo mode off, no
  recipient, per-run limit) and `BLOCKED` (placeholder number, unusable provider)
  both carry a `reason` that reaches the incident payload and the console;
* everything runs on a worker thread, so a slow telephony API cannot stall the
  video loop, and `close()` waits for the in-flight notification.

Both channel states travel to the dashboard in `demo_emergency_call` and
`demo_emergency_sms` on every incident, with the literal labels
`DEMO EMERGENCY CALL` / `DEMO SMS`; a `demo_calls.jsonl` line is the audit trail.

### 2.17 Configuration, calibration and tests

Every threshold is externalised in `config.yaml` with range and
cross-section validation at load time, three override layers (file, `SAFEVISION_*`
environment variables, `--set`), `scripts/validate_thresholds.py` for
regression-testing the decision logic on nine scenarios, `scripts/calibrate.py`
for measuring the real score distribution on labelled footage, and 327 automated
tests that need no weights and no network.

---

## 3. The boundary with my teammate's 40%

My code ends at a function call. The AI module never opens a socket and never
touches a database. Its one outbound action is the opt-in demonstration call in
§2.16, which exists for the demo, is disabled by default, and reports its status
through the same JSON contract as everything else.

```python
# ---------- the entire interface my teammate needs ----------
from ai_engine.pipeline import AccidentPipeline
from ai_engine.schemas import generate_emergency_incident

pipeline = AccidentPipeline(load_config("config.yaml"), on_incident=my_handler)
pipeline.process_video("clip.mp4")          # -> RunResult with .incidents (plain dicts)
pipeline.get_incident_result()              # -> the most recent incident as a dict
pipeline.generate_emergency_incident(minimum_score=0.5)   # -> compact alert dict
```

| My module provides | Their system consumes |
|---|---|
| `incident_id`, `status`, `timestamp` | primary key, event type, ordering |
| `accident.score`, `accident.severity` | dashboard cards, alert priority |
| `location` (camera-registered) | map view, dispatch form |
| `evidence.reasons`, `evidence.agreement_signals` | "why was this raised?" panel |
| `media.*` (frames, clip, hashes) | evidence gallery, audit trail |
| `schema_version` | compatibility handling |
| `on_incident` callback | their queue / API / database insert |

Explicitly **not** implemented, by design: any police or ambulance dispatch, SMS
or messaging integration, real emergency phone calls, production database,
dashboard UI, authentication, admin panel.

---

## 4. How I can demonstrate each part

| Claim | How to show it |
|---|---|
| Detection works | `python main.py --check-model` then run on a clip with `--show` |
| Tracking works | watch the `#track_id` labels stay on the same vehicle |
| Motion / trajectory / collision work | `python scripts/manual_test.py clip.mp4` prints the component scores per frame |
| Temporal fusion works | the `temporal` column rises over ~1 s and never spikes for one frame |
| False-alarm prevention works | `python scripts/validate_thresholds.py`: six traps stay quiet, three crashes fire |
| Scoring is transparent | `incident["evidence"]` shows components, weights, penalties, reasons |
| Severity is explained | `incident["accident"]["severity_reasons"]` |
| Location is registered | `incident["location"]["is_camera_registered"]` + `python main.py --list-cameras` |
| Evidence is captured | open `evidence/INC-.../` (frames + meta.json + SHA-256) |
| The state machine works | `python main.py --demo` prints every transition |
| The contract is stable | `python main.py --print-schema`, `examples/integration_example.py` |
| It is not a mockup | 207 tests, `python -m unittest discover -s tests -t .` |

---

## 5. Honest limitations of my 60%

* The thresholds shipped in `config.yaml` were calibrated on **scripted
  scenarios**, not on a labelled accident dataset. They are a documented
  starting point, not a validated operating point, and `scripts/calibrate.py`
  exists precisely because they must be re-tuned on real footage.
* I do not claim an accuracy figure. The one to measure and report is the
  false-alarm rate on normal traffic together with the detection rate on real
  accident clips.
* No injury detection, no damage estimation, no live GPS. The location is the
  camera's registered metadata and the severity is an *AI-estimated scene*
  severity; both are labelled as such in every output.
* The detector's own accuracy (COCO, daylit road scenes) is the ceiling of the
  whole system: night, rain, glare and heavy traffic remain hard.
