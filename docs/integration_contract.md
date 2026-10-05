# Integration contract - AI engine → backend (the 40% handoff)

This is the complete interface between my 60% (the AI/CV engine) and my
teammate's 40% (backend, database, dashboard, notifications, emergency-response
application). Nothing else crosses the boundary.

The engine **reads video and writes JSON**. It never opens a socket and never
touches a database. Its single outbound action is an opt-in **demonstration
phone call** (§5.1), off unless `DEMO_MODE=true` is set in `.env`.

---

## 1. Three ways to plug in

### 1.1 Callback (lowest latency, single process)

```python
from ai_engine import load_config
from ai_engine.pipeline import AccidentPipeline

def on_incident(incident: dict) -> None:
    """Called the moment an accident is confirmed."""
    queue.put(incident)          # <- your queue / API call / DB insert

pipeline = AccidentPipeline(load_config("config.yaml"), on_incident=on_incident)
pipeline.process_video("rtsp://camera/stream")
```

The callback runs on the detection thread: keep it fast, or hand the payload to
a queue and process it elsewhere.

### 1.2 File (simplest to deploy, survives restarts)

```bat
python main.py --source clip.mp4            :: writes incidents.json
```

```python
payload = json.load(open("incidents.json"))  # a RunResult dict
for incident in payload["incidents"]:
    save_to_database(incident)
```

Or consume stdout directly - with `--json` the engine prints **exactly one JSON
object per confirmed incident and nothing else** (all logging goes to stderr),
so it can be piped or tailed:

```bat
python main.py --source rtsp://camera/stream --json > incidents.ndjson
python main.py --source clip.mp4 --json | while read -r line; do ...; done
```

### 1.3 Reader (you own the frame loop)

Use this when the frames already come from a client you manage:

```python
for result in pipeline.stream("rtsp://camera/stream"):
    print(result.state, result.accident_score)
    if result.incident:
        save(result.incident)
```

---

## 2. The public API

| Call | Returns | Notes |
|---|---|---|
| `AccidentPipeline(config, *, detector, on_incident, camera_id, source_label)` | pipeline | every argument is optional |
| `pipeline.process_frame(frame, frame_index=None, timestamp=None)` | `FrameResult` | the core unit; BGR `numpy.ndarray` in |
| `pipeline.stream(source=None, max_frames=0)` | generator of `FrameResult` | the streaming API |
| `pipeline.process_video(source=None, max_frames=0, on_frame=None, output_video=None)` | `RunResult` | the batch API |
| `pipeline.get_incident_result()` | `dict` | most recent incident, `{}` if none |
| `pipeline.generate_emergency_incident(minimum_score=0.0)` | `dict` | compact alert payload, `{}` if filtered |
| `pipeline.incidents` | `list[IncidentResult]` | all confirmed incidents |
| `pipeline.state` | `DetectionState` | `NORMAL`/`SUSPICIOUS`/`VERIFYING`/`CONFIRMED_ACCIDENT`/`FALSE_ALARM` |
| `pipeline.add_callback(fn)` | – | register another consumer |
| `pipeline.reset()` | – | clear per-stream state, keep the model |
| `pipeline.close()` | – | release the model and wait for any in-flight demo call; also works with `with` |
| `pipeline.demo_calls` | `list[dict]` | the demo-call audit trail (empty when demo mode is off) |
| `pipeline.emergency` | `DemoCallDispatcher \| None` | `None` unless `DEMO_MODE` is on |

Standalone helpers:

```python
from ai_engine.schemas import (
    generate_emergency_incident,   # IncidentResult | dict -> alert dict
    get_incident_result,           # anything                    -> incident dict
    validate_incident,             # dict -> list[str] of problems
    INCIDENT_JSON_SCHEMA,          # draft-07 schema
    SCHEMA_VERSION,                # "1.3.0"
)
from ai_engine.location import LocationManager, CameraLocation
pipeline.locations.register(CameraLocation(camera_id="CAM-002", name="...", latitude=..., longitude=...))
```

`.env` (git-ignored, never commit) carries only the demo switch and the demo
recipient:

```ini
DEMO_MODE=true
EMERGENCY_CONTACT_NUMBER=+91XXXXXXXXXX   ; your authorised demo recipient
```

The engine never needs it: `load_config()` works with no `.env` at all, and demo
calling simply stays off.

---

## 3. `FrameResult` - what you get per frame

Use it for a live dashboard; it is JSON-serialisable via `.to_dict()`.

| Field | Meaning |
|---|---|
| `frame_index`, `timestamp`, `frame_size` | frame identity |
| `state` / `status` | internal state / reported status |
| `accident_score` | instantaneous AI-estimated probability (0–1) |
| `component_scores` | `collision`, `motion`, `trajectory`, `temporal`, `object_evidence`, `scene_evidence` |
| `negative_evidence` | the counter-evidence currently active |
| `vehicles_involved`, `persons_in_scene`, `object_counts` | object census |
| `detections`, `tracks` | this frame's raw detections and tracked objects (with ids) |
| `collision` | the best collision candidate, with every signal and its value |
| `temporal`, `verification` | fused evidence, state machine, blocking reasons |
| `score_detail` | full breakdown: components, weights, penalties, reasons |
| `performance` | `fps`, `inference_ms`, `tracking_ms`, `analysis_ms`, `total_ms` |
| `annotated_frame` | the overlay image (when `visualization.enabled`) |
| `incident` | the incident dict **only on the frame an incident is confirmed** |
| `demo_call` | live state of the demonstration call (`{}` when demo mode is off) |
| `warnings` | non-fatal problems (e.g. a failed inference on that frame) |

---

## 4. The incident payload

Full shape in `README.md` §13; the machine-checkable contract is
`ai_engine.schemas.INCIDENT_JSON_SCHEMA` (`python main.py --print-schema`).
A complete, real example produced by the engine is in
[`../sample_incident.json`](../sample_incident.json) - regenerate it any time
with `python scripts/example_incident.py --out sample_incident.json`.

Fields you will definitely use:

```
incident_id                "INC-20260928-0001"        primary key
status                     "CONFIRMED_ACCIDENT"      (emitted only for confirmed)
timestamp                  ISO-8601 UTC
camera.camera_id           "CAM-001"
location.name              "Chennai Main Road"
location.latitude          13.0827                    null if unregistered
location.longitude         80.2707                    null if unregistered
location.is_camera_registered  true|false
accident.score             0.50                       AI-estimated probability
accident.confidence_percent 50
accident.severity          "MEDIUM"                   AI-estimated SCENE severity
accident.severity_score    0.53
accident.confirmation      "multi-frame evidence (temporal fusion + independent signals)"
evidence.reasons           ["Sudden vehicle velocity change", ...]
evidence.negative_evidence ["..."]
evidence.agreement_signals 3
evidence.signals.collision  {... every signal and its value ...}
objects.vehicles_involved  2
objects.persons_detected   0
media.evidence_dir         "evidence/INC-20260928-0001"
media.annotated_frame      "evidence/INC-20260928-0001/annotated.jpg"
media.evidence_frames      ["...before.jpg", "...collision.jpg", ...]
media.clip                 null                       (optional)
media.files                {"before": "<sha256>", ...}
performance.fps            7.4
demo_emergency_call        {label, attempted, status, incident_id, recipient, ...}  (see §5.1)
demo_emergency_sms         {label, attempted, status, provider, recipient, body,
                            sent, sent_at, reason, ...}  (the SMS that follows the call)
warnings                   []
disclaimer                 "SafeVision output is an AI-estimated accident probability ..."
report                     {what_happened, why, confidence, severity, location, evidence_files, markdown}
```

### Versioning

`schema_version` is `MAJOR.MINOR.PATCH`:

* **MAJOR** - a field was removed or its meaning changed → reject it
* **MINOR** - a field was added → ignore what you do not know
* **PATCH** - clarification only

`IncidentResult.from_dict()` is provided so you can round-trip a payload you
have stored.

---

## 5. The emergency-response interface

```python
from ai_engine.schemas import generate_emergency_incident

# compact alert payload for a dispatcher card
alert = generate_emergency_incident(incident, minimum_score=0.5)
# -> {} when the score is below the threshold, otherwise a flat dict with
#    incident_id, status, timestamp, camera_id, location_name, lat/lon,
#    accident_score, confidence_percent, severity, severity_score,
#    vehicles_involved, persons_detected, demo_emergency_call, reasons,
#    annotated_frame, clip, evidence_dir, disclaimer
```

There is **no SMS, no HTTP call and no database** anywhere in this repository -
that is the other 40%.

The one outbound action the AI side performs is the **demonstration phone call**
in §5.1, and it is opt-in, demo-only and off by default.

### 5.1 The demonstration emergency call

When the project is being demonstrated (`DEMO_MODE=true`), the engine places
**one** demonstration call per **confirmed** incident. This is a presentation
aid for our demo; it is not connected to any ambulance, police or 112 line.

What your dashboard needs - all of it arrives in the payload, there is no button
to wire up:

```
demo_emergency_call.label             "DEMO EMERGENCY CALL"     render this text
demo_emergency_call.attempted         true|false                false = demo mode off
demo_emergency_call.status            INITIATED|RINGING|IN_PROGRESS|COMPLETED|
                                      BUSY|NO_ANSWER|FAILED|SKIPPED|BLOCKED
demo_emergency_call.trigger           "automatic_on_confirmation"
demo_emergency_call.incident_id       ties the call to its incident
demo_emergency_call.recipient         "+91*****3949"            masked
demo_emergency_call.recipient_full    present only when emergency.expose_recipient
demo_emergency_call.provider          "simulation" | "twilio" | "none"
demo_emergency_call.reason            why it was skipped or blocked
demo_emergency_call.call_placed       true when a provider took the call
demo_emergency_call.history           [{status, at, provider_status}, ...]
demo_emergency_call.disclaimer        the "not an emergency service" wording
```

`SKIPPED` and `BLOCKED` mean **no call was placed** - a safety interlock refused,
which is a success of the false-alarm prevention, never a failure. Always render
`label` + `status` together so a demo call can never be mistaken for a real
emergency notification.

### 5.2 The SMS that follows the call

One confirmed incident produces **exactly one call and exactly one SMS**, in
that order. Both live on the same per-`incident_id` record, so the pairing is
structural.

```
demo_emergency_sms.label      "DEMO SMS"
demo_emergency_sms.status     PENDING|SENT|FAILED|SKIPPED|BLOCKED
demo_emergency_sms.trigger    "automatic_after_call"
demo_emergency_sms.provider   "simulation" | "twilio" | "none"
demo_emergency_sms.body       the exact text the recipient receives
demo_emergency_sms.sent       true when a provider took it
demo_emergency_sms.sent_at    ISO-8601 timestamp
demo_emergency_sms.reason     why it was skipped or blocked
```

`body` is built from the confirmed incident (id, date, time, camera-registered
location, camera id, severity, score) and always states the location source:

```
🚨 ACCIDENT ALERT
Incident: INC-20261002-0006
Date: 02-10-2026
Time: 13:23:03
Location: Chennai Main Road (13.082700, 80.270700)
Camera: CAM-001
Severity: MEDIUM (AI-estimated scene severity, confidence 42%)
...
Location source: CAMERA-REGISTERED (not a GPS fix)
```

SafeVision has **no GPS receiver**: the coordinates are the camera's registered
position. For an unregistered camera the body says
`UNREGISTERED CAMERA - no GPS fix available` rather than inventing a location.

The SMS is sent only after a call that actually reached the recipient's network
(`RINGING` / `IN_PROGRESS` / `COMPLETED` / `BUSY` / `NO_ANSWER`). If the call was
skipped, blocked or failed, the SMS is `SKIPPED` with the reason.

Because both channels run on a worker thread, the payload captured at incident
time shows `PENDING`; subscribe to `on_call_status` for the live transitions.

The status advances *after* the incident is emitted, so for a live view either
poll `pipeline.emergency.status()` or subscribe:

```python
pipeline = AccidentPipeline(config, on_call_status=lambda s: print(s["label"], s["status"]))
# or
result.demo_call    # the same dict on every FrameResult while demo mode is on
pipeline.demo_calls # the full audit trail
```

The engine also appends one JSON line per call to `demo_calls.jsonl`
(`emergency.call_log`) - useful as the record of what a demo actually did.

Tuning (all in `config.yaml`, section `emergency`): `demo_mode`, `provider`,
`max_calls_per_run`, `min_gap_seconds`, `voice_message`, `expose_recipient`.

---

## 6. Errors and edge cases you should handle

| Situation | What the engine does |
|---|---|
| video file missing / webcam busy / RTSP refused | logs a clear error, `RunResult.video_stats["error"]` is set, no exception escapes |
| stream drops mid-run | reopens up to `video.max_reconnect_attempts` times |
| one frame fails to decode / inference throws | logged, frame skipped, run continues (`performance.detection_failures`) |
| no camera registered | incident still emitted, `location.is_camera_registered = false` + a warning |
| unknown `schema_version` | *your* code decides; the engine never blocks on it |
| nothing detected in a whole clip | no incident; `pipeline.incidents == []` |
| `DEMO_MODE` off, or no recipient in `.env` | no call; `demo_emergency_call.attempted = false` with a `reason` |
| telephony API unreachable | `status = FAILED` with `error`; the incident is still reported |
| a subscriber's callback raises | logged and ignored; the call and the incident both survive |

---

## 7. A minimal receiving endpoint (your side, not implemented here)

```python
@app.post("/api/incidents")
def receive(payload: dict):
    if payload.get("schema_version", "0.0.0").split(".")[0] != "1":
        raise HTTPException(422, "unsupported schema version")
    problems = validate_incident(payload)        # optional, from ai_engine.schemas
    if problems:
        log.warning("incident schema problems: %s", problems)
    incidents.insert(payload["incident_id"], payload)
    dispatch_queue.put(payload["alert"] if "alert" in payload else payload)
    return {"stored": payload["incident_id"]}
```

---

## 8. Database mapping suggestion (not implemented)

| Column | Source |
|---|---|
| `id` | `incident_id` |
| `status` | `status` |
| `detected_at` | `timestamp` (ISO-8601 UTC) |
| `camera_id` | `camera.camera_id` |
| `latitude`, `longitude`, `location_name` | `location.*` |
| `accident_score` | `accident.score` |
| `severity`, `severity_score` | `accident.severity`, `accident.severity_score` |
| `vehicles_involved`, `persons_detected` | `objects.*` |
| `reasons` (JSON) | `evidence.reasons` |
| `evidence_dir`, `annotated_frame`, `clip` | `media.*` |
| `report_markdown` | `report.markdown` |
| `raw_json` (JSON) | the whole payload - keep it, it is the audit trail |
