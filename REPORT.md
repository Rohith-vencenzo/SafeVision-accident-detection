# SafeVision — Engineering Report

Append-only. One dated section per phase. Newest at the bottom.

Status vocabulary: **PASS** (all exit criteria verified) · **PARTIAL** (some
criteria met, blockers recorded) · **BLOCKED** (cannot proceed without input).

Standing rules for every phase: never claim 100% accuracy, zero false alarms,
live GPS, successful real call/SMS, emergency-service routing, or a 30–60 s
response without the evidence named inline. A provider API response is not proof
of call answer or SMS delivery.

---

## 2026-10-02 — Phase 1: repository audit, current state, integration design

**Status: PASS** (audit complete; all five phases GO except the items marked
BLOCKED below, which need operator input and are *not* worked around).

**Nature of this phase:** read-only. No provider request was made, no call or
SMS was sent, no public-emergency contact was attempted or configured.

### 1.1 Workspace and VCS state

| Item | Finding |
|---|---|
| Workspace | `C:\Users\acer` (VS Code root) |
| Accident project root | `C:\Users\acer\Desktop\SafeVision` — confirmed |
| NIPS / X-NIDS project | `C:\Users\acer\nids-project` — **separate; not opened, not modified** |
| Git in SafeVision | **No `.git`.** The project is untracked, so there are no commits, staged files or uncommitted user changes to preserve. Nothing was reset, cleaned or deleted. |
| Pre-existing work preserved | 327 passing tests, all CLI modes, demo mode, evidence capture, camera registry all left intact and re-verified in this phase. |

### 1.2 Reported baseline — verified against source and execution

Each of the five reported gaps was **executed**, not assumed. Evidence:
`scripts/phase1_baseline_probe.py` → `reports/phase1_baseline.json`
(outbound HTTP was replaced with a raising stub for the whole probe; **0 network
attempts** were recorded).

#### Gap A — calls/SMS are demo-only due to missing Twilio credentials → **CONFIRMED**

Credential state (values never printed; token shape only):

```
TWILIO_ACCOUNT_SID         ABSENT
TWILIO_AUTH_TOKEN          ABSENT
TWILIO_FROM_NUMBER         ABSENT
TWILIO_PHONE_NUMBER        ABSENT
EMERGENCY_CONTACT_NUMBER   PRESENT  len=13, country=+9
DEMO_MODE                  PRESENT
config emergency.provider = auto
RESOLVED call provider    = simulation
RESOLVED sms provider     = simulation
```

No telephony SDK is installed and `requirements.txt` declares none; the Twilio
paths are hand-rolled on `urllib`. The real provider code **exists** and is
correct in shape — explicit `provider: twilio` with no credentials fails loudly:

```
TwilioProvider.available()  -> False,
  "missing Twilio credential(s) in .env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER"
TwilioSmsProvider.available()-> False,
  "missing Twilio credential(s) in .env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER"
```

So the integration is real but **unreachable**: no real call or SMS can be sent
from this checkout. Nothing has been sent, and no delivery has been claimed.

#### Gap B — **NEW, higher severity than reported: `provider: auto` reports false success**

Not in the reported list; found during this audit and rated the single most
important issue in the project.

`ai_engine/emergency/providers.py:355-357`:

```python
if twilio().available()[0]:
    return twilio()
return SimulationProvider(ring_after_seconds, duration_seconds)   # <-- silent demo
```

`config.yaml` ships `emergency.provider: auto`. With **zero** credentials the
engine therefore resolves to the simulation provider and reports:

```
with zero credentials the engine reported call=COMPLETED sms=SENT call_placed=True
```

A downstream dashboard, log or database cannot distinguish that from a real
notification. This is exactly the "silent substitution of demo for real success"
the specification prohibits, and it is a **hard blocker for the Phase 4 exit
criteria** ("missing credentials show `NOT_CONFIGURED` rather than DEMO-as-
success"). Fix belongs in Phase 4; recorded here, not worked around.

#### Gap C — live GPS absent, coordinates are camera-registered → **CONFIRMED**

A grep for `gps|GPS|LIVE_GPS|fix_timestamp|nmea|NMEA` over every `.py`/`.yaml`
in the project returns **only comments and label strings**. There is no GPS
receiver, no geolocation API, no NMEA parsing and no IP-geolocation. All
coordinates come from `cameras.yaml`:

```yaml
CAM-001: {latitude: 13.0827, longitude: 80.2707, location_name: Chennai Main Road}
```

The existing wording is already honest ("not a live GPS fix", `NO_GPS`), which is
creditworthy — the *labels* are correct; what is missing is the provider
interface, accuracy/age fields and a `LIVE_GPS` path. Schema now specified in
[`docs/location_provenance.md`](docs/location_provenance.md).

#### Gap D — 505 frames / 70.8 s → **CONFIRMED, and re-characterised**

| Property | Value |
|---|---|
| Clip | `test_videos/accident_demo.mp4` |
| Frames / native FPS | 505 @ 30.0 = **16.8 s of footage** |
| Resolution | 640×360 |
| Model | `models/yolo11n.pt`, imgsz 640, `device: auto` (CPU — Intel Family 6 Model 190) |
| Sampling | `frame_stride: 1`, `target_width: 640`, `process_fps: None` (every frame) |
| Wall clock | 70.8 s, sequential decode, single process |
| Effective rate | **≈7.1 FPS analysed vs 30 FPS native → 4.21× slower than real time** |

**This is offline file throughput, not live response.** No frame was dropped
after capture; the run was simply slower than the footage's own frame rate. It
says nothing about how fast the system would react to an event on a live camera,
and it must never be quoted as a response time.

Stage timings do not exist: `PerfStats` carries `inference_ms`, `tracking_ms`,
`analysis_ms`, `total_ms` only. There is no capture→incident or
incident→notification measurement. Phase 3 scope.

#### Gap E — one accident, zero false alarms, proves nothing → **CONFIRMED**

| Item | Finding |
|---|---|
| Test suite | 17 modules, ~327 test methods, all passing |
| Regression scenarios | 9, all **synthetic scripted detections** |
| Real-world labelled data | **none exists** |
| `scripts/calibrate.py` | expects labelled folders that are not present |
| Real evidence | 1 clip, 1 incident, 505 frames, 1 camera, 1 event |

No accuracy figure and no false-alarm rate can be stated. One event with zero
false alarms is consistent with a perfect classifier *and* with one that alarms on
everything in a clip that happens to contain a single event. Phase 2 scope.

#### Gap F — call-failure-to-SMS fallback unspecified → **CONFIRMED**

There is **no `sms_on_call_failure` flag**. Full `EmergencyConfig` field list
inspected: `async_calls, call_log, contact_env_var, demo_mode,
emergency_instruction, enabled, expose_recipient, max_calls_per_run,
min_gap_seconds, poll_attempts, poll_interval_seconds, provider, request_timeout_seconds,
send_sms, simulated_duration_seconds, simulated_ring_after_seconds, sms_body,
sms_max_body_length, sms_provider, twilio_from_env_var, twilio_phone_env_var,
twilio_sid_env_var, twilio_token_env_var, voice_message, write_call_log`.

Behaviour is hard-coded: the SMS is `SKIPPED` whenever the call did not reach
the recipient's network, reason `"call was <STATE>: <reason>"`. That is the
*opposite* of the required fallback and is not operator-configurable.

### 1.3 Call / SMS function inventory

Full graph, inputs, outputs and side effects. Every entry point is
`DemoCallDispatcher.handle_incident(incident: Mapping) -> DemoCallRecord | None`,
reached from `AccidentPipeline._build_incident()` **only** on the
`verification.is_new_incident` branch.

| Layer | Function | In → Out | Real side effect? |
|---|---|---|---|
| gate | `_apply_gates` | record → void | no |
| gate | `_dial` (worker thread) | record → void | yes, via provider |
| call | `CallProvider.place(recipient, message, on_status)` | → `ProviderResult` | `SimulationProvider` **no** / `TwilioProvider` **yes** |
| call | `TwilioProvider.place` | → `ProviderResult` | **yes** — `POST /2010-04-01/Accounts/{sid}/Calls.json` with inline `Twiml` |
| call | `TwilioProvider._poll` | sid → status callbacks | **yes** — GET call resource, up to 15×4 s = 60 s blocking |
| sms | `SmsProvider.send(recipient, body)` | → `ProviderResult` | `SimulationSmsProvider` **no** / `TwilioSmsProvider` **yes** |
| sms | `TwilioSmsProvider.send` | → `ProviderResult` | **yes** — `POST /2010-04-01/Accounts/{sid}/Messages.json` |
| body | `build_sms_body(incident, instruction)` | → str | no |
| audit | `_write_log` | record → `demo_calls.jsonl` | no (disk append) |

**State transitions implemented today:** `INITIATED → RINGING → IN_PROGRESS →
COMPLETED` (+ `BUSY`, `NO_ANSWER`, `FAILED`) mapped from Twilio's documented
status enum `queued|ringing|in-progress|completed|busy|no-answer|failed|canceled`;
SMS `PENDING → SENT|FAILED|SKIPPED|BLOCKED`.

**Retries: none. Timeout: one HTTP read timeout (`request_timeout_seconds`, 10 s).
Error handling: HTTP/URL errors are caught, surfaced in `ProviderResult.error`
and logged; a raising provider is caught and turned into `FAILED`.**

#### Provider API verified against official documentation

Checked against the Twilio REST reference (read-only doc fetch, no API call):

| Item | Verified |
|---|---|
| Call create endpoint | `POST /2010-04-01/Accounts/{AccountSid}/Calls.json` — **matches code** |
| Inline `Twiml` parameter | supported, max 4000 chars — **matches code** |
| Call status enum | `queued, ringing, in-progress, canceled, completed, busy, no-answer, failed` — **matches `_STATE_MAP`** |
| `StatusCallback` / `StatusCallbackEvent` | **available but unused**; code polls instead |
| `Timeout` (ring seconds) | available, **unused** (default 60 s applies) |
| Idempotency on create | **not documented** → app-level idempotency is mandatory |
| `completed` semantics | *"a connection was established... can occur when answered by a person, an IVR phone tree menu, or even a voicemail"* |
| `answered_by` / `MachineDetection` | available, **unused** |

Two consequences carried into Phase 4: `COMPLETED` must never be rendered as
"a human was notified", and polling must give way to a signed webhook.

### 1.4 Missing code vs missing configuration (recorded separately)

**Missing code — must be written (Phase 3–5):**

| Gap | Phase |
|---|---|
| GPS/location provider interface, `LIVE_GPS`, fix acceptance policy | 3 |
| Stage + end-to-end latency instrumentation | 3 |
| Live/recorded mode distinction in the dashboard | 3 |
| Durable notification outbox (unique index, crash-safe) | 4 |
| App-level idempotency key | 4 |
| Bounded retry with backoff + circuit breaker | 4 |
| Signed provider webhook receiver | 4 |
| `NOT_CONFIGURED` state; removal of the silent demo fallback | 4 |
| `SMS_ON_CALL_FAILURE` policy flag + (b)(c)(d) behaviours | 4 |
| Kill switch / feature flags | 5 |
| Recipient directory, allowlist, consent, jurisdiction | 5 |
| Labeled evaluation set + leakage-safe splitter + PR-AUC/FPR/FNR | 2 |

**Missing configuration — operator must supply (cannot be written by me):**

| Item | Status |
|---|---|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, sender number | **ABSENT — BLOCKER for any real call/SMS** |
| Verified consenting test recipient | one exists in `.env`, **not yet confirmed by the operator** |
| Approved evaluation dataset | **ABSENT — BLOCKER for any accuracy claim** |
| Jurisdiction / dispatch authorisation | **ABSENT — emergency routing stays NOT_APPROVED** |

No secret was read into this report; only presence, length and country prefix.

### 1.5 Architecture and policy defined

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 15-box modular target, dependency rule (detection must never import notification layers), feature-flag table |
| [`docs/notification_state_machine.md`](docs/notification_state_machine.md) | lifecycle vocabulary, ASCII state diagram, `SMS_ON_CALL_FAILURE` behaviours (a)–(e), idempotency, retry classes, callback requirements |
| [`docs/location_provenance.md`](docs/location_provenance.md) | `CAMERA_REGISTERED` / `LIVE_GPS` / `LOCATION_UNAVAILABLE`, accuracy + fix age, fix rejection policy, prohibited practices |

Initial safety defaults chosen: no live provider sends, no public-emergency
routing, explicit location source labels, no accuracy or latency claims.

### 1.6 Privacy / security posture

| Aspect | Current |
|---|---|
| Secrets in env | yes — `.env`, git-ignored (`.env`, `.env.*`, `*.env`, `*.env.*`, `.envrc`) |
| `.env.example` | placeholders only, verified: no non-fictional phone token |
| Number masking | `+91*****3949` in logs, overlay, call log |
| Secret leakage test | a test fails the build if any non-ignored file contains a real phone number |
| Encryption at rest | **none** — `incidents.json`, `demo_calls.jsonl`, `evidence/` are plain files |
| Retention limits | **none** — `evidence.max_evidence_dirs` prunes by count only |
| Access control | **none** — filesystem ACLs only |
| Transport | HTTPS to the provider (TLS default); no inbound endpoint exists |

### 1.7 Verification run in this phase

| Command | Result |
|---|---|
| `python scripts\phase1_baseline_probe.py` | exit 0, **0 network attempts** |
| `python -m unittest discover -s tests -t .` | **327 passed**, exit 0 |
| `python scripts\e2e_check.py --real-footage` | **77/77**, exit 0 |
| `python scripts\validate_thresholds.py` | **9/9 scenarios**, exit 0 |
| `python scripts\check_readme_commands.py` | **14/14 commands**, exit 0 |

### 1.8 Files changed in Phase 1

Added (documentation + evidence only — **no engine behaviour changed**):
`REPORT.md`, `docs/architecture.md`, `docs/notification_state_machine.md`,
`docs/location_provenance.md`, `scripts/phase1_baseline_probe.py`,
`reports/phase1_baseline.json`.

Modified: none. No source, config, threshold or test was touched in Phase 1.

### 1.9 GO / BLOCKED decisions

| Phase | Decision | Condition |
|---|---|---|
| **2 — data & metrics** | **GO** | Can build the harness, annotation guide and leakage-safe splitter. Will report accuracy as **UNESTABLISHED** and label the synthetic set as non-production. Requires no operator input to *start*. |
| **3 — live video, location, latency** | **GO** | Can build the location interface, stage instrumentation and live/recorded separation, and produce honest p50/p95/max latency. **`LIVE_GPS` will be implemented as an interface only — no GPS hardware or API exists, so it will report `LOCATION_UNAVAILABLE`.** |
| **4 — provider adapters** | **GO for implementation, BLOCKED for live verification** | All code, mocks, `NOT_CONFIGURED`, outbox, idempotency, retries and webhook can be written and tested offline. The controlled real send to a consenting test number is **BLOCKED** pending credentials + operator confirmation. |
| **5 — routing & launch** | **BLOCKED** | Recipient directory has no vetted source; emergency-service routing has no jurisdiction, authorisation or agreement and stays **NOT_APPROVED**, returning `NO_APPROVED_RECIPIENT`. Feature flags, kill switch, audit log and launch checklist can still be built. |

### 1.10 Exact next-phase entry conditions

**PHASE=2** — entry conditions, all met:
1. ✅ Phase 1 report written with source-verified evidence.
2. ✅ Credential/authorisation/data blockers identified without exposing secrets.
3. ✅ Call/SMS fallback state machine specified.
4. ✅ GPS provenance schema specified.
5. ✅ GO/BLOCKED decisions recorded.

**BLOCKERS that must be resolved before any real communication (Phase 4
live-test step, Phase 5 launch):**

| # | Blocker | Owner |
|---|---|---|
| B1 | `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / sender number in the secrets manager | operator |
| B2 | Explicit operator confirmation that the configured recipient consents to test calls/SMS | operator |
| B3 | An approved, labelled evaluation dataset | operator |
| B4 | Jurisdiction, authorisation and agreement for any public-emergency routing | operator — **out of scope until supplied; routing stays NOT_APPROVED** |---

## 2026-10-02 — Phase 2: labelled data, calibration, honest quality metrics

**Status: PARTIAL** — harness complete and tested; **accuracy remains
UNESTABLISHED** because no approved labelled dataset exists. The rest of the
project proceeds; any production-quality claim is formally blocked.

**No calls/SMS sent. No public-emergency contact attempted or configured.**

### 2.1 What data actually exists (verified, not assumed)

| Asset | Size | Nature |
|---|---|---|
| `test_videos/accident_demo.mp4` | 3.44 MB, 505 frames @30 fps, 16.8 s | **real footage**, 1 event, camera unknown, event **not independently reviewed** |
| `test_videos/accident.mp4` | 175 KB, 90 frames | **synthetic drawing** — YOLO finds no real objects |
| `test_videos/normal_traffic.mp4` | 140 KB, 90 frames | **synthetic drawing** |
| `test_videos/suspicious.mp4` | 155 KB, 90 frames | **synthetic drawing** |
| `test_videos/street_photo.jpg` | 137 KB | single real still |
| `calibration/`, `data/`, `labels/`, `eval*` | — | **do not exist** |

`scripts/calibrate.py` documents a workflow requiring
`calibration/accident/*.mp4` and `calibration/normal/*.mp4`. **Neither folder
exists.** The script has therefore never been run on real data.

Total usable real video: **one clip, one event, one camera, one location.**

### 2.2 Harness delivered

New package `ai_engine/evaluation/`:

| Module | Responsibility |
|---|---|
| `manifest.py` | event-level label format, `quality` grades, validation, `quality_blockers()` |
| `splits.py` | `source_id`-grouped, class-stratified, seeded splits; `assert_no_leakage()` |
| `metrics.py` | confusion matrix, precision/recall/F1/FPR/FNR, **Wilson 95 % CIs**, PR curve, PR-AUC, imbalance |
| `evaluate.py` | runner, provenance hashing, threshold suggestion, human-readable report |

New entry point: `python scripts\evaluate.py`
New data: `evaluation/manifest.json` (the shipped-asset probe set)
New guide: [`docs/data_annotation_guide.md`](docs/data_annotation_guide.md)

### 2.3 Leakage control

The split unit is the **`source_id` group** (recording session / camera /
location / time block), never the clip and never the frame. `assert_no_leakage()`
raises if a group appears in two partitions. Verified:

```
leakage check  : PASS - no source_id appears in more than one partition
```

Splits are deterministic for a given `--seed`, and the report records
`data_fingerprint`, `threshold_set`, `model_weights_sha256`, `split_seed` so a
later run can prove it evaluated the same thing.

### 2.4 Measured result on the shipped assets — and why it means nothing

`python scripts\evaluate.py`:

```
clips          : 4 (1 real, 3 synthetic, 0 verified)
events         : 1 labelled accident event(s) across 4 source group(s)
class imbalance: positives 1 / negatives 3 (majority 75.0%)
leakage check  : PASS

event-level metrics
  [train]  TP=1 FP=0 FN=0 TN=3
      precision 1.000  [95% CI 0.206-1.000]  n=1
      recall    1.000  [95% CI 0.206-1.000]  n=1
      FPR       0.000  [95% CI 0.000-0.561]  n=3
      FNR       0.000  [95% CI 0.000-0.793]  n=1
```

**These numbers must not be quoted.** The 95 % CI on precision spans
0.21–1.00 from a single positive clip; the FPR spans 0.00–0.56 from three
synthetic negatives that contain no detectable object at all. The harness says so
in the report itself:

```
THIS SET MAY NOT SUPPORT A QUALITY CLAIM:
  * 3 clip(s) are synthetic and cannot measure detector quality
  * 4 clip(s) are not independently verified (grades: ['PROVISIONAL','SYNTHETIC'])
  * only 1 real clip(s); at least ~20 clips from >=5 distinct cameras/locations
    are needed before any rate is meaningful
  * only 4 independent source group(s); leakage-safe splits need >=5

CLAIMS withheld:
  - accuracy / false-alarm rate claims
  - any statement about detector accuracy on real-world footage
  - any claim that the thresholds are optimal
  - any calibrated-probability interpretation of 'accident_score'
```

**Recorded finding: accuracy is UNESTABLISHED.** Not "low", not "unknown" —
*unmeasured*. No number for detection rate, false-alarm rate, precision, recall
or PR-AUC may be published for SafeVision until a labelled set exists.

### 2.5 Threshold selection

The harness selects a threshold from the **validation split only**, and
`config.yaml` is never modified automatically.

On the shipped set it reports:

```
none: the validation split is empty or has no positive example, so no threshold
can be selected without tuning on data the detector was fitted to. This is
reported as 'no suggestion', not guessed.
```

This is a bug I introduced and then fixed: the first version silently fell back
to the *training* split while labelling itself "validation only", which would
have produced a flattering threshold. Now it refuses.

### 2.6 Calibration status

`accident_score` is an **uncalibrated evidence index**, not a probability. The
report prints that instruction verbatim. No reliability curve or ECE can be
computed without labels, so **no calibrated-probability claim is permitted**.

### 2.7 Bugs found and fixed in this phase

| Bug | Impact |
|---|---|
| `pr_auc` divided by the positive count after already computing a proportion | perfect separation scored **0.5 instead of 1.0** — a silently wrong metric |
| threshold suggestion fell back to the training split | would have tuned a safety gate on fitted data |
| `group_split` had leftover dead code (`del remaining`) | `UnboundLocalError` on any manifest with both positive and negative groups |
| rendered report omitted the uncalibrated-score warning | a reader could take the score for a probability |

### 2.8 Verification

| Command | Result |
|---|---|
| `python -m unittest tests.test_evaluation` | **32 passed** |
| `python scripts\evaluate.py` | exit 0, report written to `reports/evaluation_report.json` |
| `python -m unittest discover -s tests -t .` | see §2.9 |
| `python scripts\e2e_check.py --real-footage` | re-run in Phase 3 |

### 2.9 Files changed

Added: `ai_engine/evaluation/{__init__,manifest,splits,metrics,evaluate}.py`,
`evaluation/manifest.json`, `scripts/evaluate.py`,
`docs/data_annotation_guide.md`, `tests/test_evaluation.py`,
`reports/evaluation_report.json`.

Modified: none outside `ai_engine/evaluation/`. Detection, verification,
notification, config and the existing 327 tests were untouched.

### 2.10 Privacy / safety

* The evaluation runner **forcibly disables** `emergency.enabled` and
  `emergency.demo_mode` before building a pipeline, and a test patches
  `urlopen` to raise and asserts **zero** network attempts.
* No invented annotations: the single real event is graded `PROVISIONAL` and its
  note states the label came from the pipeline's own state transitions, i.e. it is
  circular and is excluded from quality claims.
* Synthetic clips are graded `SYNTHETIC` and structurally cannot support a
  production claim.

### 2.11 Blocker to close

| # | Blocker | Owner | Consequence while open |
|---|---|---|---|
| **B3** | Approved labelled evaluation set: ≥20 real clips, ≥5 source groups, ≥5 cameras/locations, ≥6 hard-negative categories, `VERIFIED` labels | operator | no accuracy or false-alarm figure may be published |

The annotation format, the closed subtype vocabulary, the hard-negative
taxonomy and the exact acceptance gate are documented in
[`docs/data_annotation_guide.md`](docs/data_annotation_guide.md) so the set can
be built without further design work.

### 2.12 Exact next-phase entry conditions

**PHASE=3** — all met:

1. ✅ Phase 1 completed (report, state machine, GPS schema).
2. ✅ Evaluation harness built, deterministic, leakage-safe, 32 tests passing.
3. ✅ Accuracy formally recorded as UNESTABLISHED with the gate that would lift it.
4. ✅ No regression to existing functionality.

**PHASE=3 will still be PARTIAL on one axis:** `LIVE_GPS` cannot be demonstrated
because this machine has no GPS receiver and the project has no geolocation
service. Phase 3 will implement the provider interface and the fix-acceptance
policy, and the engine will report `LOCATION_UNAVAILABLE` until a real fix
source is supplied. That is the correct behaviour, not a workaround.

---

## 2026-10-03 — Phase 3: live video, location provenance, end-to-end latency

**Status: PARTIAL** — latency instrumentation, live/recorded separation and the
location-provenance layer are implemented and tested. **A real live-source
latency measurement is BLOCKED** (no webcam, no reachable RTSP camera on this
machine), and **`LIVE_GPS` is not demonstrable** (no GNSS receiver, no
geolocation service wired in). Both are reported as blocked rather than
approximated.

**No calls/SMS sent. No public-emergency contact attempted or configured.**

### 3.1 Delivered

| Module | Responsibility |
|---|---|
| `ai_engine/telemetry/latency.py` | `LatencyLedger`, `StageTimer`, `percentiles()` (p50/p95/max + sample count, nearest-rank — a p95 is a real observed sample, never interpolated) |
| `ai_engine/location/location_provider.py` | `LocationProvider`, `CameraRegisteredProvider`, `GpsDeviceProvider`, `FixPolicy`, `LocationFix`, `GpsFix`, `LocationProviderRegistry` |
| `scripts/latency_bench.py` | recorded-file throughput + stage distributions |
| `scripts/live_latency_bench.py` | live-paced measurement, `--live-replay` and `--live-real` |
| `pipeline.ledger`, `pipeline.dashboard_fields()`, `pipeline.dropped_frames` | instrumentation wiring + dashboard contract |
| `extra.location_provenance` on every incident | provenance travels with the incident |

### 3.2 Clock discipline

| Concern | Treatment |
|---|---|
| Durations | `time.perf_counter()` (monotonic) — an NTP correction cannot skew a measurement |
| Event time | source timestamp, kept in a separate field (`media_time_seconds`) |
| Percentiles | nearest-rank, always with `count` attached |
| Failed stage | recorded as an **error**, never as 0 ms |

A report may not collapse "provider accepted the request", "a terminal call
state was reported" and "an SMS reached a handset" into one number. The stage
vocabulary keeps them separate: `notify_enqueue` ≠ `provider_request` ≠
`call_terminal` ≠ `sms_status`.

### 3.3 Measured — recorded file (`reports/latency_recorded.json`)

Baseline reproduction of the previously reported "505 frames / 70.8 s", now with
a measurement method attached.

| Stage | n | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|
| `detect` | 505 | 136.162 | 214.818 | 5185.230 |
| `track` | 505 | 1.010 | 2.192 | 1417.066 |
| `analysis` | 505 | 4.212 | 12.862 | 54.652 |
| `verify` | 505 | 0.040 | 0.076 | 0.860 |
| `incident_build` | 505 | 0.002 | 0.005 | 14.578 |

**Throughput: 6.18 fps analysed vs 30.0 fps source → 4.85× slower than real time.
Mode: `RECORDED`.**

`detect` dominates: its p50 alone (136 ms) exceeds the 33 ms budget of a 30 fps
frame by 4×. `analysis` at 4 ms and `verify` at 0.04 ms are irrelevant to
throughput; the multi-frame verification state machine costs essentially
nothing, which is the reassuring result here.

The `max` column is dominated by the first inference (model warm-up, 5.2 s) and
by two GC/OS pauses. Those are real samples and are reported rather than trimmed.

**This is offline throughput and must never be quoted as a live response time.**
The clip was already on disk; nothing was captured, encoded or transmitted.

### 3.4 Measured — live-paced (`reports/latency_live_replay.json`)

| Stage | n | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|
| `detect` | 250 | 108.492 | 156.989 | 7719.570 |
| `track` | 250 | 1.015 | 1.397 | 1965.248 |
| `analysis` | 250 | 3.551 | 5.534 | 6.858 |
| `verify` | 250 | 0.048 | 0.070 | 0.610 |
| `incident_build` | 250 | 0.003 | 0.005 | 17.574 |

**Result: target 30 fps → achieved 6.3 fps. `keeps_up = false`.**

Honest reading: on this CPU-only machine the engine processes roughly **6 fps of
640×360 with `yolo11n` @ imgsz 640**. Against a 30 fps camera it would need to
drop ~79 % of frames, or run at 640→ smaller input, or a lighter model, or a
GPU. `dropped_frames` exists precisely so that degradation is visible rather
than silent. (In this harness the counter reads 0 because a paced replay has no
queue — frames are pulled synchronously, so the measurement is of throughput,
not of queue behaviour.)

Mode is recorded as `LIVE` and the report carries:

> SIMULATED LIVE PACING over a recorded clip. No capture, no network, no encoder,
> no source clock. This measures how fast the ENGINE can process 640x360 frames
> on CPU under a fixed arrival rate. It is NOT a live response time.

**A real live source (`--live-real`) is BLOCKED on this machine:** no webcam and
no reachable RTSP camera. `scripts/live_latency_bench.py --live-real` exists and
is ready, but it has never produced a number, and none is claimed.

### 3.5 Location provenance

| Situation | `source` | `confidence` | Wording |
|---|---|---|---|
| Registered camera with coordinates | `CAMERA_REGISTERED` | `APPROXIMATE_CAMERA_POSITION` | "approximate, not a live GPS fix" |
| Registered camera, **no** coordinates | `LOCATION_UNAVAILABLE` | `NONE` | "registered but has no coordinates" |
| Unregistered camera | `LOCATION_UNAVAILABLE` | `NONE` | "not registered and no GPS fix is available" |
| Fresh accurate fix present | `LIVE_GPS` | `DEVICE_REPORTED` | "+/-8.0 m, age 0 s" |
| Fix stale / imprecise / out of bounds | rejected → falls back, **reason recorded** | `APPROXIMATE_CAMERA_POSITION` | "no usable live GPS fix; falling back to the registered position" |

Verified behaviour:

* A registered camera **without** coordinates reports `LOCATION_UNAVAILABLE`, not
  `CAMERA_REGISTERED` — a bug I introduced and fixed during this phase, where an
  entry with no coordinates still advertised a position.
* A rejected fix is preserved in `rejected_fixes`, so an operator can see that a
  fix existed and was refused instead of wondering why none appeared.
* A raising GPS reader degrades to the registered position; it never propagates
  and never invents a coordinate.
* **`IP_GEOLOCATION` is never produced.** `describe()` states
  `"rejected by policy - never used"`.
* `accuracy_m` / `radius_m` / `fix_timestamp` / `fix_age_seconds` are `null` for a
  camera position — a registered point genuinely has no error estimate, and
  emitting one would be a fabrication.

**No GPS hardware or geolocation API exists in this project.** The interface is
implemented and the `LIVE_GPS` path is exercised by an injected reader in tests;
in production the engine correctly reports `CAMERA_REGISTERED` or
`LOCATION_UNAVAILABLE`. No live GPS claim is made.

### 3.6 Dashboard contract

`pipeline.dashboard_fields()` and `extra.location_provenance` expose:
`processing_mode` (`LIVE` / `RECORDED` / `UNKNOWN`), `source_kind`,
`frames_processed`, `dropped_frames`, per-stage `p95_ms` **with sample counts**,
`incident_timelines_ms`, `latency_methodology` (clock, boundaries, percentile
rule, and the caveat that stage times do not bound the provider's or the
network's contribution), and the full location provenance block.

The methodology object is emitted with every report so a consumer cannot read a
p95 without also reading how it was measured.

### 3.7 Bugs found and fixed in this phase

| Bug | Impact |
|---|---|
| **`_build_incident` truncated by a stray `return payload`** | the `on_incident` callback loop became unreachable — `--json` output and every consumer callback silently stopped firing while incidents were still produced internally. Caught by the CLI-contract tests. |
| `notify_enqueue` recorded inside `_build_incident` referencing `t_build`, which does not exist there | `NameError` on the first confirmed incident |
| multi-line conditional expression in `__init__` | `SyntaxError`, pipeline unimportable |
| `PacedReplay` passed the whole `Config` to `VideoReader` | `AttributeError: read_retries` |
| `time.sleep()` with a negative remainder | `ValueError` whenever a frame arrived early |
| benchmark never told the pipeline the source kind | report header read `mode=UNKNOWN` while the footer read `RECORDED` |
| registered camera without coordinates reported `CAMERA_REGISTERED` | advertised a position it did not have |

### 3.8 Verification

| Command | Result |
|---|---|
| `python -m unittest tests.test_latency_location` | **41 passed** |
| `python -m unittest tests.test_cli_contract tests.test_pipeline_e2e tests.test_latency_location` | **74 passed** |
| `python -m unittest discover -s tests -t .` | see §3.9 |
| `python scripts\e2e_check.py --real-footage` | see §3.9 |

### 3.9 Files changed

Added: `ai_engine/telemetry/{__init__,latency}.py`,
`ai_engine/location/location_provider.py`, `scripts/latency_bench.py`,
`scripts/live_latency_bench.py`, `tests/test_latency_location.py`,
`reports/latency_recorded.json`, `reports/latency_live_replay.json`.

Modified: `ai_engine/pipeline/accident_pipeline.py` (ledger wiring,
`dashboard_fields()`, `dropped_frames`, incident provenance),
`ai_engine/location/__init__.py`, `scripts/latency_bench.py`,
`examples/integration_example.py` (`show_dashboard_fields`).

Detection, verification, scoring, camera registry and the notification providers
were **not** modified.

### 3.10 Blocker to close

| # | Blocker | Owner | Consequence while open |
|---|---|---|---|
| **B5** | A real live source (webcam or RTSP camera) to measure `--live-real` | operator | live latency stays unmeasured; only throughput and paced-replay numbers exist |
| **B6** | A GNSS receiver, vehicle tracker or geolocation API to exercise `LIVE_GPS` end to end | operator | `LIVE_GPS` is interface-only and unproven in production; the engine reports camera-registered or unavailable |

### 3.11 Explicitly NOT claimed

* ❌ 30–60 second incident-to-notification response — **not demonstrated**.
* ❌ "Keeps up with a 30 fps camera" — measured **false** at 6.3 fps on CPU.
* ❌ Live GPS — no device, no API, `LIVE_GPS` path unproven.
* ❌ That any stage time bounds when a call is answered or an SMS is read.

---

## 2026-10-03 — Phase 4: provider integration, durable delivery, gated tests

**Status: PARTIAL** — every code, safety and durability requirement is
implemented and covered by offline tests. **The live provider test is BLOCKED**:
no Twilio credentials exist, and no real call or SMS has ever been sent. No
delivery is claimed.

**No calls/SMS sent. No public-emergency contact attempted, configured or
reachable. The live-test script refuses to run and was verified to refuse.**

### 4.1 The Phase 1 critical finding is closed

`provider: auto` with no credentials used to resolve to the **simulation**
provider and report `call=COMPLETED, sms=SENT`. A dashboard could not tell that
from a real notification. Verified before and after:

```
BEFORE (Phase 1 probe)                     AFTER (Phase 4)
provider=auto, 0 credentials               provider=auto, 0 credentials
  -> call=simulation                         -> call=not_configured
  -> call=COMPLETED  sms=SENT                -> call=NOT_CONFIGURED  (ok=False)
  -> call_placed=True                        -> was_placed=False

provider=auto, DEMO_MODE=true              provider=auto, DEMO_MODE=true
  -> simulation                             -> simulation   (explicit, unchanged)
```

`build_provider` / `build_sms_provider` now take `demo_mode` explicitly. It is
never inferred: inference is what allowed the silent substitution in the first
place.

New terminal state `NOT_CONFIGURED` on both `CallStatus` and `SmsStatus`,
deliberately distinct from all three of:

| State | Means |
|---|---|
| `COMPLETED` / `SENT` | the provider reported success |
| `FAILED` | the provider refused or was unreachable |
| `NOT_CONFIGURED` | **there was no provider to ask** — nobody was told |

A fresh install cannot notify anybody: `allow_real_call`, `allow_real_sms`,
`webhook_enabled` and `allow_emergency_service_routing` all default to `false`.

### 4.2 Durable, exactly-once delivery

`ai_engine/emergency/outbox.py` — SQLite with `UNIQUE (incident_id, channel)`.
Exactly-once is a property of the **schema**, not of a code path remembering to
check, so it survives a restart and two workers on one camera.

| Property | Verified by |
|---|---|
| one row per `(incident, channel)` | `test_a_second_enqueue_does_not_create_a_second_row` |
| claim is exclusive | `test_claim_is_exclusive` |
| exactly one winner under 8 concurrent threads | `test_concurrent_claims_produce_exactly_one_winner` |
| a terminal row is never claimed again | `test_a_terminal_row_is_never_claimed_again` |
| a crashed worker's lease is reclaimed | `test_an_expired_lease_is_reclaimed` |
| a restart does not re-notify | `test_survives_a_reopen_of_the_database` |
| a gate-refused notification still leaves an audit row | `test_the_outbox_records_that_nobody_was_told` |
| no unmasked number is persisted | `test_no_secret_is_stored` |

The idempotency key is written **before** the provider request and is a
deterministic SHA-256 of `(incident, channel, attempt)` — deterministic so that a
retry after a crash produces the same key, and hashed so the incident id does not
leak to a third party.

### 4.3 Retry, breaker, and what is deliberately *not* retried

| Condition | Behaviour |
|---|---|
| HTTP 429, 425, 408, 5xx | transient → bounded retry with full jitter |
| `Retry-After` from the provider | honoured over our own backoff |
| HTTP 400 / 401 / 403 / 404 | permanent → **one** attempt, no sleeping |
| circuit breaker open | provider **not contacted at all** |
| a non-terminal `IN_FLIGHT` row | not claimable — a second worker cannot double-send |

Breaker: opens after N consecutive failures, half-opens after the cool-off, and
re-opens immediately if the trial fails. An open breaker is reported as
`BLOCKED` with `"circuit breaker open"` — not as a delivery failure, so an
operator does not go hunting for a bad phone number during a provider outage.

`test_an_open_breaker_stops_the_provider_being_called` asserts the provider is
never contacted; `test_open_breaker_does_not_stall_the_incident` asserts the
incident still terminates with a reason rather than hanging.

### 4.4 `SMS_ON_CALL_FAILURE` — explicit, configurable, tested

`emergency.sms_on_call_failure` (default `false`, preserving the original
behaviour). `emergency.sms_on_call_success` (default `true`).

| Call outcome | `sms_on_call_failure=false` | `=true` |
|---|---|---|
| `COMPLETED` / `IN_PROGRESS` | SMS sent | SMS sent |
| `BUSY` / `NO_ANSWER` (line rang) | SMS sent | SMS sent |
| `FAILED` / never initiated | **skipped** | one fallback SMS |
| blocked, breaker open, not configured | skipped | skipped |

Exactly one fallback SMS per incident is guaranteed by the outbox row, not by the
trigger: `test_a_fallback_sms_is_not_sent_twice`.

`both_channels_failed_policy` (`none` / `flag` / `escalate`) records the failure.
**`escalate` stops at a log entry and an outbox row.** It does not contact
police, ambulance or 112.

### 4.5 Provider callbacks — verified, not trusted

`ai_engine/emergency/webhook.py` implements Twilio's documented algorithm
(HMAC-SHA1 over the URL plus sorted POST params, base64, compared in constant
time), a replay window, HTTPS enforcement, and idempotent application.

| Check | Result on tampering |
|---|---|
| valid signature | accepted |
| forged signature | rejected |
| any parameter changed | signature invalid → rejected |
| wrong auth token | rejected |
| signature absent | rejected |
| `http://` when HTTPS required | rejected |
| replayed after the tolerance window | rejected |
| timestamp in the future | rejected |
| duplicate `(CallSid, CallStatus)` | logged no-op, not an error |
| callback for an unknown incident | ignored |

`test_completed_never_claims_a_human_answered` asserts the vocabulary never
contains "human" or "answered_by". Twilio's own documentation, verified in Phase 1:

> a completed call indicates that a connection was established … can occur when a
> call is answered by a person, an IVR phone tree menu, or even a voicemail.

`webhook_enabled` defaults to `false`; **no HTTP server is started anywhere in
this project.** The verification logic is complete and tested; a receiver would
be a Phase 5 deployment concern behind TLS termination.

### 4.6 Log correlation

Every attempt logs `incident`, `channel`, `provider`, `correlation_id`, `attempt`
and the provider resource id. One `correlation_id` per notification is shared by
the call row, the SMS row and both outbox rows
(`test_the_correlation_id_is_shared_by_both_channels`). The provider's own
message id is recorded (`test_the_outbox_records_the_provider_message_id`).

No secret or unmasked number is ever written to a record, the outbox or a log
(`test_credentials_are_never_written_into_the_record`, `test_no_secret_is_stored`).

### 4.7 Gated tests

**Layer (a) — mocked, runs on every test run (77 tests in this module, all
offline).** Every provider failure mode is exercised on the real dispatcher code
path: auth failure, rate limit, timeout, invalid number, delivery failure,
provider raising, busy line, no answer. `test_no_phase4_test_can_reach_a_network`
replaces `urlopen` with a raising stub and asserts zero attempts.

**Layer (b) — live, opt-in only: `scripts/live_provider_test.py`.**

```
$ python scripts\live_provider_test.py
[FAIL] 1. explicit --live flag            not passed; this script is a no-op without it
[FAIL] 2. TWILIO_ACCOUNT_SID present      missing from .env
[FAIL] 3. TWILIO_AUTH_TOKEN present       missing from .env
[FAIL] 4. sender number present           missing from .env
[FAIL] 5. consent flag ...=YES            the operator must set this to acknowledge
[FAIL] 6. recipient configured            EMERGENCY_CONTACT_LIVE_TEST_NUMBER missing
[FAIL] 7. recipient is NOT an emergency number
REFUSED. No provider was contacted and no message was sent.
exit=2
```

Eight gates: explicit `--live`, three credentials, operator consent
(`EMERGENCY_CONTACT_LIVE_TEST=YES`), a recipient, a **hard block** on emergency
numbers, and acknowledgement when the recipient is the project's real contact.
It never proceeds on a partial pass and never falls back to a simulation.

Emergency numbers are blocked, not warned about: `112`, `911`, `999`, `100`,
`+91 112`, `+44 999`, `00112` are all refused, while a mobile merely *ending* in
112 is correctly allowed.

The script reports **provider accepted the request** and **terminal status**
separately, and asks the operator out of band whether a human received anything —
recorded as `human_confirmed_received`. Provider acceptance is never presented as
delivery to a person.

### 4.8 Bugs found and fixed in this phase

| Bug | Impact |
|---|---|
| **`_build_incident` truncation** (found in Phase 3 verification) | `on_incident` callbacks and `--json` output silently stopped firing |
| **`TransientFailure` could not carry `retry_after`** | every rate-limit classification raised `TypeError` instead of `TransientFailure` |
| **outbox `claim` re-claimed live `IN_FLIGHT` rows** | two workers could both send — exactly-once was not actually enforced |
| **circuit breaker never gated the call path** | it only wrapped the SMS; a provider outage still dialled on every incident |
| **the provider's error was discarded when `on_status` had already set a terminal state** | `record.error` was empty, so an operator saw `FAILED` with no reason; found by `test_auth_failure_is_recorded_not_raised` |
| **gate-refused notifications left no outbox row** | "nobody was told" was indistinguishable from "never attempted" in an audit |
| `DEMO_MODE` snapshot in `__init__` | broke callers that flip the flag on a live dispatcher |
| `is_emergency_number` merged the country code into the short code | `+91 112` → `9112`, which was not blocked |
| SMS error text replaced the provider's diagnostic | `"unauthorised"` became a generic sentence, destroying the only clue |
| a simulation provider wired into a real-send config | would have reported a fake call as real; now `BLOCKED` |

### 4.9 Verification

| Command | Result |
|---|---|
| `python -m unittest tests.test_notification_reliability` | **77 passed** |
| `python -m unittest tests.test_emergency_call tests.test_emergency_notification tests.test_config` | **131 passed** |
| `python -m unittest discover -s tests -t .` | see §4.10 |
| `python scripts\e2e_check.py --real-footage` | see §4.10 |
| `python scripts\validate_thresholds.py` | see §4.10 |
| `python scripts\check_readme_commands.py` | see §4.10 |
| `python scripts\live_provider_test.py` | **exit 2 — correctly refused, 0 requests** |

### 4.10 Files changed

Added: `ai_engine/emergency/reliability.py`, `ai_engine/emergency/outbox.py`,
`ai_engine/emergency/webhook.py`, `scripts/live_provider_test.py`,
`tests/test_notification_reliability.py`.

Modified: `providers.py` (NotConfigured providers, `demo_mode` plumbing),
`dispatcher.py` (policy, outbox, retry, breaker, correlation id),
`call_record.py` (`correlation_id`), `schemas/enums.py` (`NOT_CONFIGURED`),
`config/classes.py` + `config.yaml` (14 new keys), `tests/helpers.py`
(`test_config` no longer writes a database), and three existing tests updated to
the new, safer behaviour with the reason recorded in each.

Detection, verification, scoring, tracking, camera registry and location were
**not** touched.

### 4.11 Blocker to close

| # | Blocker | Owner | Consequence while open |
|---|---|---|---|
| **B1** | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, sender number | operator | no real call/SMS is reachable; the live test refuses |
| **B2** | explicit operator confirmation that the recipient consents to test calls/SMS | operator | the consent gate cannot be satisfied |
| **B7** | a live recipient distinct from the project's real contact | operator | the acknowledgement gate cannot be satisfied |

### 4.12 Explicitly NOT claimed

* ❌ That any real call or SMS has been sent, accepted or delivered. **None has.**
* ❌ That a human has been reached — `COMPLETED` can mean an IVR or voicemail.
* ❌ That provider acceptance implies delivery.
* ❌ Any statement about SMS delivery latency or reliability.
* ❌ That the webhook receiver is deployed — it is verified logic, not a server.

---

## 2026-10-03 — Phase 5: routing, dispatch rules, deployment safety

**Status: PARTIAL** — routing, the decision log, operator override, deployment
and rollback documentation are implemented and tested. **Live dispatch is
BLOCKED** pending credentials and operator consent, so no real alert has ever
been produced.

**No calls/SMS sent. Public-emergency routing: NOT APPROVED, refused by
validation, and not implemented.**

### 5.1 Recipient selection is declared, not hard-coded

`dispatch/recipients.json` names the **environment variable** that holds each
number — never the number itself, so the file is safe to commit. A test asserts
that no `+\d{8,}` pattern appears in it.

`routing:` in `config.yaml` declares the rules:

```yaml
routing:
  dispatch_enabled: true         # master kill switch
  mode: simulation               # simulation | live
  directory: dispatch/recipients.json
  required_role: safety-officer
  required_jurisdiction: IN-TN
  require_location: false
  allow_emergency_service: false # NOT_APPROVED - refused by validation
```

Contacts carry roles, jurisdictions and a `consented_by` record. An
emergency-service contact with no consent record is **disabled on construction**,
so an unvetted one cannot be selected even by mistake.

### 5.2 Every routing decision is explained

`Router.route()` applies eight checks in a deliberate order — most decisive
first, because the point is that the first thing to refuse is the dangerous
thing:

| # | Check | Refusal code |
|---|---|---|
| 1 | `dispatch_enabled` (master kill switch) | `SUPPRESSED_BY_OVERRIDE` |
| 2 | operator suppression for this incident | `SUPPRESSED_BY_OVERRIDE` |
| 3 | incident is `CONFIRMED_ACCIDENT` | `NOT_CONFIRMED` |
| 4 | a directory is configured | `NO_DIRECTORY` |
| 5 | mode | — (simulation records only) |
| 6 | an emergency-service request | `EMERGENCY_ROUTING_NOT_APPROVED` |
| 7 | role / jurisdiction / consent eligibility | `NO_APPROVED_RECIPIENT` |
| 8 | a position is available, if required | `LOCATION_UNAVAILABLE` |

Every decision names its `rule`, its `reason`, every `evaluated` candidate with
a **non-reversible label** (`SAFETY-LEAD-01#6bf11976`, never digits), and — on
refusal — a written explanation an operator can act on. `trace.to_json()` is
asserted to contain no `+91` and no digits.

Verified live trace (`scripts/routing_report.py --mode live`):

```
outcome       : WOULD DISPATCH
rule applied  : eligible_contact
reason        : first eligible designated contact SAFETY-LEAD-01 of 1
[pass] incident_confirmed         {'status': 'CONFIRMED_ACCIDENT'}
[pass] eligibility                {'considered': [{'recipient_id': 'SAFETY-LEAD-01',
                                   'kind': 'DESIGNATED_CONTACT', 'eligible': True},
                                  {'recipient_id': 'FIREDUKE-EXAMPLE',
                                   'kind': 'EMERGENCY_SERVICE', 'eligible': False}]}
[pass] emergency_service_excluded {'excluded': []}
```

The emergency-service contact in the shipped directory is shown being excluded.

### 5.3 Public-emergency routing is refused three times over

1. `emergency.allow_emergency_service_routing: true` → **ConfigError at startup**.
2. `routing.allow_emergency_service: true` → **ConfigError at startup**.
3. An incident carrying `request_emergency_service` → `EMERGENCY_ROUTING_NOT_APPROVED`,
   and no recipient is returned.
4. An emergency contact present in the directory is excluded even if it claims
   consent, and a directory containing *only* emergency contacts is refused.

`both_channels_failed_policy: escalate` records the failure and stops. There is
no automatic escalation to any second party.

### 5.4 Human in the loop

`RoutingTrace.suppress(incident_id, reason)` stops dispatch for one incident,
with the reason logged. `unsuppress()` lifts it. Suppression is scoped to one
incident id and is tested not to affect another.

### 5.5 Location in the decision

`require_location: true` refuses a dispatch when no position exists, reporting
`LOCATION_UNAVAILABLE` rather than sending an alert with no location. The
provenance source is recorded in the decision
(`CAMERA_REGISTERED`, `LIVE_GPS`, `LOCATION_UNAVAILABLE`). **No live GPS exists
in this project**; the `LIVE_GPS` provider path is interface-only and the engine
correctly reports a camera-registered position or nothing.

### 5.6 Fail-safe defaults

| Requirement | Implementation | Test |
|---|---|---|
| unconfirmed incident never notified | check 3, before any recipient work | `test_an_unconfirmed_incident_is_never_routed` (5 statuses) |
| no automatic escalation | `escalate` logs and stops | `test_both_channels_failing_is_recorded_and_routes_nowhere` |
| no restart storm | the outbox row survives; `claim` returns `None` | `test_a_restart_cannot_replay_a_dispatch` (5 simulated restarts) |
| no self-test SMS/call path | a non-`CONFIRMED_ACCIDENT` payload is refused | `test_there_is_no_self_test_notification_path` |
| missing config disables the feature, does not fail silently | `NO_DIRECTORY` / `NOT_CONFIGURED` are explicit outcomes | `test_a_missing_directory_is_empty_and_says_so` |
| a simulation provider cannot serve a real-send config | `BLOCKED` with a reason | `test_a_real_send_request_cannot_be_served_by_the_simulation` |

A fresh install: `routing.mode = simulation`, `allow_real_call = false`,
`allow_real_sms = false`, `dispatch_enabled = true` but unreachable — **it can
notify nobody.**

### 5.7 Operations

[`docs/deployment.md`](docs/deployment.md) covers modes, the four switches, the
pre-flight checklist, secrets handling, PII and retention, the stop/rollback
procedure, an "an alert did not arrive" runbook, health signals, and the two
gated test layers.

Fastest stop, effective immediately with no restart:

```yaml
routing:
  dispatch_enabled: false
```

Retention is an honest gap: `incidents.json`, `evidence/`, `demo_calls.jsonl` and
the outbox are all **unbounded** by default, there is **no encryption at rest**
and **no access control** beyond filesystem ACLs. Those are prerequisites for
handling real incident footage under data-protection law, and the document says
so rather than implying they are handled.

### 5.8 Bugs found and fixed in this phase

| Bug | Impact |
|---|---|
| `str()` on a `str`-mixin `Enum` serialised as `DispatchRefusal.NOT_CONFIRMED` | decision logs and refusal counts used the wrong key |
| simulation mode produced `would_dispatch=True` | a dry run could have been mistaken for a real dispatch |
| the directory check ran after the mode check | a missing directory was invisible in simulation |
| two `emergency_service_excluded` entries, the first without the excluded list | an auditor could not see which contact was dropped |
| `kind` serialised as `ContactKind.DESIGNATED_CONTACT` | noisy, machine-hostile output |
| `DispatchMode("simulation")` raised `ValueError` | config vocabulary and enum vocabulary disagreed |
| **`routing:` block swallowed the four `twilio_*` keys** | `config.yaml` failed to load — caught immediately by the probe |
| persistent outbox made `e2e_check.py` non-re-runnable | second run skipped the notification for a reused incident id (a true exactly-once property, wrong for a self-test) |
| **the secret-leak test caught a real number I had added to a new test** | exactly the guard it exists for; the number was removed |

### 5.9 Verification

| Command | Result |
|---|---|
| `python -m unittest tests.test_dispatch_routing` | **29 passed** |
| `python -m unittest tests.test_notification_reliability` | **77 passed** |
| `python -m unittest discover -s tests -t .` | see §5.10 |
| `python scripts\e2e_check.py --real-footage` | see §5.10 |
| `python scripts\validate_thresholds.py` | see §5.10 |
| `python scripts\check_readme_commands.py` | see §5.10 |
| `python scripts\routing_report.py` | see §5.10 |
| `python scripts\live_provider_test.py` | **exit 2 — refused, 0 requests** |

### 5.10 Files changed

Added: `ai_engine/dispatch/{__init__,routing}.py`, `dispatch/recipients.json`,
`scripts/routing_report.py`, `tests/test_dispatch_routing.py`,
`docs/deployment.md`.

Modified: `config/classes.py` (`RoutingConfig` + validation), `config.yaml`
(`routing:` section), `scripts/e2e_check.py` (isolated outbox + the corrected
`NOT_CONFIGURED` expectation), `tests/test_notification_reliability.py` (number
removed), `README.md` (§24).

Detection, verification, scoring, tracking, camera registry, latency and
notification providers were **not** touched.

### 5.11 Blockers to close before any live alert

| # | Blocker | Owner |
|---|---|---|
| **B1** | Twilio credentials | operator |
| **B2** | recipient consent to test calls/SMS | operator |
| **B7** | a live recipient distinct from the project's real contact | operator |
| **B8** | encryption at rest and access control for incident footage | operator |
| **B9** | a retention policy for incidents, evidence, call log and outbox | operator |
| **B4** | jurisdiction/authorisation for any emergency-service routing | **out of scope** — routing stays NOT_APPROVED |

### 5.12 Explicitly NOT claimed

* ❌ That any alert has ever reached a person.
* ❌ That `COMPLETED` means a human answered.
* ❌ That SafeVision can route to police, ambulance or 112.
* ❌ That live GPS exists or that a position is a live fix.
* ❌ That accuracy or false-alarm rate is known.
* ❌ That the 30–60 s response target is met.

---

## 2026-10-03 — Final verification across all five phases

Every command run against the project as it now stands.

| Command | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | **511 passed**, exit 0 |
| `python scripts\e2e_check.py --real-footage` | **79/79**, exit 0 (idempotent across repeat runs) |
| `python scripts\validate_thresholds.py` | **9/9 scenarios**, exit 0 |
| `python scripts\check_readme_commands.py` | **14/14 commands**, exit 0 |
| `python scripts\evaluate.py --describe-only` | exit 0, `usable_for_quality_claims: False` |
| `python scripts\routing_report.py` | exit 0, `SIMULATION`, `would_dispatch: False` |
| `python scripts\live_provider_test.py` | **exit 2 — refused**, 8 gates reported, 0 requests |

Test count by phase: 327 at the start of this work → **511** (+184 new tests).
The pre-existing 327 were never modified except where a required behaviour change
made an assertion obsolete; in each of those three cases the test was rewritten
to assert the new, safer behaviour and the reason was recorded in the phase
report.

### Bugs found and fixed across all five phases

19 distinct bugs, of which these were in existing code rather than new code:

| Bug | Consequence |
|---|---|
| **`_build_incident` truncated by a stray `return`** | `on_incident` callbacks and `--json` output silently stopped firing while incidents were still produced |
| **the provider's error was dropped when `on_status` had already set a terminal state** | an operator saw `FAILED` with no reason at all |
| **the outbox could re-claim a live `IN_FLIGHT` row** | exactly-once was not actually enforced under concurrency |
| **the circuit breaker never gated the call path** | a provider outage still dialled on every confirmed incident |
| **a gate-refused notification left no outbox row** | "nobody was told" was indistinguishable from "never attempted" |
| **persistent outbox made self-tests non-rerunnable** | e2e's second run skipped its own notification |
| **the secret-leak guard caught a real phone number I had added to a test** | exactly what it exists for |

### Final state of the project

```
511 tests                     79 e2e checks        9 threshold scenarios
 14 readme commands            5 documented phases   0 real notifications sent
```

Nothing in this project has ever placed a call or sent a message. Every
provider code path has been exercised with mocks; none has been exercised
against a live provider, because doing so requires credentials and an explicit
operator act.

---

## 2026-10-03 — Hotfix: "detection works but no call or SMS"

Reported symptom: the engine reaches `CONFIRMED_ACCIDENT`, then the console shows
`demo emergency call: INITIATED` and `demo emergency sms : PENDING` and nothing
appears to happen.

**The notification workflow was in fact running correctly.** `demo_calls.jsonl`
for the reported incident already recorded `COMPLETED` with the full
`RINGING -> IN_PROGRESS -> COMPLETED` history and an `SENT` SMS. Two independent
defects made that invisible and, in one configuration, made the flag itself a
no-op.

### Root cause 1 — the run summary reported a pre-call snapshot

`main.py` printed the summary *before* `pipeline.close()` joined the notification
worker. The incident payload is a snapshot taken when the incident was built, so
it always read `INITIATED` / `PENDING` even though the provider had already
dialled. Fixed by waiting for the notification first, then reporting the live
record via `DemoCallDispatcher.final_states()`.

### Root cause 2 — `--demo-call` was silently overridden by `.env`

`--demo-call` set `config.emergency.demo_mode = True`, but the dispatcher reads
`DEMO_MODE` from the **environment** on every access. With `DEMO_MODE=false` in
`.env` the environment won, the gate reported `SKIPPED`, and no call was placed
at all. The flag's own help text promised otherwise. Fixed by setting
`os.environ["DEMO_MODE"]` as well, so the CLI is authoritative for the run.

### Root cause 3 — a real call could be truncated at shutdown

`AccidentPipeline.close()` joined the worker with `request_timeout_seconds`
(10 s, an HTTP read timeout). A real Twilio call rings far longer than that, so
REAL MODE could exit mid-call. Replaced with `notification_wait_seconds`
(75 s), and `DemoCallDispatcher.wait()` now reports how many notifications were
still unresolved.

### Root cause 4 — `allow_real_call` blocked real mode

`config.yaml` shipped `allow_real_call: false`, so `DEMO_MODE=false` could never
send even with valid credentials - the exact mode the operator asked for. The
library defaults stay `False` (importing SafeVision can never notify anybody);
`config.yaml` now sets both allow-flags true, leaving credentials as the real
gate.

### Root cause 5 — a static config rule misfired

`allow_real_call and provider == "simulation"` was rejected at startup, which
broke the documented demo configuration. The check now lives only in the runtime
gate, which knows whether `DEMO_MODE` is actually on.

### Changes

| File | Change |
|---|---|
| `main.py` | `--demo-call`/`--no-demo-call` set the environment too; wait for the notification before reporting; report the live final state; print the verbatim SMS and correlation id |
| `ai_engine/emergency/dispatcher.py` | `wait()`, `pending()`, `final_states()`; `close()` releases the owned outbox handle |
| `ai_engine/pipeline/accident_pipeline.py` | `wait_for_notifications()`; join with `notification_wait_seconds` |
| `ai_engine/config/classes.py` | `notification_wait_seconds`; removed the misfiring static rule |
| `config.yaml` | `notification_wait_seconds: 75.0`; `allow_real_call`/`allow_real_sms: true` |
| `examples/integration_example.py` | same env fix; prints the final call/SMS states |
| `tests/test_notification_workflow.py` | **new, 19 tests**: call-before-SMS ordering, exactly-once, restart replay, SMS content, provider failures never reported as success |
| `tests/test_emergency_call.py`, `test_emergency_notification.py` | two tests updated from obsolete "demo off means no call" semantics to the explicit disabled path, plus a new test asserting REAL MODE does dial |

### Verification

| Command | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | **533 passed**, exit 0 |
| `python scripts\e2e_check.py --real-footage` | **79/79**, exit 0 |
| `python scripts\validate_thresholds.py` | **9/9**, exit 0 |
| `python scripts\check_readme_commands.py` | **14/14**, exit 0 |
| `python main.py --source test_videos\accident_demo.mp4 --demo-call` | CONFIRMED_ACCIDENT -> call `COMPLETED` -> SMS `SENT` |
| `python main.py --check-emergency` | reports `not_configured` and names the three missing credentials |
| `python scripts\live_provider_test.py` | exit 2, refused, 0 requests |

Detection thresholds were **not** changed: `validate_thresholds.py` still reports
9/9 and the confirmed score is unchanged at 0.44.

---

## 2026-10-03 — Workflow visibility: the ANALYZE stage and the whole pipeline, made visible

Reported symptom: "the system detects and confirms an accident, but the ANALYZE
stage and the emergency call / SMS are not displayed."

Nothing in the detection or notification logic was missing. Both worked. What was
missing was **any way to see them while they happened**: the run printed start-up
lines, then went quiet for ~70 seconds, then printed a summary built from a
snapshot taken *before* the provider was contacted.

### Root cause

There was no live feed from the pipeline to the display. Specifically:

1. **No transition callback.** The state machine recorded its transitions in a
   list that was only read after the whole video had been processed, so the
   ANALYZE and VERIFICATION stages could not be displayed as they occurred.
2. **No analysis payload.** Even replaying a transition would have shown only a
   state name - not the confidence, the signals, the vehicles, the negative
   evidence or the reason.
3. **The notification was rendered from a stale snapshot.** The incident payload
   is captured when the incident is built, so it always read `INITIATED` /
   `PENDING`.
4. **No visual output by default.** Neither the preview window nor the annotated
   video was requested unless a flag was passed, and nothing said so.
5. **`--show` failed silently.** On a headless OpenCV build the window cannot
   open; the old code logged one warning and continued, producing no visuals at
   all.
6. **A terminal run looked like a hung process** between start-up and summary.

### Changes

| File | Change |
|---|---|
| `ai_engine/pipeline/accident_pipeline.py` | `on_transition` callback; every state change now carries an `analysis` block (confidence, physical/temporal evidence, per-component scores, agreeing signals, evidence sufficiency, negative evidence, blocking reasons, frames/seconds held, collision candidate) |
| `ai_engine/visualization/workflow.py` | **new.** `WorkflowReporter` - the ordered 10-stage workflow, each stage marked reached / active / refused, with the real data for that stage |
| `main.py` | reporter wired to the transition, incident and call-status feeds; waits for the notification before reporting; prints the verbatim SMS and the saved report paths; live progress line; `--show` falls back to an annotated video when no GUI backend exists; explicit "how to see visuals" hint; `_DropNoisyDeprecations` |
| `ai_engine/location/location_provider.py` | a missing camera id now falls back to the registry default, matching `LocationManager.get()` |
| `examples/integration_example.py` | prints the final call and SMS states |
| `tests/test_workflow_display.py` | **new, 19 tests** for the display contract and the transition hook |

### Bugs found and fixed while doing this

| Bug | Consequence |
|---|---|
| provenance reported `LOCATION_UNAVAILABLE` with `camera_id: null` while the incident's own location block correctly said "Chennai Main Road" | two parts of one incident contradicted each other, and the SMS said `CAMERA-REGISTERED` while the console said unavailable |
| `SMS GENERATED` stayed permanently pending | the fast path goes straight to `SENT` without an intermediate `PENDING`, so the stage was never marked |
| the confirmation block printed "0 frames held over 0.0s" | the counter resets on the transition, so it contradicted the "1.8s of verification" in the reason line |

### Result

Every stage now renders with its actual data, and the run ends with an ordered
checklist - `[x]` reached, `[!]` refused, `[ ]` never happened. A quiet clip
leaves the notification stages pending rather than claiming success.

---

## 2026-10-03 — "it detects but does not call my number"

**This was configuration, not a fault.** Two independent reasons, both verified:

1. `DEMO_MODE=true` in `.env` selects the **simulation** provider. The
   `RINGING / IN_PROGRESS / COMPLETED` transitions were simulated; nothing left
   the machine and no number was dialled.
2. `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` and
   `TWILIO_PHONE_NUMBER` are **empty**, so a real send is impossible.
   `main.py --check-emergency` reports `real call possible: False`.

**The defect this did expose:** the console printed `CALL COMPLETED` for a
simulated call, which reads exactly like a delivered call. A working demo looked
like a broken notification path. Fixed:

* a `SIMULATION MODE` banner before any frame is processed
* simulated transitions tagged `[SIM]`, with an explicit
  `SIMULATED - no phone call was placed` / `no SMS was sent to any real number`
* a real-mode banner naming the missing credentials and stating
  `NOTHING WILL BE SENT`
* a closing reminder when the run was simulated

No credentials were invented and nothing was sent.

### Separately: two e2e checks were asserting the wrong property

The calm opening of `accident_demo.mp4` produces two consecutive frames above
`verification.suspicious_score` (physical evidence 0.391 and 0.518 against a 0.30
threshold), so the state machine correctly opens an investigation on frames
14-15 and correctly drops it again.

Two checks failed because they asserted "SUSPICIOUS must never appear". That is
the wrong property: `NORMAL -> SUSPICIOUS -> VERIFYING -> CONFIRMED_ACCIDENT` is
the designed state machine, and opening an investigation is supposed to notify
nobody. Corrected to assert what actually matters - calm footage reaches **no
incident**, **no call**, **no SMS**, and never escalates to `CONFIRMED_ACCIDENT`.

**No detection threshold was changed.** `suspicious_score` is still 0.30,
`verify_entry_score` 0.34, `confirm_score` 0.42; `validate_thresholds.py` still
reports 9/9. The frame states are still reported, as an informational row.

---

## 2026-10-05 — Fix: `live_provider_test.py` crashed on a parser/gate mismatch

**Symptom**

```
$ python scripts\live_provider_test.py --live --channel sms
AttributeError: 'Namespace' object has no attribute 'acknowledge_real_number'
```

**Root cause.** `gate_report()` read `args.acknowledge_real_number`, but the
parser declared `--i-understand-this-will-ring-the-real-number`, whose argparse
dest is `i_understand_this_will_ring_the_real_number`. The two names never
matched, so every gated run died at the acknowledgement gate - i.e. *after* the
operator had passed `--live` and supplied credentials, which is precisely when
the gate matters.

**Fixes**

1. Extracted `build_parser()` so the CLI definition is testable on its own.
2. Added `--acknowledge-real-number` as the explicit, readable flag, with
   `dest="acknowledge_real_number"`; `--i-understand-this-will-ring-the-real-number`
   is kept as an alias so existing invocations still work.
3. `gate_report()` now reads every flag through `getattr(args, name, False)`.
   This is **fail-closed**: if the parser and the gate ever disagree again, the
   run *refuses* instead of raising. A crash must never be mistaken for "gates
   passed", and a flag that cannot be read must never satisfy the
   acknowledgement.
4. Gate 8's message names the new flag and says what to do.
5. `--help` gained an epilogue listing all eight gates.
6. Docs updated: `docs/deployment.md` (new "Live-test flags" section) and
   `README.md` §24.

**No safety gate was removed or weakened.** All eight gates remain, consent is
still `EMERGENCY_CONTACT_LIVE_TEST=YES`, the recipient still comes from
`EMERGENCY_CONTACT_LIVE_TEST_NUMBER`, and emergency numbers are still hard-blocked.

**Regression tests** (`tests/test_notification_reliability.py`, 7 new):

* every attribute `gate_report` reads must exist on the real parser, for five
  different argv shapes - this is the test that would have caught the bug
* `--live --channel sms` reaches the gates without raising
* both flag spellings set `acknowledge_real_number`
* a namespace *missing* the attribute refuses rather than crashing
* the real-contact acknowledgement gate still exists and still blocks
* the consent flag is still required
* emergency numbers are still refused

**Result:** 560 unit tests, 80/80 e2e, 9/9 threshold scenarios, 14/14 documented
commands. The gated SMS run now stops at gate 8 and refuses, contacting nothing.

---

## 2026-10-05 — Fix: Twilio trial-account SMS rejected with error 572006

**Symptom.** The gated live SMS test passed all eight safety gates, then Twilio
rejected the request:

```
HTTP 400
code: 572006
message: "Invalid template name. Trial accounts can only use predefined SMS templates."
```

**Root cause.** SafeVision POSTed a free-form ``Body`` to
``POST /2010-04-01/Accounts/{sid}/Messages.json``. A Twilio **trial account**
cannot send free-form SMS: the message must use a predefined Twilio Messaging
Template, addressed by its SID as ``ContentSid``. Both the production provider
(``TwilioSmsProvider.send``) and the gated live test built that same free-form
request, so both would have failed identically.

**Fix.**

1. Added ``build_message_payload()`` in ``ai_engine/emergency/providers.py`` -
   the single source of truth for the request shape. With a template it sends
   ``To``, ``From``, ``ContentSid`` (+ optional ``ContentVariables``); without
   one it sends ``To``, ``From``, ``Body`` as before. ``Body`` and
   ``ContentSid`` are mutually exclusive, which Twilio requires.
2. ``TwilioSmsProvider`` gained ``content_sid`` / ``content_variables``, read
   from ``TWILIO_CONTENT_SID`` / ``TWILIO_CONTENT_VARIABLES`` via config keys
   ``twilio_content_sid_env_var`` / ``twilio_content_variables_env_var``. Only
   the *variable names* live in ``config.yaml``; values stay in ``.env``.
3. ``scripts/live_provider_test.py`` now imports the **same** builder, so the
   live test can never pass with a request shape production would reject.
4. Errors are reported honestly and actionably: Twilio's own ``code`` and
   ``message`` are parsed out of the error body and preserved verbatim, with a
   hint appended for 572006 (set ``TWILIO_CONTENT_SID``), 21612 / 21614
   (recipient not verified) and 21211 (alphanumeric sender). An unknown error is
   passed through unchanged.
5. The live test prints a warning when ``TWILIO_CONTENT_SID`` is unset, so a
   free-form send on a trial account is visible before the request is made.

**The auth token is never exposed.** It is not placed in the payload, and
``explain_sms_error`` composes text from Twilio's response only. Tests assert
the token appears in neither the payload nor any error text.

**No detection logic was touched.** ``validate_thresholds.py`` still reports 9/9;
``suspicious_score`` 0.30, ``verify_entry_score`` 0.34, ``confirm_score`` 0.42,
``temporal_confirm_score`` 0.40, ``release_score`` 0.22 - all unchanged, and
asserted by ``test_verification_thresholds_are_unchanged``.

**Regression tests** (``tests/test_twilio_trial_template.py``, 19 tests): error
572006 is recognised and explained; Twilio's wording is preserved; unknown
errors pass through; the payload uses ``ContentSid`` instead of ``Body`` when a
template is set and vice versa; ``ContentVariables`` is only sent with a
template; the provider and the live test build identical requests; the provider
reads the template from the environment; the token never appears in a payload or
an error; and the verification thresholds and incident SMS body are unchanged.

**Verification.** 579 unit tests, 80/80 e2e, 9/9 threshold scenarios, 14/14
documented commands.

**No SMS was sent while developing this fix.** The payload shapes and the error
translation were verified offline. Whether Twilio now accepts a real request
depends on a real template being configured, which only the operator can do.
