# SafeVision — Target Architecture

The modular decomposition the existing code is migrated towards, one service per
concern. Every box marked **exists** is already working code; boxes marked
*new* are Phase 3–5 work. Nothing here is a rewrite: the detection and
verification core is retained as-is.

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. SOURCE / DECODER                             ai_engine/video/    │
│     VideoReader  (file | webcam | rtsp)   [exists]                   │
│     -> exposes: SourceKind {RECORDED, LIVE}, decode ts, queue depth  │
├──────────────────────────────────────────────────────────────────────┤
│  2. FAST DETECTION                             ai_engine/detection/ │
│     YoloDetector / ScriptedDetector          [exists]                │
│     PerfStats.inference_ms                    [exists]               │
├──────────────────────────────────────────────────────────────────────┤
│  3. TRACKING + ANALYSIS                                                  │
│     ByteTracker   tracking/                   [exists]                │
│     MotionAnalyzer motion/                    [exists]                │
│     TrajectoryAnalyzer trajectory/            [exists]                │
│     CollisionAnalyzer collision/              [exists]                │
│     TemporalFusion temporal/                  [exists]                │
├──────────────────────────────────────────────────────────────────────┤
│  4. INCIDENT VERIFICATION                     ai_engine/verification/│
│     FalseAlarmPrevention (state machine)      [exists]                │
│     -> the ONLY producer of CONFIRMED_ACCIDENT                        │
├──────────────────────────────────────────────────────────────────────┤
│  5. SEVERITY + EVIDENCE                                                 │
│     SeverityAnalyzer severity/               [exists]                │
│     EvidenceCapture  evidence/               [exists]                │
├──────────────────────────────────────────────────────────────────────┤
│  6. LOCATION PROVIDER                      ai_engine/location/       │
│     LocationManager  (CAMERA_REGISTERED)     [exists]                │
│     GpsLocationProvider (LIVE_GPS)                [PHASE 3 - new]     │
│     FixPolicy (staleness / accuracy / bounds)      [PHASE 3 - new]     │
│     provenance labels                      [Phase 1 schema defined]   │
├──────────────────────────────────────────────────────────────────────┤
│  7. INCIDENT LIFECYCLE                                                  │
│     IncidentResult (schema v1.3.0)           [exists]                │
│     states: NORMAL → SUSPICIOUS → VERIFYING →                         │
│             CONFIRMED_ACCIDENT | FALSE_ALARM          [exists]      │
│     immutable audit trail                    [exists]                │
├──────────────────────────────────────────────────────────────────────┤
│  8. RECIPIENT / DISPATCH DIRECTORY                                 *new*│
│     RecipientDirectory (vetted, versioned, jurisdiction-aware)          │
│     DesignatedTestContact | EmergencyServiceContact                     │
│     -> NO_APPROVED_RECIPIENT when the directory has no verified entry  │
│     -> EMERGENCY_DISPATCH: NOT_APPROVED (hard block)                   │
├──────────────────────────────────────────────────────────────────────┤
│  9. NOTIFICATION OUTBOX (transactional, exactly-once)            *new*│
│     SQLite outbox, unique index (incident_id, channel)                 │
│     claim -> in_flight -> done/failed, crash-safe                      │
│     idempotency_key stored before any provider request                 │
├──────────────────────────────────────────────────────────────────────┤
│ 10. CALL ADAPTER                ai_engine/emergency/providers.py      │
│     DemoCallAdapter   (simulation)          [exists - rename]         │
│     TwilioCallAdapter (real REST)           [exists - harden]        │
│     -> real mode returns provider states, never fabricates success    │
├──────────────────────────────────────────────────────────────────────┤
│ 11. SMS ADAPTER                 ai_engine/emergency/providers.py      │
│     DemoSmsAdapter    (simulation)           [exists - rename]        │
│     TwilioSmsAdapter  (real REST)            [exists - harden]        │
├──────────────────────────────────────────────────────────────────────┤
│ 12. NOTIFICATION ORCHESTRATOR        ai_engine/emergency/dispatcher.py │
│     DemoCallDispatcher -> EmergencyNotifier      [exists - extend]     │
│     call-then-SMS sequencing, SMS_ON_CALL_FAILURE policy               │
├──────────────────────────────────────────────────────────────────────┤
│ 13. PROVIDER CALLBACK HANDLER                                      *new*│
│     signature verification, replay window, idempotent state apply      │
│     Twilio webhook receiver (opt-in, HTTPS only)                      │
├──────────────────────────────────────────────────────────────────────┤
│ 14. METRICS / DASHBOARD                    ai_engine/visualization/    │
│     OverlayRenderer (live HUD)            [exists]                    │
│     Live vs recorded badge                [PHASE 3 - new]             │
│     stage latency p50/p95/max             [PHASE 3 - new]             │
│     call/SMS state + location source      [exists, extend]            │
├──────────────────────────────────────────────────────────────────────┤
│ 15. EVALUATION / CALIBRATION                  scripts/                │
│     calibrate.py / validate_thresholds.py    [exists]                 │
│     labeled-split evaluator + PR-AUC/FPR/FNR      [PHASE 2 - new]     │
│     leakage-safe split by source/camera/event     [PHASE 2 - new]     │
└──────────────────────────────────────────────────────────────────────┘
```

## Dependency rule

Dependencies point **downward only**. The detection/verification core
(boxes 2–5) must never import from boxes 8–13: an incident must be decidable
with the network and the telephony provider entirely absent. That property is
what makes the test suite hermetic and it must be preserved.

## Feature flags (Phase 5)

| Flag | Fresh-install default |
|---|---|
| `features.detection` | `true` |
| `features.real_call` | **`false`** |
| `features.real_sms` | **`false`** |
| `features.sms_on_call_failure` | `false` |
| `features.emergency_service_routing` | **`false`** (and `NOT_APPROVED`) |
| `features.webhook_receiver` | `false` |

External communication is disabled by default; a fresh install cannot send
anything.