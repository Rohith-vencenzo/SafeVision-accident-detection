# SafeVision — Deployment, Operation and Rollback

Phase 5 operations guide. Everything here assumes you have read
[`notification_state_machine.md`](notification_state_machine.md) and
[`location_provenance.md`](location_provenance.md).

## 1. Modes

| Mode | What it does | Set with |
|---|---|---|
| `simulation` | decides and records routing; **sends nothing** | `routing.mode` in `config.yaml` |
| `live` | may send to a **consented designated contact** | `routing.mode: live` **plus** `emergency.allow_real_call`/`allow_real_sms` |

Default is `simulation`. Reaching `live` requires four independent switches to
agree, so no single edit can cause an alert to leave the machine:

```
routing.mode == live
  AND emergency.allow_real_call == true
  AND emergency.allow_real_sms  == true
  AND provider credentials resolve to a real provider
```

If any one is false, the incident records `NOT_CONFIGURED` or `SKIPPED` and
nobody is told.

## 2. What is NOT available in any mode

* **Public-emergency routing.** No police, ambulance, fire or 112 integration
  exists, there is no jurisdiction and no authorisation. Setting
  `emergency.allow_emergency_service_routing: true` **fails config validation**
  rather than being ignored. A directory that lists an emergency-service contact
  is logged and excluded on every routing decision.
* **Automatic escalation.** `both_channels_failed_policy: escalate` records the
  failure and stops. It does not contact anyone else.
* **Live GPS.** Coordinates come from `cameras.yaml` and are labelled
  `CAMERA_REGISTERED`. There is no GNSS receiver or geolocation service wired in.

## 3. Pre-flight checklist

Run in order. Any failure means **do not start the service**.

```bat
:: 1. configuration is valid and nothing dangerous is enabled
python main.py --check-config

:: 2. every gate refuses to send
python -m unittest tests.test_dispatch_routing tests.test_notification_reliability

:: 3. detection still behaves
python scripts\validate_thresholds.py
python scripts\e2e_check.py --real-footage

:: 4. the live test refuses without credentials and consent
python scripts\live_provider_test.py
```

Step 4 must print `REFUSED` and exit **2**. If it ever reaches the provider
without `--live`, that is a bug — stop and report it.

## 4. Secrets

| Secret | Lives in | Never |
|---|---|---|
| Twilio SID, auth token | `.env` or the platform secret store | in `config.yaml`, in code, in a log, in a screenshot |
| Recipient number | `.env` as `EMERGENCY_CONTACT_NUMBER` | in `dispatch/recipients.json` (only the *variable name* is stored) |
| Consent acknowledgement | operator action, `EMERGENCY_CONTACT_LIVE_TEST=YES` | committed |

`.gitignore` covers `.env`, `.env.*`, `*.env`, `*.env.*`, `.envrc`, with
`!.env.example`. A test fails the build if a non-ignored file contains a real
phone number, so a leak cannot be merged unnoticed.

Rotate the auth token if it appears anywhere it should not: it is a bearer
credential, so rotation is the only reliable remedy.

## 5. Personal data and retention

| Artefact | Contains | Default retention | Notes |
|---|---|---|---|
| `incidents.json` | incident id, timestamps, score, camera, location | **unbounded** | add an operator retention job |
| `evidence/` | frames of public road traffic | pruned by `max_evidence_dirs` | count-based, not age-based |
| `demo_calls.jsonl` | masked recipient, provider status, correlation id | **unbounded** | number masked unless `expose_recipient` |
| `notification_outbox.sqlite3` | masked recipient, provider refs, states | **unbounded** | the delivery audit trail |
| logs | masked recipient, correlation id | by log rotation | never a token or full number |

There is **no encryption at rest** and **no access control** in this project —
filesystem ACLs only. Both are prerequisites for handling real incident footage
in a jurisdiction with data-protection law.

Set `expose_recipient: false` before sharing `incidents.json` or
`demo_report.json` with anyone: the number then leaves only in masked form.

## 6. Rollback / stop procedure

### Stop sending immediately

Fastest and safest — no restart, no state change:

```yaml
routing:
  dispatch_enabled: false      # master kill switch, effective at once
```

Equivalent, without touching config: kill the process. Notifications are sent
from a daemon worker; stopping the process stops all sending.

### Stop sending but keep detecting

```yaml
emergency:
  enabled: false        # no dispatcher is constructed; the number is never read
```

Incidents are still detected, scored and written. Nothing is dialled.

### Undo a bad dispatch

The outbox row is the record. It cannot be edited into a different outcome — by
design. To re-notify an incident after a false alarm:

1. Confirm the incident really was a false alarm.
2. `RoutingTrace.suppress(incident_id, reason)` records the decision to notify
   the suppression, with a reason.
3. Clear the outbox row deliberately (`DELETE FROM notification_outbox WHERE
   incident_id = ?`) and record why, or issue a *new* incident rather than
   re-sending the old one.

Never edit a `DONE` row to `PENDING` silently: exactly-once exists so a restart
cannot turn into an alert storm.

### Rolling back a version

There is no `.git` in this checkout, so **take a copy before changing anything**:

```bat
xcopy /E /I SafeVision SafeVision.bak-2026-10-03
```

Then restore that directory. The outbox is a separate file — keep it, or the
engine will treat previously-notified incidents as new.

## 7. Runbook: an alert did not arrive

Work down this list; each step names the evidence to look for.

1. **Was it confirmed?** `incidents.json` → `status` must be
   `CONFIRMED_ACCIDENT`. `SUSPICIOUS` / `VERIFYING` / `FALSE_ALARM` never
   notify, by design.
2. **Was the feature on?** `demo_mode` and `allow_real_call`.
3. **Was a provider resolved?** `NOT_CONFIGURED` in the incident payload means
   **nobody was told** — this is the state to look for after a misconfiguration.
4. **Was the circuit breaker open?** Look for `circuit breaker open` in the
   logs; `breaker.to_dict()` gives the state and remaining seconds.
5. **Did the provider accept it?** `notification_outbox.sqlite3` → the
   `provider_ref` and `provider_status` for that incident.
6. **Was it deduplicated?** The row may already be `DONE` from an earlier run.
   That is the exactly-once guarantee working, not a bug.

`CALL -> COMPLETED` is **not** proof a person answered. Twilio documents that
`completed` can mean an IVR or voicemail answered. Only the recipient can confirm.

## 8. Health checks

| Signal | Healthy | Unhealthy |
|---|---|---|
| `detect` p95 latency | < frame budget (33 ms @ 30 fps) | above it — see Phase 3: this build runs ~6 fps on CPU |
| dropped frames | near zero on a live source | rising — the engine cannot keep up |
| outbox `PENDING` backlog | empty | growing — workers are blocked or the breaker is open |
| breaker state | `closed` | `open` — the provider is failing; sending is refused |
| `NOT_CONFIGURED` count | zero in `live` mode | any occurrence means nothing was sent |
| refused-routing count | any value is fine, but should be read | a spike means incidents are not reaching people |

## 9. Two gated test layers

| Layer | Command | When | Contacts a provider? |
|---|---|---|---|
| mocked | `python -m unittest discover -s tests -t .` | every change | **never** |
| live | `python scripts\live_provider_test.py --live --channel sms` | before a release, with consent | yes — gated by 8 checks |

The live test reports provider acceptance and terminal status **separately**, and
asks the operator out of band whether a person received anything. It must never
be run against a public-emergency number; those are hard-blocked by
`is_emergency_number()`.

### Live-test flags

```bat
python scripts\live_provider_test.py --help

:: the standard live SMS check
python scripts\live_provider_test.py --live --channel sms

:: additionally required when the live-test recipient IS the project's own
:: contact (EMERGENCY_CONTACT_LIVE_TEST_NUMBER == EMERGENCY_CONTACT_NUMBER)
python scripts\live_provider_test.py --live --channel sms --acknowledge-real-number
```

`--i-understand-this-will-ring-the-real-number` remains accepted as an alias for
`--acknowledge-real-number`. Gate 8 exists because the recipient in
`EMERGENCY_CONTACT_LIVE_TEST_NUMBER` is often the same number as
`EMERGENCY_CONTACT_NUMBER` — a real test message to a real contact should be a
conscious decision, not a default.

Every gate is read defensively: a flag that cannot be read is treated as **not
given**, so a parser/gate mismatch degrades to a refusal rather than a crash or
a false pass.

### Twilio trial accounts: `TWILIO_CONTENT_SID`

A Twilio **trial** account cannot send free-form SMS. Twilio answers a
free-form `Body` with:

```
HTTP 400
code: 572006
message: "Invalid template name. Trial accounts can only use predefined SMS templates."
```

The fix is to send a **predefined Twilio Messaging Template**, addressed by its
SID as `ContentSid`:

| `.env` | request sent | works on |
|---|---|---|
| `TWILIO_CONTENT_SID` **set** | `To`, `From`, `ContentSid`, `ContentVariables` | trial **and** paid |
| `TWILIO_CONTENT_SID` **empty** | `To`, `From`, `Body` (the incident text) | paid only |

Both are produced by one shared function, `build_message_payload()`, so the live
test and the production path can never send different shapes. `Body` and
`ContentSid` are mutually exclusive — Twilio rejects them together.

Find the SID in the Twilio Console under **Messaging → Templates**; it starts
with `HX`. Optional template variables go in `TWILIO_CONTENT_VARIABLES` as JSON.

**This applies to the production path too.** `TwilioSmsProvider.send()` uses the
same builder, so on a trial account the incident alert is sent as the template
rather than the composed body — the incident data still reaches the recipient,
through the template you approved. Note the consequence: with a template, the
incident id, date, location and severity appear as **template variables**, not as
the free-form text, so the template must contain those placeholders. On a paid
account nothing changes and the full incident body is sent as-is.

Other trial-account failures reported with an actionable hint:
`21612`/`21614` (recipient not verified) and `21211` (alphanumeric sender).