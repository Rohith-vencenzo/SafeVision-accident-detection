# Call / SMS Notification State Machine and Fallback Policy

Defines the states a notification passes through, exactly what each state
*means*, and the policy for sending an SMS when the call does not succeed.

This document exists because the current implementation has **no fallback** and
reports **false success** when credentials are absent. Both are Phase 4 work; the
design is fixed here so it can be tested against.

## 1. Vocabulary — never conflate these

| Term | Means | Never means |
|---|---|---|
| `REQUEST_ACCEPTED` | The provider's API returned 2xx and a resource id | that a phone rang, or a person answered |
| `QUEUED` | The provider accepted the request and has not dialled yet | anything about the recipient |
| `RINGING` | The provider reports the destination is ringing | that a human picked up |
| `IN_PROGRESS` | A connection was established and audio is flowing | that a *human* answered (could be IVR/voicemail) |
| `COMPLETED` | A connection was established and ended normally | that a human answered, or that anyone was told anything |
| `BUSY` | The destination's line was busy | — |
| `NO_ANSWER` | Ringing, not answered within the provider timeout | — |
| `FAILED` | Could not be completed as dialed (bad number, carrier reject, auth) | — |
| `SMS_ACCEPTED` | Provider accepted the message for delivery | that it arrived |
| `SMS_DELIVERED` | Carrier reported handset delivery | — |
| `SMS_UNDELIVERED` | Carrier could not deliver | — |

> **Twilio documentation, verified Phase 1:** *"A completed call indicates that a
> connection was established, and audio data was transferred. This can occur when
> a call is answered by a person, an IVR phone tree menu, or even a voicemail."*
> The engine therefore never reports "a human was notified". `answered_by` and
> `MachineDetection` are the only fields that can speak to that, and they are
> `null` unless answering-machine detection was requested.

## 2. Per-incident state machine

One row per `incident_id`. Every channel transition is applied to that row, so
exactly-once is structural rather than a convention.

```
                    ┌──────────────────────────────────────────┐
                    │            NOT_ATTEMPTED                 │
                    └───────────────┬──────────────────────────┘
                                    │ incident CONFIRMED_ACCIDENT
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  BLOCKED  (safety interlock / no config)  │  ← terminal, nothing sent
                    └──────────────────────────────────────────┘

        ┌───────────────────────────────────────────┐
        │  CALL_INITIATING                          │
        └───┬───────────────────────────────┬───────┘
   2xx +id │                        reject/timeout│ (no id)
            ▼                                  ▼
   ┌────────────────────┐            ┌────────────────────────┐
   │  CALL_QUEUED       │            │  CALL_INIT_FAILED      │
   └───┬────────────────┘            └───────────┬────────────┘
       │                                          │ policy (a)
       ▼                                          ▼
   ┌────────────────────┐            ┌────────────────────────┐
   │  CALL_RINGING      │            │  FALLBACK_SMS_PENDING  │
   └───┬────────────────┘            └───────────┬────────────┘
       │                                          │
       ▼                                          ▼
   ┌────────────────────┐            ┌────────────────────────┐
   │  CALL_IN_PROGRESS  │            │  SMS_ACCEPTED           │
   └───┬────────────────┘            └───────────┬────────────┘
       │                                          │
       ▼                                          ▼
   ┌────────────────────┐            ┌────────────────────────┐
   │  CALL_COMPLETED    │            │  SMS_DELIVERED / UNDEL  │
   └────────────────────┘            └────────────────────────┘
       │
       │ BUSY / NO_ANSWER / FAILED  (terminal, non-success)
       ▼
   ┌────────────────────────┐
   │  CALL_TERMINAL_UNANSWERED │
   └───────────┬────────────┘
               │ policy (b)
               ▼
   ┌────────────────────────┐
   │  FALLBACK_SMS_PENDING  │
   └────────────────────────┘
```

### Terminal states

`CALL_COMPLETED`, `BUSY`, `NO_ANSWER`, `FAILED`, `CALL_INIT_FAILED`,
`BLOCKED`, `SMS_DELIVERED`, `SMS_UNDELIVERED`.

## 3. `SMS_ON_CALL_FAILURE` policy

Default: **`false`** (matches today's behaviour). When `true`:

| # | Situation | Call state | SMS behaviour |
|---|---|---|---|
| (a) | Call API **rejects** (4xx) or **times out** with no call id | `CALL_INIT_FAILED` | Enqueue **one** fallback SMS |
| (b) | Call later reports `BUSY` / `NO_ANSWER` / `FAILED` via verified callback or bounded terminal timeout | `CALL_TERMINAL_UNANSWERED` | Enqueue **one** fallback SMS **only if no SMS was already sent** |
| (c) | Call reaches `IN_PROGRESS` or `COMPLETED` | `CALL_COMPLETED` | Send **one** primary SMS (subject to `primary_sms_on_success`, default `true`) |
| (d) | Call still `QUEUED` past `call_pending_timeout_seconds` | `CALL_PENDING_TIMEOUT` | Apply `pending_timeout_policy`: `fallback_sms` \| `wait` (default `wait`) |
| (e) | A late/duplicate callback arrives after a fallback SMS was sent | any | **No second SMS.** Idempotent no-op, logged |

**Exactly-once is enforced by the row, not by the trigger.** A single
`notification_outbox` row per `(incident_id, channel)` with a unique index.
Duplicate callbacks, worker restarts and retries all collapse onto that row.

## 4. Idempotency

* App-level: `idempotency_key = f"sv-{incident_id}-{channel}-{attempt}"`, stored
  before the request and checked before any retry.
* Provider-level: **Twilio's Calls and Messages create endpoints are not
  documented as idempotent** (verified against the REST reference in Phase 1), so
  the app key is authoritative and the row is the single source of truth.

## 5. Retries

Retry only transient, safe-to-retry errors:

| Error class | Retry? |
|---|---|
| HTTP 429 rate limit | yes, bounded backoff, honour `Retry-After` |
| HTTP 5xx | yes, bounded backoff |
| connection reset / timeout **before any id was returned** | yes, bounded |
| HTTP 400 / 401 / 403 / 404 | **no** — permanent |
| timeout **after** an id was returned | **no** — the resource exists; poll it instead |

Bounds: `retry.max_attempts` (default 3), `retry.base_delay_seconds` (default 2),
exponential with full jitter, plus a per-recipient rate limit and a circuit
breaker that opens after `breaker.failure_threshold` consecutive failures.

## 6. Provider callbacks

Phase 1 verified the code **polls** (`_poll`, `poll_attempts=15`,
`poll_interval_seconds=4` → up to 60 s of blocking) and registers **no**
`StatusCallback`. Twilio supports `StatusCallback` + `StatusCallbackEvent`
(`initiated`, `ringing`, `answered`, `completed`) on call creation, and
`StatusCallbackEvent` on messages.

Phase 4 replaces polling with a verified webhook receiver:

| Requirement | Detail |
|---|---|
| Signature | `X-Twilio-Signature`, HMAC-SHA1 of the full URL + sorted POST params, keyed with the auth token |
| Replay protection | reject if timestamp older than `webhook.tolerance_seconds` (default 300) |
| Matching | callback must reference a known `CallSid` / `MessageSid` we issued |
| Idempotency | a state transition already applied is a logged no-op |
| Transport | HTTPS only; endpoint is opt-in and disabled by default |

## 7. Current implementation vs this design

| Capability | Today (verified Phase 1) | Target (Phase 4) |
|---|---|---|
| Real Twilio call | code exists, unreachable (no creds) | reachable, gated |
| Real Twilio SMS | code exists, unreachable (no creds) | reachable, gated |
| Missing creds in `auto` mode | **silently simulated + reports success** | `NOT_CONFIGURED` |
| Missing creds in `twilio` mode | `BLOCKED` with a reason | same (already correct) |
| Durable outbox | none (in-memory dict) | SQLite outbox, unique index |
| Idempotency key | none | app-level key + row check |
| Retry / backoff | none | bounded, transient-only |
| Callback receiver | none | signed webhook |
| `SMS_ON_CALL_FAILURE` | not configurable; SMS always skipped on call failure | explicit policy flag |
| Kill switch | none | feature flag + file flag |
| Recipient allowlist / consent | none | allowlist + jurisdiction policy |