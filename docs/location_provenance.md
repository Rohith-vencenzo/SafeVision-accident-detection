# SafeVision — Location Provenance Schema

Defines how every coordinate that leaves the engine is labelled, so that a
camera's registered position can never be mistaken for a live GPS fix of the
accident.

## Why this exists

A fixed CCTV camera cannot report where it *is* — that fact is installed with the
camera. SafeVision reads coordinates from `cameras.yaml`. A responder reading an
incident must be able to tell instantly whether a position is:

* the **camera** (approximate — could be tens of metres from the crash), or
* a **live fix** (from a device that actually knows where it is).

Conflating them is a safety problem, not a cosmetic one: it sends a responder to
the wrong place with false confidence.

## `location` block in the incident contract

```json
"location": {
  "source": "CAMERA_REGISTERED",
  "camera_id": "CAM-001",
  "name": "Chennai Main Road",
  "latitude": 13.0827,
  "longitude": 80.2707,
  "accuracy_m": null,
  "radius_m": null,
  "fix_timestamp": null,
  "fix_age_seconds": null,
  "captured_at": "2026-10-02T18:31:35.482913+00:00",
  "is_camera_registered": true,
  "confidence": "APPROXIMATE_CAMERA_POSITION"
}
```

### `source` — the only field that must never be guessed

| Value | Meaning | Produced by |
|---|---|---|
| `CAMERA_REGISTERED` | Coordinates were configured for the camera. **Approximate.** | `LocationManager` (current behaviour) |
| `LIVE_GPS` | A real fix from a GPS/GNSS receiver or geolocation API. | `GpsLocationProvider` (Phase 3, **not yet implemented**) |
| `LOCATION_UNAVAILABLE` | No usable position. Reported explicitly rather than guessed. | any provider that cannot fix |
| `IP_GEOLOCATION` | *Rejected by policy.* Coarse, often city-level, and would be misread as a fix. | never produced |

### Required companion fields

| Field | `CAMERA_REGISTERED` | `LIVE_GPS` |
|---|---|---|
| `accuracy_m` | `null` — a registered point has no error estimate | required, metres (1σ) |
| `radius_m` | `null` | required, search/uncertainty radius |
| `fix_timestamp` | `null` — the coordinates were never "fixed" at a time | required, ISO-8601 UTC of the fix |
| `fix_age_seconds` | `null` | required — age at incident time |
| `captured_at` | required — when the incident was created | required |
| `confidence` | `APPROXIMATE_CAMERA_POSITION` | `DEVICE_REPORTED` |

## Fix acceptance policy (Phase 3)

A `LIVE_GPS` fix is **rejected** — downgraded to `LOCATION_UNAVAILABLE` with a
reason — when any of these hold:

| Condition | Rationale |
|---|---|
| `fix_age_seconds > max_fix_age_seconds` (default 30) | a fix from a parked phone is not where the car is now |
| `accuracy_m > max_accuracy_m` (default 100) | a 500 m fix is worse than the camera position |
| `radius_m > max_radius_m` (default 200) | cell/Wi-Fi-derived radii are city-scale |
| latitude outside `[-90, 90]` or longitude outside `[-180, 180]` | invalid |
| provider reports `no_fix` / `permission_denied` | no data |
| timestamp is in the future beyond clock-skew tolerance (default 5 s) | bogus clock |

When rejected, the engine **may** fall back to `CAMERA_REGISTERED` — but the
block must then say so explicitly and set `confidence:
APPROXIMATE_CAMERA_POSITION`. It must never silently substitute one for the
other.

## Prohibited

* Inferring coordinates from the source IP address.
* Fabricating, snapping or "correcting" a coordinate to look precise.
* Calling a camera position "live GPS", "current location" or "exact location"
  in any log, SMS, voice script or dashboard field.
* Reporting `LIVE_GPS` without `accuracy_m`, `fix_timestamp` and
  `fix_age_seconds`.

## Backwards compatibility

`LocationInfo.is_camera_registered` and `.source` already exist and are used by
`cameras.yaml` consumers. `source` is being widened from
`"camera_metadata"` / `"unregistered"` to the vocabulary above. The mapping is:

| old `source` | new `source` |
|---|---|
| `camera_metadata` (coordinates present) | `CAMERA_REGISTERED` |
| `camera_metadata` (no coordinates) | `CAMERA_REGISTERED` + `confidence: NONE` |
| `unregistered` | `LOCATION_UNAVAILABLE` |

A MINOR schema bump carries this; existing consumers reading
`is_camera_registered` keep working.

## Current status

**`CAMERA_REGISTERED` is implemented and honest. `LIVE_GPS` does not exist.**
Phase 1 verified by grep that no GPS, NMEA, geolocation or IP-geolocation code
exists anywhere in the repository. Phase 3 adds the provider interface and the
`LIVE_GPS` path; until then the engine must keep saying "camera-registered".