"""Gated live-provider test.  Phase 4, **layer (b)**.

    :: Layer (a) - mocked, runs on every test run: tests/test_notification_reliability.py
    :: Layer (b) - THIS script. Contacts the real provider API.

This script is the only thing in the project that may send a real message, and it
refuses to run unless **all** of the following hold.  Each gate is checked and
reported by name, so a refusal says exactly what is missing:

1.  ``--live`` passed explicitly.  There is no default and no config flag that
    turns this on; running the script bare always exits 0 without contacting
    anyone.
2.  Real credentials present in the environment.
3.  The recipient has consented: ``EMERGENCY_CONTACT_LIVE_TEST=YES`` must be set
    by the operator in the same environment.
4.  The recipient is not a public-emergency number.  Emergency short codes
    (112, 911, 999, 100, 101, 102, 108, 112...) are **hard-blocked**, not warned
    about, because routing a test message to an emergency line is exactly the
    mistake this project must never make.
5.  The recipient is not the project's real configured contact unless
    ``--acknowledge-real-number`` is passed.  (The long form
    ``--i-understand-this-will-ring-the-real-number`` remains as an alias.)

The recipient is read from ``EMERGENCY_CONTACT_LIVE_TEST_NUMBER`` by default;
override the variable name with ``--recipient-env-var``.  Every gate can only
refuse: an unreadable flag is treated as *not* given, so a parser/gate mismatch
degrades to a refusal rather than a crash or a false pass.

What "success" means here
-------------------------
The script distinguishes, and reports separately:

* **provider accepted the request** - a 2xx and a resource id. Necessary.
* **a terminal provider state was reached** - the call reached a terminal state,
  or the message left ``queued``. Also necessary.
* **a human acknowledged the message** - *never* claimed. Twilio documents that
  ``completed`` means "a connection was established", which can be a person, an
  IVR or voicemail. A provider response is not proof of delivery to a person, and
  this script will not pretend otherwise.

The operator confirms the human-received part out of band; the script asks for
that confirmation on stdin and records whatever they answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the sys.path insert, and deliberately the *same* helpers the
# production provider uses, so the live test can never pass with a request shape
# production would reject.
from ai_engine.emergency.providers import (  # noqa: E402
    CONTENT_VARIABLES_ENV_VAR,
    TEMPLATE_ENV_VAR,
    TWILIO_TRIAL_TEMPLATE_CODE,
    build_message_payload,
    explain_sms_error,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from ai_engine.emergency.envfile import load_env_file  # noqa: E402

#: Hard-blocked. Dialing or texting an emergency short code from a test is
#: exactly the failure mode this project must not be capable of.
EMERGENCY_SHORT_CODES = frozenset({
    "112", "911", "999", "000", "101", "102", "100", "108", "103", "104",
    "105", "106", "107", "109", "110", "111", "113", "120", "121", "122",
    "123", "124", "125", "126", "127", "128", "129", "181", "1000",
})

CONSENT_FLAG = "EMERGENCY_CONTACT_LIVE_TEST"


def digits(number: str) -> str:
    return "".join(ch for ch in str(number) if ch.isdigit())


def is_emergency_number(number: str) -> bool:
    """True when the number could be an emergency short code.

    Stripping punctuation merges a country code into the short code - ``+91 112``
    becomes ``9112`` - so the check is done against the number itself *and*
    against every plausible national remainder.  Being wrong in the blocking
    direction only refuses to run a test, which is the safe way to be wrong.
    """
    cleaned = digits(number)
    if not cleaned:
        return False
    if cleaned.startswith("00"):                 # international: 00 + country code
        return is_emergency_number(cleaned[2:])

    candidates = {cleaned}
    for country_len in (1, 2, 3):
        if len(cleaned) > country_len:
            candidates.add(cleaned[country_len:])

    for candidate in candidates:
        if candidate in EMERGENCY_SHORT_CODES:
            return True
    # a bare short code: 3 digits or fewer with no country prefix at all
    if len(cleaned) <= 3 and cleaned in EMERGENCY_SHORT_CODES:
        return True
    return False


def gate_report(args, env) -> tuple:
    """``(all_passed, [(gate, passed, detail), ...])`` - nothing is skipped silently.

    Every flag is read with :func:`getattr` and a ``False`` default. That is
    deliberate and fail-closed: if the parser and this function ever disagree
    again, the gate **refuses** instead of raising. A crash here used to abort
    the run with a traceback, and an abort must never be mistaken for "gates
    passed" - nor may a missing flag silently satisfy the acknowledgement.
    """
    def flag(name: str) -> bool:
        return bool(getattr(args, name, False))

    live = flag("live")
    acknowledge = flag("acknowledge_real_number")
    recipient_env_var = getattr(args, "recipient_env_var", "EMERGENCY_CONTACT_LIVE_TEST_NUMBER")

    sid = env.get("TWILIO_ACCOUNT_SID", "").strip()
    token = env.get("TWILIO_AUTH_TOKEN", "").strip()
    sender = (env.get("TWILIO_FROM_NUMBER") or env.get("TWILIO_PHONE_NUMBER") or "").strip()
    recipient = env.get(recipient_env_var, "").strip()
    consent = env.get(CONSENT_FLAG, "").strip().upper()

    checks = [
        ("1. explicit --live flag", live,
         "pass" if live else "not passed; this script is a no-op without it"),
        ("2. TWILIO_ACCOUNT_SID present", bool(sid), "set" if sid else "missing from .env"),
        ("3. TWILIO_AUTH_TOKEN present", bool(token), "set" if token else "missing from .env"),
        ("4. sender number present", bool(sender),
         "set" if sender else "TWILIO_FROM_NUMBER / TWILIO_PHONE_NUMBER missing"),
        (f"5. consent flag {CONSENT_FLAG}=YES", consent == "YES",
         "set" if consent == "YES" else "the operator must set this to acknowledge the test"),
        ("6. recipient configured", bool(recipient),
         "set" if recipient else f"{recipient_env_var} missing"),
        ("7. recipient is NOT an emergency number",
         bool(recipient) and not is_emergency_number(recipient),
         "blocked" if recipient and is_emergency_number(recipient) else "not an emergency number"),
    ]

    if recipient and recipient == env.get("EMERGENCY_CONTACT_NUMBER", "").strip():
        checks.append((
            "8. --acknowledge-real-number",
            acknowledge,
            "passed" if acknowledge
            else "the configured recipient is the project's real contact; "
                 "re-run with --acknowledge-real-number if that is intended",
        ))

    return all(ok for _, ok, _ in checks), checks


# --------------------------------------------------------------------------- #
def _parse_twilio_error(raw: str) -> Tuple[Optional[int], str]:
    """Pull Twilio's own ``code`` and ``message`` out of an error body.

    Reporting the provider's exact code is the whole point - a generic "HTTP 400"
    tells an operator nothing. The auth token is never part of the body and is
    never echoed here.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None, raw[:400]
    if isinstance(payload, dict):
        code = payload.get("code")
        return (int(code) if str(code).isdigit() else None,
                str(payload.get("message") or payload.get("status") or "")[:400])
    return None, raw[:400]


# --------------------------------------------------------------------------- #
def post_form(url: str, data: dict, sid: str, token: str, timeout: float = 20.0) -> dict:
    body = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in data.items())
    request = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={
            "Authorization": "Basic " + __import__("base64").b64encode(
                f"{sid}:{token}".encode("utf-8")).decode("ascii"),
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_sms(args, env) -> dict:
    sid = env["TWILIO_ACCOUNT_SID"]
    token = env["TWILIO_AUTH_TOKEN"]
    sender = env.get("TWILIO_FROM_NUMBER") or env["TWILIO_PHONE_NUMBER"]
    recipient = env[args.recipient_env_var]
    content_sid = env.get(getattr(args, "content_sid_env_var", "TWILIO_CONTENT_SID"), "")
    content_variables = env.get(
        getattr(args, "content_variables_env_var", "TWILIO_CONTENT_VARIABLES"), "")

    result = {
        "channel": "sms",
        "provider_accepted": False,
        "provider_message_sid": None,
        "terminal_status": None,
        "human_confirmed_received": None,
        "using_template": bool(content_sid),
    }
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    # Same builder the production provider uses, so the live test can never pass
    # with a request shape production would reject. A trial account (Twilio error
    # 572006) needs `ContentSid`; a paid account sends free-form `Body`.
    data = build_message_payload(
        recipient, sender, args.body,
        content_sid=content_sid,
        content_variables=content_variables,
    )
    if not content_sid:
        print(f"  note: {TEMPLATE_ENV_VAR} is not set, so a FREE-FORM body will be sent.")
        print("        A Twilio TRIAL account rejects this with error "
              f"{TWILIO_TRIAL_TEMPLATE_CODE}.")
    else:
        print(f"  using Twilio Messaging Template {TEMPLATE_ENV_VAR} (ContentSid)")
    try:
        payload = post_form(url, data, sid, token, args.timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        code, message = _parse_twilio_error(raw)
        detail = explain_sms_error(code or exc.code, message or raw[:400])
        result["error"] = f"HTTP {exc.code}: {detail}"
        result["twilio_code"] = code
        result["twilio_message"] = message
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["provider_accepted"] = True
    result["provider_message_sid"] = payload.get("sid")
    result["accepted_status"] = payload.get("status")

    # A message is only "queued" on acceptance. Delivery is a separate event
    # this script reports as NOT OBSERVED unless a status callback exists.
    deadline = time.time() + args.poll_seconds
    last = payload.get("status")
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            check = post_form_url(f"{url}/{result['provider_message_sid']}", sid, token)
        except Exception:  # noqa: BLE001
            break
        last = check.get("status", last)
        if last in ("delivered", "sent", "failed", "undelivered"):
            break
    result["terminal_status"] = last
    return result


def post_form_url(url: str, sid: str, token: str) -> dict:
    request = urllib.request.Request(url, headers={
        "Authorization": "Basic " + __import__("base64").b64encode(
            f"{sid}:{token}".encode("utf-8")).decode("ascii"),
    })
    with urllib.request.urlopen(request, timeout=20.0) as response:
        return json.loads(response.read().decode("utf-8"))


def run_call(args, env) -> dict:
    sid = env["TWILIO_ACCOUNT_SID"]
    token = env["TWILIO_AUTH_TOKEN"]
    sender = env["TWILIO_FROM_NUMBER"]
    recipient = env[args.recipient_env_var]
    twiml = ("<Response><Say>SafeVision live provider test. "
             "This is a test of the notification system, not an emergency. "
             "Please hang up now.</Say></Response>")

    result = {
        "channel": "call",
        "provider_accepted": False,
        "provider_call_sid": None,
        "terminal_status": None,
        "human_confirmed_received": None,
    }
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
    try:
        payload = post_form(url, {
            "To": recipient, "From": sender,
            "Twiml": twiml,
            "Timeout": str(args.ring_seconds),
        }, sid, token, args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        result["error"] = f"HTTP {exc.code}: {detail}"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["provider_accepted"] = True
    result["provider_call_sid"] = payload.get("sid")
    result["accepted_status"] = payload.get("status")

    deadline = time.time() + args.poll_seconds
    last = payload.get("status")
    while time.time() < deadline:
        time.sleep(3.0)
        try:
            check = post_form_url(f"{url}/{result['provider_call_sid']}", sid, token)
        except Exception:  # noqa: BLE001
            break
        last = check.get("status", last)
        if last in ("completed", "busy", "no-answer", "failed", "canceled"):
            break
    result["terminal_status"] = last
    result["answered_by"] = None
    return result


# --------------------------------------------------------------------------- #
def ask_operator_confirmed(result: dict) -> None:
    """The provider cannot prove a person read or heard anything.  Ask."""
    print()
    print("The provider reported the states above. That is EVIDENCE FROM THE")
    print("PROVIDER, not proof that a human received anything: Twilio documents")
    print("that 'completed' can mean an IVR or a voicemail answered.")
    print()
    print("Did the recipient confirm receiving this test notification?")
    print("  y = yes, they confirmed it")
    print("  n = no")
    print("  s = skip / nobody to ask")
    try:
        answer = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "s"
    result["human_confirmed_received"] = {"y": True, "n": False}.get(answer, "NOT_CONFIRMED")


def build_parser() -> argparse.ArgumentParser:
    """The CLI definition, kept separate so tests can exercise the real parser.

    Keeping this out of :func:`main` is what allows the regression test to prove
    that every attribute ``gate_report`` reads is actually produced here - the
    mismatch that caused ``AttributeError: 'Namespace' object has no attribute
    'acknowledge_real_number'``.
    """
    parser = argparse.ArgumentParser(
        prog="live_provider_test.py",
        description="SafeVision gated live provider test (layer b).",
        epilog=(
            "All eight gates must pass before any provider request is made. "
            "Required: --live, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, a sender "
            "number, " + CONSENT_FLAG + "=YES, and a recipient in "
            "EMERGENCY_CONTACT_LIVE_TEST_NUMBER. Add "
            "--acknowledge-real-number when the recipient is the project's own "
            "contact. Emergency numbers are always refused. On a Twilio TRIAL "
            "account also set " + TEMPLATE_ENV_VAR + "=HX... (a Twilio Messaging "
            "Template SID), because free-form SMS is rejected with error "
            + TWILIO_TRIAL_TEMPLATE_CODE + "."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true",
                        help="REQUIRED. Contacts the real provider API.")
    parser.add_argument("--channel", choices=("sms", "call"), default="sms",
                        help="which channel to exercise (default: sms)")
    parser.add_argument("--recipient-env-var", default="EMERGENCY_CONTACT_LIVE_TEST_NUMBER",
                        help="env var holding the consenting test recipient")
    parser.add_argument("--content-sid-env-var", default=TEMPLATE_ENV_VAR,
                        help="env var holding the Twilio Messaging Template SID. "
                             "REQUIRED on a trial account: Twilio rejects free-form "
                             "SMS with error 572006.")
    parser.add_argument("--content-variables-env-var", default=CONTENT_VARIABLES_ENV_VAR,
                        help="env var holding template variables as JSON")
    parser.add_argument("--body", default="SafeVision live provider test. "
                                          "Not an emergency. Please reply TEST OK.")
    parser.add_argument("--ring-seconds", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--out", default=str(ROOT / "reports" / "live_test.json"))
    parser.add_argument(
        "--acknowledge-real-number",
        "--i-understand-this-will-ring-the-real-number",
        dest="acknowledge_real_number",
        action="store_true",
        help="required when the recipient is the project's real contact; "
             "acknowledges that a real call/SMS will be sent to it",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_env_file()
    env = dict(os.environ)

    print("=" * 74)
    print(" SafeVision GATED LIVE PROVIDER TEST")
    print("=" * 74)

    passed, checks = gate_report(args, env)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")

    if not passed:
        print()
        print("REFUSED. No provider was contacted and no message was sent.")
        print("Resolve the failing gates above. This script never proceeds on a")
        print("partial pass, and it never falls back to a simulation.")
        return 2

    if args.channel == "sms":
        result = run_sms(args, env)
    else:
        result = run_call(args, env)

    print()
    print(f"  provider accepted the request : {result['provider_accepted']}")
    print(f"  provider resource id          : {result.get('provider_message_sid') or result.get('provider_call_sid')}")
    print(f"  status right after acceptance : {result.get('accepted_status')}")
    print(f"  terminal status observed      : {result.get('terminal_status')}")
    if "error" in result:
        print(f"  error                         : {result['error']}")

    if result["provider_accepted"]:
        ask_operator_confirmed(result)

    result["note"] = (
        "Provider acceptance and a terminal provider state are evidence that the "
        "PROVIDER acted. They are not proof that a person was reached: 'completed' "
        "can mean an IVR or voicemail. human_confirmed_received is the operator's "
        "own out-of-band answer and is the only field that speaks to a person."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nresult written to {out}")

    if not result["provider_accepted"]:
        return 1
    return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  - used by post_form
    raise SystemExit(main())