"""Automatic **demonstration** emergency call (demo mode only).

This package is the only place in SafeVision that can initiate an outbound
phone call, and it does so under strict conditions:

* only when ``DEMO_MODE=true`` (from the environment or ``.env``),
* only after an incident has been **confirmed** by the false-alarm-prevention
  pipeline - never on a suspicious or verifying frame,
* **exactly one call per confirmed incident**,
* only to the recipient in ``EMERGENCY_CONTACT_NUMBER`` (from ``.env``),
* and only through a provider that is actually usable - with no telephony
  credentials the simulation provider is selected, so an accidental run cannot
  ring anybody's phone.

It is a demonstration aid.  It is not connected to any ambulance, police or
112 line, and no emergency service is ever notified.
"""

from __future__ import annotations

from ai_engine.emergency.call_record import CALL_LOG_SCHEMA_VERSION, TRIGGER_AUTOMATIC, DemoCallRecord
from ai_engine.emergency.dispatcher import (
    DEMO_CALL_DISCLAIMER,
    DEMO_SMS_DISCLAIMER,
    DemoCallDispatcher,
)
from ai_engine.emergency.envfile import DOTENV_NAME, env_flag, env_file_path, load_env_file, parse_env_text
from ai_engine.emergency.providers import (
    CallProvider,
    NullProvider,
    NullSmsProvider,
    ProviderResult,
    SimulationProvider,
    SimulationSmsProvider,
    SmsProvider,
    TwilioProvider,
    TwilioSmsProvider,
    build_provider,
    build_sms_provider,
)
from ai_engine.emergency.sms_record import DEMO_SMS_LABEL, DemoSmsRecord, SmsStatus, build_sms_body
from ai_engine.emergency.recipient import (
    RecipientError,
    is_placeholder,
    mask_phone,
    normalize_phone,
)

__all__ = [
    "CALL_LOG_SCHEMA_VERSION",
    "DEMO_CALL_DISCLAIMER",
    "DEMO_SMS_DISCLAIMER",
    "DEMO_SMS_LABEL",
    "DOTENV_NAME",
    "TRIGGER_AUTOMATIC",
    "CallProvider",
    "DemoCallDispatcher",
    "DemoCallRecord",
    "DemoSmsRecord",
    "NullProvider",
    "NullSmsProvider",
    "ProviderResult",
    "RecipientError",
    "SimulationProvider",
    "SimulationSmsProvider",
    "SmsProvider",
    "SmsStatus",
    "TwilioProvider",
    "TwilioSmsProvider",
    "build_provider",
    "build_sms_body",
    "build_sms_provider",
    "env_file_path",
    "env_flag",
    "is_placeholder",
    "load_env_file",
    "mask_phone",
    "normalize_phone",
    "parse_env_text",
]
