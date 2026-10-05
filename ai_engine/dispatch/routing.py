"""Recipient directory and dispatch routing.  Phase 5.

Two problems this solves.

**Who gets told.**  A notification system that hard-codes one phone number cannot
grow, cannot be audited and cannot answer "why did this person get that?". Every
routing decision here is made from declared configuration, is written to a
decision log, and names the rule that produced it.

**What may never happen.**  SafeVision has no public-emergency integration, no
jurisdiction and no authorisation. ``EmergencyServiceContact`` therefore exists as
a *representable but unreachable* type: selecting one returns
``NO_APPROVED_RECIPIENT`` rather than a number. Silently substituting a personal
contact for an emergency service — or quietly dialling 112 — is the failure this
package is built to make impossible.

Nothing in this module sends anything. It decides, explains and records.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ai_engine.emergency.recipient import mask_phone, normalize_phone
from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "ContactKind",
    "DispatchMode",
    "DispatchRefusal",
    "Recipient",
    "RecipientDirectory",
    "RoutingDecision",
    "RoutingPolicy",
    "RoutingTrace",
    "load_directory",
]

_LOGGER = get_logger("dispatch")


class ContactKind(str, Enum):
    """What a contact *is*.  The distinction is a safety boundary."""

    #: a consenting person or team who has agreed to receive test/demo alerts
    DESIGNATED_CONTACT = "DESIGNATED_CONTACT"
    #: police / ambulance / fire / 112.  **NOT APPROVED in this project.**
    EMERGENCY_SERVICE = "EMERGENCY_SERVICE"


class DispatchMode(str, Enum):
    """How much of the pipeline is allowed to act."""

    #: nothing leaves the machine; decisions are recorded only
    SIMULATION = "SIMULATION"
    #: real provider calls are permitted to consented contacts
    LIVE = "LIVE"

    @classmethod
    def from_config(cls, value: Any) -> "DispatchMode":
        """Accept the lowercase spelling used in ``config.yaml``.

        Config files read better in lower case, the enum stays consistent with
        the other contract enums (which are upper case), so the conversion is
        explicit rather than accidental.
        """
        text = str(value or "simulation").strip().lower()
        for member in cls:
            if member.value.lower() == text:
                return member
        raise ValueError(
            f"unknown dispatch mode {value!r}; expected 'simulation' or 'live'"
        )


class DispatchRefusal(str, Enum):
    """Why no recipient was selected.  Never a silent empty result."""

    NO_APPROVED_RECIPIENT = "NO_APPROVED_RECIPIENT"
    EMERGENCY_ROUTING_NOT_APPROVED = "EMERGENCY_ROUTING_NOT_APPROVED"
    NO_DIRECTORY = "NO_DIRECTORY"
    ALL_CONTACTS_INELIGIBLE = "ALL_CONTACTS_INELIGIBLE"
    SUPPRESSED_BY_OVERRIDE = "SUPPRESSED_BY_OVERRIDE"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    LOCATION_UNAVAILABLE = "LOCATION_UNAVAILABLE"

    @property
    def explanation(self) -> str:
        return {
            DispatchRefusal.NO_APPROVED_RECIPIENT:
                "no vetted recipient matches this incident's jurisdiction and role",
            DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED:
                "SafeVision has no public-emergency integration, jurisdiction or "
                "authorisation; routing to police/ambulance/112 is NOT APPROVED and "
                "is not implemented",
            DispatchRefusal.NO_DIRECTORY:
                "no recipient directory is configured",
            DispatchRefusal.ALL_CONTACTS_INELIGIBLE:
                "every configured contact failed its eligibility rules",
            DispatchRefusal.SUPPRESSED_BY_OVERRIDE:
                "an operator suppressed notifications for this incident",
            DispatchRefusal.NOT_CONFIRMED:
                "the incident did not reach CONFIRMED_ACCIDENT",
            DispatchRefusal.LOCATION_UNAVAILABLE:
                "no usable position was available and the policy requires one",
        }[self]


@dataclass(frozen=True)
class Recipient:
    """One vetted recipient.

    ``number`` is a raw number and is never logged; ``masked`` is the only form
    that reaches a decision log, an incident payload or a screen.
    """

    recipient_id: str
    kind: ContactKind
    roles: Sequence[str] = ()
    jurisdictions: Sequence[str] = ()
    number_env_var: str = ""
    #: free-text provenance: who vetted this and when
    consented_by: str = ""
    consented_at: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.kind is ContactKind.EMERGENCY_SERVICE and not self.consented_by:
            # not an error - it simply makes the contact ineligible below
            object.__setattr__(self, "enabled", False)

    @property
    def is_emergency_service(self) -> bool:
        return self.kind is ContactKind.EMERGENCY_SERVICE

    def number(self, environ: Optional[Dict[str, str]] = None) -> str:
        env = environ if environ is not None else {}
        raw = env.get(self.number_env_var, "") if self.number_env_var else ""
        return normalize_phone(raw) if raw else ""

    @property
    def masked(self) -> str:
        """A stable, non-reversible label.  Never the digits."""
        digest = hashlib.sha256(self.recipient_id.encode("utf-8")).hexdigest()[:8]
        return f"{self.recipient_id}#{digest}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipient_id": self.recipient_id,
            "kind": str(self.kind.value),
            "roles": list(self.roles),
            "jurisdictions": list(self.jurisdictions),
            "number_env_var": self.number_env_var or None,
            "consented_by": self.consented_by or None,
            "consented_at": self.consented_at or None,
            "enabled": bool(self.enabled),
            "label": self.masked,
        }


# --------------------------------------------------------------------------- #
@dataclass
class RoutingDecision:
    """The outcome, always with the rule that produced it."""

    incident_id: str
    recipient: Optional[Recipient]
    refused: Optional[DispatchRefusal] = None
    rule: str = ""
    reason: str = ""
    evaluated: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = str(DispatchMode.SIMULATION.value)
    decided_at: float = 0.0
    #: True when the recipient exists but nothing may be sent in this mode
    would_dispatch: bool = False

    @property
    def is_dispatchable(self) -> bool:
        return self.recipient is not None and self.refused is None and self.would_dispatch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "mode": self.mode,
            "rule": self.rule or None,
            "reason": self.reason,
            "recipient_id": self.recipient.recipient_id if self.recipient else None,
            "recipient_label": self.recipient.masked if self.recipient else None,
            "refused": str(self.refused.value) if self.refused else None,
            "refusal_explanation": (
                self.refused.explanation if self.refused else None
            ),
            "would_dispatch": bool(self.would_dispatch),
            "evaluated": list(self.evaluated),
            "decided_at_unix": self.decided_at or None,
        }


@dataclass
class RoutingTrace:
    """Append-only decision log.  One entry per routing attempt."""

    entries: List[RoutingDecision] = field(default_factory=list)
    #: operator suppressions: incident_id -> reason
    suppressed: Dict[str, str] = field(default_factory=dict)

    def record(self, decision: RoutingDecision) -> RoutingDecision:
        self.entries.append(decision)
        return decision

    def suppress(self, incident_id: str, reason: str = "operator override") -> None:
        """Human-in-the-loop kill switch for one incident."""
        self.suppressed[incident_id] = reason
        _LOGGER.warning("notifications SUPPRESSED for incident %s: %s", incident_id, reason)

    def unsuppress(self, incident_id: str) -> None:
        self.suppressed.pop(incident_id, None)

    def is_suppressed(self, incident_id: str) -> bool:
        return incident_id in self.suppressed

    def for_incident(self, incident_id: str) -> List[RoutingDecision]:
        return [e for e in self.entries if e.incident_id == incident_id]

    def summary(self) -> Dict[str, Any]:
        dispatched = [e for e in self.entries if e.is_dispatchable]
        refused = [e for e in self.entries if e.refused is not None]
        counts: Dict[str, int] = {}
        for entry in refused:
            key = str(entry.refused.value)
            counts[key] = counts.get(key, 0) + 1
        return {
            "decisions": len(self.entries),
            "dispatchable": len(dispatched),
            "refused": len(refused),
            "refusals_by_reason": counts,
            "suppressed_incidents": list(self.suppressed),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"summary": self.summary(), "entries": [e.to_dict() for e in self.entries]},
            indent=indent,
        )


# --------------------------------------------------------------------------- #
class RecipientDirectory:
    """The vetted set of people who may be told.

    A missing or unreadable directory is **not** an error that silently passes:
    :meth:`resolve` returns ``NO_DIRECTORY`` so the failure is visible in the
    decision log rather than inferred from an absence.
    """

    def __init__(self, recipients: Optional[Sequence[Recipient]] = None) -> None:
        self._recipients: List[Recipient] = list(recipients or [])

    def __len__(self) -> int:
        return len(self._recipients)

    def __iter__(self):
        return iter(self._recipients)

    def add(self, recipient: Recipient) -> "RecipientDirectory":
        self._recipients.append(recipient)
        return self

    def get(self, recipient_id: str) -> Optional[Recipient]:
        for recipient in self._recipients:
            if recipient.recipient_id == recipient_id:
                return recipient
        return None

    def eligible(
        self,
        role: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> List[Recipient]:
        """Contacts that could legally and practically receive this alert."""
        out: List[Recipient] = []
        for recipient in self._recipients:
            if not recipient.enabled:
                continue
            if role and role not in recipient.roles:
                continue
            if jurisdiction and jurisdiction not in recipient.jurisdictions:
                continue
            out.append(recipient)
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> "RecipientDirectory":
        p = Path(path)
        if not p.is_file():
            _LOGGER.error("recipient directory not found: %s", p)
            return cls([])
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries = raw.get("recipients", raw if isinstance(raw, list) else [])
        recipients: List[Recipient] = []
        for index, entry in enumerate(entries):
            try:
                recipients.append(Recipient(
                    recipient_id=str(entry["recipient_id"]),
                    kind=ContactKind(entry.get("kind", "DESIGNATED_CONTACT")),
                    roles=tuple(entry.get("roles", ())),
                    jurisdictions=tuple(entry.get("jurisdictions", ())),
                    number_env_var=str(entry.get("number_env_var", "")),
                    consented_by=str(entry.get("consented_by", "")),
                    consented_at=str(entry.get("consented_at", "")),
                    enabled=bool(entry.get("enabled", True)),
                ))
            except (KeyError, ValueError) as exc:
                _LOGGER.error("recipient entry %d is invalid and was skipped: %s", index, exc)
        return cls(recipients)


def load_directory(path: str | Path) -> RecipientDirectory:
    return RecipientDirectory.from_file(path)


# --------------------------------------------------------------------------- #
@dataclass
class RoutingPolicy:
    """Declared routing rules.  No routing decision is hard-coded elsewhere."""

    mode: DispatchMode = DispatchMode.SIMULATION
    #: role the chosen contact must hold, e.g. "safety-officer"
    required_role: str = ""
    #: jurisdiction the contact must be authorised in
    required_jurisdiction: str = ""
    #: refuse when no position is available (default False: a camera position
    #: is usually enough to name a junction)
    require_location: bool = False
    #: allow emergency-service contacts.  Always False in this project.
    allow_emergency_service: bool = False
    #: refuse everything - the master kill switch
    dispatch_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "required_role": self.required_role or None,
            "required_jurisdiction": self.required_jurisdiction or None,
            "require_location": self.require_location,
            "allow_emergency_service": self.allow_emergency_service,
            "dispatch_enabled": self.dispatch_enabled,
            "emergency_service_routing": "NOT_APPROVED",
        }


class Router:
    """Applies :class:`RoutingPolicy` to an incident and records why.

    Fail-safe ordering, most decisive check first, because the point of this
    class is that the *first* thing that refuses is the thing that would have
    been dangerous:

    1. master kill switch
    2. operator suppression for this incident
    3. incident not ``CONFIRMED_ACCIDENT``
    4. mode
    5. emergency-service routing attempt
    6. directory
    7. eligibility (role, jurisdiction, consent)
    8. location requirement
    """

    def __init__(
        self,
        policy: Optional[RoutingPolicy] = None,
        directory: Optional[RecipientDirectory] = None,
        trace: Optional[RoutingTrace] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or RoutingPolicy()
        self.directory = directory if directory is not None else RecipientDirectory([])
        self.trace = trace if trace is not None else RoutingTrace()
        self.clock = clock

    def route(
        self,
        incident: Dict[str, Any],
        environ: Optional[Dict[str, str]] = None,
    ) -> RoutingDecision:
        incident_id = str(incident.get("incident_id") or "")
        evaluated: List[Dict[str, Any]] = []
        policy = self.policy

        def decide(
            rule: str,
            recipient: Optional[Recipient] = None,
            refused: Optional[DispatchRefusal] = None,
            reason: str = "",
        ) -> RoutingDecision:
            decision = RoutingDecision(
                incident_id=incident_id,
                recipient=recipient,
                refused=refused,
                rule=rule,
                reason=reason or (refused.explanation if refused else ""),
                evaluated=evaluated,
                mode=str(policy.mode.value),
                decided_at=self.clock(),
                # Simulation records a decision and stops; it never dispatches.
                would_dispatch=(
                    recipient is not None
                    and refused is None
                    and policy.mode is DispatchMode.LIVE
                ),
            )
            self.trace.record(decision)
            return decision

        # 1. master kill switch
        if not policy.dispatch_enabled:
            evaluated.append({"rule": "dispatch_enabled", "passed": False})
            return decide("dispatch_enabled", refused=DispatchRefusal.SUPPRESSED_BY_OVERRIDE,
                          reason="routing.dispatch_enabled is false - the master kill switch is on")
        evaluated.append({"rule": "dispatch_enabled", "passed": True})

        # 2. operator suppression
        if self.trace.is_suppressed(incident_id):
            evaluated.append({"rule": "operator_suppression", "passed": False})
            return decide("operator_suppression",
                          refused=DispatchRefusal.SUPPRESSED_BY_OVERRIDE,
                          reason=self.trace.suppressed[incident_id])
        evaluated.append({"rule": "operator_suppression", "passed": True})

        # 3. only a confirmed incident is ever routed
        status = str(incident.get("status") or "")
        if status != "CONFIRMED_ACCIDENT":
            evaluated.append({"rule": "incident_confirmed", "passed": False,
                              "status": status})
            return decide("incident_confirmed", refused=DispatchRefusal.NOT_CONFIRMED,
                          reason=f"incident status is {status!r}, not CONFIRMED_ACCIDENT")
        evaluated.append({"rule": "incident_confirmed", "passed": True, "status": status})

        # 4. a directory must exist in ANY mode - a missing directory is a
        #    configuration error, and reporting it here means it shows up in the
        #    decision log even during a dry run
        if len(self.directory) == 0:
            evaluated.append({"rule": "directory_present", "passed": False})
            return decide("directory_present", refused=DispatchRefusal.NO_DIRECTORY,
                          reason="no recipient directory is configured")
        evaluated.append({"rule": "directory_present", "passed": True,
                          "contacts": len(self.directory)})

        # 5. mode
        if policy.mode is DispatchMode.SIMULATION:
            evaluated.append({"rule": "mode", "passed": True, "mode": "SIMULATION"})
            recipient = self._first_eligible(evaluated)
            if recipient is None:
                return decide("mode",
                              refused=DispatchRefusal.ALL_CONTACTS_INELIGIBLE,
                              reason="simulation mode: no eligible recipient to record a decision for")
            return decide("mode", recipient=recipient, reason="simulation mode; decision only")
        evaluated.append({"rule": "mode", "passed": True, "mode": "LIVE"})

        # 6. an emergency-service request is refused, always
        if incident.get("request_emergency_service"):
            evaluated.append({"rule": "emergency_service_routing", "passed": False})
            return decide("emergency_service_routing",
                          refused=DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED,
                          reason=DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED.explanation)

        # 7. eligibility
        recipients = self.directory.eligible(policy.required_role or None,
                                             policy.required_jurisdiction or None)
        considered = [
            {"recipient_id": r.recipient_id, "label": r.masked,
             "kind": str(r.kind.value), "eligible": r in recipients}
            for r in self.directory
        ]
        evaluated.append({"rule": "eligibility", "passed": bool(recipients),
                          "considered": considered})
        if not recipients:
            return decide("eligibility", refused=DispatchRefusal.NO_APPROVED_RECIPIENT,
                          reason=DispatchRefusal.NO_APPROVED_RECIPIENT.explanation)

        # an emergency-service contact can never be chosen, even if it is listed
        # and even if it somehow claims consent
        services = [r for r in recipients if r.is_emergency_service]
        if services and not policy.allow_emergency_service:
            _LOGGER.error(
                "recipient directory lists emergency-service contact(s) %s; SafeVision "
                "cannot route to them and will refuse to select one",
                [r.recipient_id for r in services],
            )
        designated = [r for r in recipients if not r.is_emergency_service]
        evaluated.append({"rule": "emergency_service_excluded", "passed": True,
                          "excluded": [r.recipient_id for r in services]})
        if not designated:
            return decide("emergency_service_excluded",
                          refused=DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED,
                          reason=DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED.explanation)

        # 8. location, if the policy demands it
        if policy.require_location:
            provenance = ((incident.get("extra") or {}).get("location_provenance")
                          or incident.get("location_provenance") or {})
            source = str(provenance.get("source") or "")
            has_position = provenance.get("latitude") is not None
            if not has_position:
                evaluated.append({"rule": "location_required", "passed": False,
                                  "source": source or None})
                return decide("location_required",
                              refused=DispatchRefusal.LOCATION_UNAVAILABLE,
                              reason="policy requires a position and none is available")
            evaluated.append({"rule": "location_required", "passed": True, "source": source})

        return decide("eligible_contact", recipient=designated[0],
                      reason=f"first eligible designated contact "
                             f"{designated[0].recipient_id} of {len(designated)}")

    def _first_eligible(self, evaluated: List[Dict[str, Any]]) -> Optional[Recipient]:
        eligible = self.directory.eligible(self.policy.required_role or None,
                                           self.policy.required_jurisdiction or None)
        designated = [r for r in eligible if not r.is_emergency_service]
        return designated[0] if designated else None