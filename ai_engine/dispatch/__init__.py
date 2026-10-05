"""Dispatch routing: who gets told, and why.  Phase 5.

    from ai_engine.dispatch import Router, RecipientDirectory, RoutingPolicy

Every routing decision is recorded with the rule that produced it, and routing to
a public emergency service is refused rather than attempted.
"""

from __future__ import annotations

from ai_engine.dispatch.routing import (
    ContactKind,
    DispatchMode,
    DispatchRefusal,
    Recipient,
    RecipientDirectory,
    Router,
    RoutingDecision,
    RoutingPolicy,
    RoutingTrace,
    load_directory,
)

__all__ = [
    "ContactKind",
    "DispatchMode",
    "DispatchRefusal",
    "Recipient",
    "RecipientDirectory",
    "Router",
    "RoutingDecision",
    "RoutingPolicy",
    "RoutingTrace",
    "load_directory",
]