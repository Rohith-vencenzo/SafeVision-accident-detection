"""Leakage-safe splitting.

The failure mode this prevents: the same 3 seconds of footage appearing in both
train and test, so a model that memorised the scene scores 100% and looks
fantastic.  Near-identical frames are the norm in video - consecutive frames
differ by a few pixels - so a naive random split of *frames* or even of *clips*
from one recording leaks silently.

The unit of splitting here is the **source group**: every clip sharing a
``source_id`` (same recording session, camera, location or time block) lands on
exactly one side of a split.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ai_engine.evaluation.manifest import ClipLabel, EvalManifest

__all__ = ["Split", "group_split", "assert_no_leakage"]


@dataclass
class Split:
    """One named partition of an evaluation set."""

    name: str
    clips: List[ClipLabel] = field(default_factory=list)

    @property
    def groups(self) -> set:
        return {c.source_id for c in self.clips}

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    @property
    def n_positive(self) -> int:
        return len([c for c in self.clips if c.label == "accident"])

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "clips": self.n_clips,
            "positive_clips": self.n_positive,
            "negative_clips": self.n_clips - self.n_positive,
            "groups": len(self.groups),
            "group_ids": sorted(self.groups),
        }


def _stratify_group(clips: Sequence[ClipLabel]) -> str:
    """A group is positive if it contains any positive clip."""
    return "pos" if any(c.label == "accident" for c in clips) else "neg"


def group_split(
    manifest: EvalManifest,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 20261002,
) -> Dict[str, Split]:
    """Split by ``source_id``, stratified by class.

    Returns ``{"train"/"validation"/"test"}``.  Assignment is deterministic for a
    given ``seed`` so a result can be reproduced exactly - an evaluation you
    cannot reproduce is an anecdote.

    Small sets are handled honestly: if there are not enough *groups* to give
    every partition at least one, the split is still returned but the caller is
    expected to report that it is degenerate (see
    :func:`ai_engine.evaluation.metrics.summarise_split_sufficiency`).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    groups = manifest.groups()
    rng = random.Random(seed)
    names = ("train", "validation", "test")
    splits: Dict[str, Split] = {name: Split(name) for name in names}

    # stratify: shuffle positive and negative group pools independently
    pools: Dict[str, List[str]] = {"pos": [], "neg": []}
    for gid, clips in groups.items():
        pools[_stratify_group(clips)].append(gid)
    for pool in pools.values():
        rng.shuffle(pool)

    remaining = list(ratios)
    for pool_name in ("pos", "neg"):
        pool = pools[pool_name]
        if not pool:
            continue
        # allocate proportionally, giving any rounding remainder to the first
        # partition so every group is used exactly once
        counts = [int(len(pool) * r) for r in ratios]
        counts[0] += len(pool) - sum(counts)
        for name, count in zip(names, counts):
            take = pool[: max(0, count)]
            pool = pool[len(take) :]
            for gid in take:
                splits[name].clips.extend(groups[gid])

    for split in splits.values():
        split.clips.sort(key=lambda c: c.clip)
    return splits


def assert_no_leakage(splits: Dict[str, Split]) -> None:
    """Raise if any ``source_id`` appears in more than one partition."""
    owner: Dict[str, str] = {}
    for name, split in splits.items():
        for gid in split.groups:
            if gid in owner and owner[gid] != name:
                raise AssertionError(
                    f"LEAKAGE: source group {gid!r} appears in both "
                    f"{owner[gid]!r} and {name!r}"
                )
            owner[gid] = name


def group_fingerprint(manifest: EvalManifest) -> str:
    """Stable hash of the group→clip assignment, for run provenance."""
    digest = hashlib.sha256()
    for gid in sorted(manifest.groups()):
        digest.update(gid.encode("utf-8"))
        for clip in sorted(c.clip for c in manifest.groups()[gid]):
            digest.update(clip.encode("utf-8"))
    return digest.hexdigest()[:16]