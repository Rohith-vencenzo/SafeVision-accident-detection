"""Repeatable evaluation entry point.

    python scripts\\evaluate.py                              # shipped probe set
    python scripts\\evaluate.py --manifest evaluation/manifest.json
    python scripts\\evaluate.py --manifest my_set.json --out reports/my_eval.json
    python scripts\\evaluate.py --describe-only               # manifest only, no inference

The evaluation never places a call or sends a message: the runner disables the
notification subsystem before building the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from ai_engine.evaluation import (  # noqa: E402
    evaluate_manifest,
    load_manifest,
    render_report,
)
from ai_engine.utils.logging_utils import configure_logging  # noqa: E402

DEFAULT_MANIFEST = ROOT / "evaluation" / "manifest.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision evaluation harness")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="path to the evaluation manifest JSON")
    parser.add_argument("--out", default="", help="write the full report JSON here")
    parser.add_argument("--seed", type=int, default=20261002, help="split seed (reproducibility)")
    parser.add_argument("--describe-only", action="store_true",
                        help="report the manifest's quality blockers and exit (no inference)")
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    manifest = load_manifest(args.manifest)

    print("=" * 78)
    print(" manifest")
    print("=" * 78)
    for key, value in manifest.describe().items():
        print(f"  {key:<28} {value}")

    if manifest.quality_blockers():
        print("\n  THIS SET MAY NOT SUPPORT A QUALITY CLAIM:")
        for blocker in manifest.quality_blockers():
            print(f"    * {blocker}")

    if args.describe_only:
        return 0

    out = args.out or str(ROOT / "reports" / "evaluation_report.json")
    report = evaluate_manifest(args.manifest, seed=args.seed, out_path=out)
    print()
    print(render_report(report))
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())