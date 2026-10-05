"""Ad-hoc diagnostic: print the per-frame component breakdown of a scenario."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ai_engine.config import load_config
from ai_engine.detection.scenarios import SCENARIOS
from ai_engine.detection.scripted import ScriptedDetector
from ai_engine.pipeline import AccidentPipeline


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "head_on_crash"
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0

    cfg = load_config()
    cfg.evidence.enabled = False
    cfg.visualization.enabled = False
    cfg.runtime.write_incidents_file = False

    pipeline = AccidentPipeline(cfg, detector=ScriptedDetector(SCENARIOS[name]), camera_id="X")
    frame = np.zeros((360, 640, 3), np.uint8)

    print(f"{name}: frame score  event col   mot   traj  temp  obj   scene veh sig  state")
    for i in range(max(last + 1, 45)):
        r = pipeline.process_frame(frame, i, i / fps)
        if first <= i <= last:
            c = r.component_scores
            v = r.verification
            print(
                f"{i:11d} {r.accident_score:5.3f} {v.event_score:5.3f} "
                f"{c.get('collision', 0):5.3f} {c.get('motion', 0):5.3f} {c.get('trajectory', 0):5.3f} "
                f"{c.get('temporal', 0):5.3f} {c.get('object_evidence', 0):5.3f} {c.get('scene_evidence', 0):5.3f} "
                f"{r.vehicles_involved:3d} {v.agreement_signals:3d}  {r.state}"
            )
        if r.collision is not None and first <= i <= last:
            sig = r.collision.signals
            print(
                f"          signals={list(sig.active)} closing={sig.closing_speed:.3f} "
                f"decel=({sig.deceleration_a:.2f},{sig.deceleration_b:.2f}) post_stop={sig.post_stop:.2f} "
                f"disp={sig.displacement:.2f} prox={sig.proximity:.2f} iou={sig.overlap:.2f} "
                f"coll_score={r.collision.score:.3f}"
            )


    print()
    print("state transitions:")
    for transition in pipeline.verification.transitions:
        print(f"  {transition.from_state:<12} -> {transition.to_state:<12} "
              f"score={transition.score:.3f} @{transition.timestamp:.2f}s  {transition.reason}")
    print()
    last = pipeline._last_result if hasattr(pipeline, "_last_result") else None
    v = last.verification if last is not None else None
    if v is not None:
        print(f"final: state={v.state} score={v.score:.3f} event={v.event_score:.3f} "
              f"temporal={v.temporal_score:.3f} sufficient={v.evidence_sufficient} "
              f"sec_in_state={v.seconds_in_state:.2f} consec={v.consecutive_above}")
        for reason in v.blocking_reasons:
            print(f"  BLOCKING: {reason}")
        if v.negative_evidence:
            print(f"  negative: {v.negative_evidence}")


if __name__ == "__main__":
    main()
