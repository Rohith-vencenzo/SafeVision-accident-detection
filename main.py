#!/usr/bin/env python
"""SafeVision - command line entry point.

Examples
--------
    # local video
    python main.py --source test_videos/accident.mp4

    # laptop webcam
    python main.py --source 0 --camera-id CAM-001

    # CCTV / IP camera (credentials come from the environment, never the URL)
    setx SAFEVISION_RTSP_USERNAME admin
    setx SAFEVISION_RTSP_PASSWORD <secret>
    python main.py --source "rtsp://192.168.1.64:554/stream" --camera-id CAM-002

    # scripted presentation
    python main.py --demo

    # the automatic DEMONSTRATION emergency call (needs DEMO_MODE=true in .env)
    python main.py --source clip.mp4 --demo-call     # force it on for one run
    python main.py --source clip.mp4 --no-demo-call   # force it off for one run
    python main.py --check-emergency                  # what would happen, and why

    # inspect what is running
    python main.py --print-config
    python main.py --list-cameras
    python main.py --print-schema
    python main.py --check-source test_videos/accident.mp4

    # tune a threshold without editing the file
    python main.py --source clip.mp4 --set verification.confirm_score=0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

# make "python main.py" work from anywhere inside the project
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_engine.config import ConfigError, describe_config, load_config, redact_url  # noqa: E402
from ai_engine.detection.yolo_detector import ModelNotAvailableError, YoloDetector  # noqa: E402
from ai_engine.emergency.dispatcher import DemoCallDispatcher  # noqa: E402
from ai_engine.emergency.envfile import env_file_path, load_env_file  # noqa: E402
from ai_engine.emergency.recipient import mask_phone  # noqa: E402
from ai_engine.pipeline.accident_pipeline import AccidentPipeline  # noqa: E402
from ai_engine.pipeline.demo import DemoRunner  # noqa: E402
from ai_engine.schemas.incident_result import (  # noqa: E402
    DISCLAIMER,
    INCIDENT_JSON_SCHEMA,
)
from ai_engine.utils.jsonio import dumps  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402
from ai_engine.video.source import VideoReader, VideoSourceError  # noqa: E402
from ai_engine.visualization.workflow import WorkflowReporter  # noqa: E402

_LOGGER = get_logger("main")


def _force_utf8_console() -> None:
    """Let the console print non-ASCII (the SMS body contains \U0001F6A8).

    Windows terminals default to cp1252, where that character raises
    UnicodeEncodeError and would take the whole run down while merely
    *printing* a result.  JSON output stays ASCII-escaped either way.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not a TTY, or already UTF-8
            pass


_force_utf8_console()

EXIT_OK = 0
EXIT_INCIDENT = 0          # an incident was found - not an error
EXIT_USAGE = 2
EXIT_ERROR = 1

#: the built-in default of ``video.source``; used to detect "nobody chose a source"
DEFAULT_SOURCE = "0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SafeVision - AI-based real-time accident detection and false-alarm prevention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "This tool only READS video and WRITES a JSON incident report.\n"
            "The one outbound action is an opt-in DEMONSTRATION phone call, placed\n"
            "once per CONFIRMED incident and only when DEMO_MODE=true. It is not\n"
            "connected to any ambulance, police or 112 line."
        ),
    )
    parser.add_argument("--source", default=None, help="video file | webcam index (0) | rtsp://... ")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--camera-id", default=None, help="registered camera id to attach")
    parser.add_argument("--model", default=None, help="YOLO weights (e.g. yolo11n.pt, models/yolo11s.pt)")
    parser.add_argument("--device", default=None, help="cpu | 0 | auto")
    parser.add_argument("--conf", type=float, default=None, help="detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="inference size")
    parser.add_argument("--width", type=int, default=None, help="downscale frames to this width")
    parser.add_argument("--fps", type=float, default=None, help="cap the analysis FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--show", action="store_true", help="open a preview window (needs a GUI)")
    parser.add_argument("--save-output", action="store_true", help="write an annotated debug video")
    parser.add_argument("--output", default=None, help="path for the annotated debug video")
    parser.add_argument("--json", action="store_true", help="print the incident JSON to stdout")
    parser.add_argument("--no-evidence", action="store_true", help="do not write evidence files")
    parser.add_argument("--no-annotation", action="store_true", help="skip drawing the debug overlay")
    parser.add_argument("--demo", action="store_true", help="run the scripted presentation mode")
    parser.add_argument(
        "--demo-call",
        action="store_true",
        help="force DEMO_MODE on for this run: one demonstration call per CONFIRMED incident",
    )
    parser.add_argument(
        "--no-demo-call",
        action="store_true",
        help="force DEMO_MODE off for this run, even when DEMO_MODE=true in .env",
    )
    parser.add_argument(
        "--check-emergency",
        action="store_true",
        help="report the demonstration-call configuration (recipient masked) and exit",
    )
    parser.add_argument("--print-config", action="store_true", help="print the effective config and exit")
    parser.add_argument("--print-schema", action="store_true", help="print the incident JSON schema and exit")
    parser.add_argument("--list-cameras", action="store_true", help="list registered cameras and exit")
    parser.add_argument("--check-source", default=None, help="validate a video source and exit")
    parser.add_argument("--check-model", action="store_true", help="load the detector once and exit")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set collision.min_signals=3",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo_call and args.no_demo_call:
        parser.error("--demo-call and --no-demo-call cannot be used together")

    # ---- .env ----------------------------------------------------------
    # Loaded before the config so DEMO_MODE (and the telephony credentials) are
    # visible to it.  Real environment variables always win over the file.
    dotenv = env_file_path()
    if dotenv is not None:
        load_env_file(dotenv)

    # ---- configuration ------------------------------------------------
    try:
        config = load_config(path=args.config, overrides=args.overrides)
    except ConfigError as exc:
        parser.error(f"configuration error: {exc}")  # exits with code 2
        return EXIT_USAGE

    # CLI flags win over .env and config.yaml for the demonstration call.
    # Both the config object AND the environment are set: the dispatcher reads
    # DEMO_MODE from the environment on every access, so setting only the config
    # left `--demo-call` silently ineffective whenever `.env` said false - which
    # is exactly the "no call was made" symptom. The flag must beat the file.
    if args.demo_call:
        config.emergency.demo_mode = True
        os.environ["DEMO_MODE"] = "true"
    elif args.no_demo_call:
        config.emergency.demo_mode = False
        os.environ["DEMO_MODE"] = "false"

    # CLI flags win over the config file
    if args.device:
        config.model.device = args.device
    if args.model:
        config.model.weights = args.model
    if args.conf is not None:
        if not 0.0 < args.conf <= 1.0:
            parser.error("--conf must be in (0, 1]")
        config.model.conf = args.conf
    if args.imgsz:
        config.model.imgsz = args.imgsz
    if args.width:
        config.video.target_width = args.width
    if args.fps:
        config.video.process_fps = args.fps
    if args.log_level:
        config.runtime.log_level = args.log_level
    if args.max_frames is not None:
        config.runtime.max_frames = args.max_frames
    if args.source is not None:
        config.video.source = args.source
    if args.camera_id:
        config.location.default_camera_id = args.camera_id
    if args.no_evidence:
        config.evidence.enabled = False
    if args.no_annotation:
        config.visualization.enabled = False
    if args.save_output:
        config.runtime.save_output_video = True
    if args.output:
        config.runtime.output_video = args.output
    if args.json:
        config.runtime.print_incident_json = True

    try:
        config.validate()
    except ConfigError as exc:
        parser.error(f"configuration error: {exc}")
        return EXIT_USAGE

    configure_logging(config.runtime.log_level, config.runtime.log_file)
    # Ultralytics/Torch emit a "'half' is deprecated" DeprecationWarning on
    # nearly every inference. It arrives through the `warnings` machinery rather
    # than logging, so a logger level cannot suppress it - and it interleaves
    # with the live progress line and floods a redirected log. Filter it; any
    # genuine fault still surfaces as an exception or an error-level log.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*ultralytics.*")
    warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
    _DropNoisyDeprecations().install()
    _LOGGER.info("SafeVision | config: %s", config.source_file)
    _LOGGER.info("%s", DISCLAIMER)

    # ---- informational modes ------------------------------------------
    if args.print_config:
        print(dumps(describe_config(config)))
        return EXIT_OK
    if args.print_schema:
        print(dumps(INCIDENT_JSON_SCHEMA))
        return EXIT_OK
    if args.list_cameras:
        return _list_cameras(config)
    if args.check_source is not None:
        return _check_source(args.check_source, config)
    if args.check_model:
        return _check_model(config)
    if args.check_emergency:
        return _check_emergency(config)

    # ---- demo mode ----------------------------------------------------
    if args.demo:
        runner = DemoRunner(config, camera_id=config.location.default_camera_id)
        runner.run(max_frames=config.runtime.max_frames)
        path = runner.write_report()
        if path:
            print(f"\ndemo report written to {path}")
        return EXIT_OK

    # ---- normal run ----------------------------------------------------
    # A bare `python main.py` must not silently open the webcam: the source has
    # to be chosen on the command line or in config.yaml (video.source).
    source_chosen = args.source is not None or str(config.video.source) != DEFAULT_SOURCE
    source = config.video.source
    if source is None or (not source_chosen and not args.demo):
        parser.error(
            "no video source: pass --source FILE | --source 0 (webcam) | --source rtsp://...\n"
            "       or set video.source in config.yaml, or run the demo with --demo"
        )
        return EXIT_USAGE

    _LOGGER.info("source: %s", redact_url(source))
    # `--json` streams one JSON object per incident on stdout, so the whole
    # stdout can be piped straight into a consumer (`| jq`, a queue reader...).
    streamed_json = bool(config.runtime.print_incident_json)
    # The workflow reporter renders the whole pipeline on stdout, so it must be
    # silent in --json mode where stdout is a machine-readable stream.
    reporter = WorkflowReporter(quiet=streamed_json)
    call_status_printer = None if streamed_json else reporter.on_call_status

    def _incident_sink(payload: Dict[str, Any]) -> None:
        """Stream the JSON, or render the workflow stage, depending on the mode."""
        if streamed_json:
            _print_incident(payload)
        else:
            reporter.on_incident(payload)

    try:
        pipeline = AccidentPipeline(
            config,
            camera_id=config.location.default_camera_id,
            source_label=redact_url(source),
            on_incident=_incident_sink,
            on_call_status=call_status_printer,
            on_transition=reporter.on_transition,
        )
    except Exception as exc:  # noqa: BLE001 - configuration / model problems
        _LOGGER.error("could not start the pipeline: %s", exc)
        return EXIT_ERROR

    # Say what is about to happen with the phone, before it happens. A simulated
    # "CALL COMPLETED" read exactly like a delivered one, which is what made a
    # working demo look like a broken notification path.
    emergency_on = pipeline.emergency is not None and pipeline.emergency.demo_mode
    if not streamed_json:
        if emergency_on:
            reporter.simulated_banner()
        else:
            # Real mode. The pipeline deliberately builds no dispatcher when demo
            # is off and no credentials exist, so read the credential state
            # directly - otherwise this would print nothing at all, which is
            # exactly the "silent, nothing happened" case we are fixing.
            reporter.banner("REAL NOTIFICATION MODE")
            emergency = config.emergency
            provider = emergency.provider or "auto"
            required = [emergency.twilio_sid_env_var, emergency.twilio_token_env_var,
                        emergency.twilio_from_env_var]
            missing = [name for name in required if not os.environ.get(name, "").strip()]
            recipient = os.environ.get(emergency.contact_env_var, "").strip()
            reporter.kv("provider", provider)
            if provider == "simulation":
                reporter.kv("provider usable", True)
                reporter.kv("WARNING", "provider is 'simulation' - still no real call will be placed")
                reporter.kv("action", "NOTHING WILL BE SENT")
            elif missing:
                reporter.kv("provider usable", False)
                reporter.kv("NOT_CONFIGURED",
                            f"missing in .env: {', '.join(missing)}")
                reporter.kv("recipient", mask_phone(recipient) if recipient
                            else f"({emergency.contact_env_var} not set)")
                reporter.kv("action", "NOTHING WILL BE SENT - no provider is configured")
            else:
                reporter.kv("provider usable", True)
                reporter.kv("recipient", mask_phone(recipient) if recipient
                            else "(unconfigured)")
                reporter.kv("action", "a REAL call then a REAL SMS per confirmed incident")
            reporter._write()
            reporter._write("   To enable a real call/SMS: set DEMO_MODE=false and fill in")
            reporter._write(f"   {', '.join(required)} in .env.")

    display = _PreviewWindow() if args.show else None
    run = None
    # `--show` on a headless OpenCV build would produce nothing at all, so fall
    # back to writing an annotated video. Visuals either way.
    visuals_written = bool(config.runtime.save_output_video)
    if display is not None and not display.available and not config.runtime.save_output_video:
        _LOGGER.warning(
            "--show was requested but this OpenCV build cannot open a window; "
            "writing an annotated video instead"
        )
        config.runtime.save_output_video = True
        visuals_written = True
        display = None
    if args.show and not args.save_output and not config.runtime.save_output_video:
        visuals_written = False
    # a terminal run must never look like a hung process
    progress = _ConsoleProgress(enabled=not args.json and not streamed_json,
                                reporter=reporter)
    on_frame = display.push if display is not None else progress.push
    started = time.perf_counter()
    exit_code = EXIT_OK
    try:
        run = pipeline.process_video(source, on_frame=on_frame)
        elapsed = time.perf_counter() - started

        # ---- wait for the notification before reporting it -------------
        # The call and the SMS run on a worker thread, and the incident payload
        # is a snapshot taken when the incident was built. Reporting before the
        # worker finishes shows INITIATED / PENDING even though the provider has
        # already dialled - which looks exactly like "no call was made". Block
        # first, then report what actually happened.
        unresolved = 0
        if run.incidents and pipeline.emergency is not None:
            unresolved = pipeline.wait_for_notifications()
            if unresolved:
                _LOGGER.warning(
                    "%d notification(s) had not finished when the video ended; "
                    "the states below may still change", unresolved,
                )
        _print_run_summary(run, elapsed, pipeline.emergency)
        if not streamed_json:
            # show the exact alert text, and the ordered workflow with each
            # stage marked reached / refused
            if pipeline.emergency is not None:
                for states in pipeline.emergency.final_states().values():
                    if states.get("sms_body"):
                        reporter.show_sms(states["sms_body"])
            reporter.on_report_saved(
                getattr(run, "incidents_file", None),
                str(pipeline.emergency.log_path) if pipeline.emergency is not None else None,
            )
            reporter.summary()
        if config.runtime.print_incident_json and run.incidents and not streamed_json:
            # No callback was attached (e.g. a caller replaced it), so emit the
            # incidents here.  When the callback already streamed them, stdout
            # stays a clean one-JSON-object-per-incident stream.
            for incident in run.incidents:
                print(dumps(incident), flush=True)
    except ModelNotAvailableError as exc:
        _LOGGER.error("%s", exc)
        return EXIT_ERROR
    except VideoSourceError as exc:
        _LOGGER.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        _LOGGER.info("stopped by the user (Ctrl+C)")
    finally:
        if display is not None:
            display.close()
        progress.close()
        pipeline.close()

    # ---- say plainly how to SEE it, if nothing visual was produced -------- #
    if not visuals_written and display is None and run is not None and run.frames_processed:
        print()
        print("no visuals were rendered for this run. To see the detection overlay:")
        print(f"    preview window     : python main.py --source {redact_url(source)} --show")
        print(f"    annotated video    : python main.py --source {redact_url(source)} "
              f"--save-output --output {config.runtime.output_video}")

    return exit_code


# --------------------------------------------------------------------------- #
def _print_demo_call_status(status: Dict[str, Any]) -> None:
    """Console line for each transition of the demonstration call.

    Suppressed automatically when ``--json`` is active, so stdout stays a clean
    one-JSON-object-per-incident stream.
    """
    headline = f"{status.get('label', 'DEMO EMERGENCY CALL')} - {status.get('status')}"
    recipient = status.get("recipient") or "(unconfigured)"
    detail = status.get("reason") or status.get("provider") or ""
    print(f"{headline} | {recipient} | {detail}", flush=True)


# --------------------------------------------------------------------------- #
def _print_run_summary(
    run: Any,
    elapsed: float,
    dispatcher: Any = None,
) -> None:
    _LOGGER.info(
        "processed %d frame(s) in %.1fs | incidents: %d | false alarms: %d",
        run.frames_processed,
        elapsed,
        len(run.incidents),
        run.false_alarms,
    )
    for transition in run.transitions:
        _LOGGER.info(
            "  state %s -> %s (%.3f) : %s",
            transition["from"],
            transition["to"],
            transition["score"],
            transition["reason"],
        )
    for incident in run.incidents:
        _LOGGER.info(
            "  %s | %s | score %.2f | severity %s | vehicles %d | evidence %s",
            incident["incident_id"],
            incident["camera"].get("camera_id") or "unregistered camera",
            incident["accident"]["score"],
            incident["accident"]["severity"],
            incident["objects"]["vehicles_involved"],
            incident["media"].get("evidence_dir") or "(disabled)",
        )
        # The incident payload is a SNAPSHOT taken when the incident was built,
        # before the provider was contacted. Reporting it as the outcome is what
        # made a completed call look like it never happened, so the live record
        # wins whenever the dispatcher is available.
        incident_id = incident.get("incident_id")
        live = (dispatcher.final_states().get(incident_id)
                if dispatcher is not None else None)

        if live is not None:
            _LOGGER.info(
                "    EMERGENCY CALL : %s -> %s%s | provider=%s%s%s",
                live["call"],
                live["recipient"] or "(unconfigured)",
                f" ({live['reason']})" if live["reason"] else "",
                live["provider"],
                f" | provider status: {live['provider_status']}" if live["provider_status"] else "",
                f" | ref: {live['provider_ref']}" if live["provider_ref"] else "",
            )
            if live["error"]:
                _LOGGER.error("    CALL ERROR    : %s", live["error"])
            _LOGGER.info(
                "    EMERGENCY SMS  : %s -> %s%s | provider=%s%s%s",
                live["sms"],
                live["recipient"] or "(unconfigured)",
                f" ({live['sms_reason']})" if live["sms_reason"] else "",
                live["sms_provider"],
                f" | provider status: {live['sms_provider_status']}"
                if live["sms_provider_status"] else "",
                f" | ref: {live['sms_provider_ref']}" if live["sms_provider_ref"] else "",
            )
            if live["sms_error"]:
                _LOGGER.error("    SMS ERROR     : %s", live["sms_error"])
            if live["sms_body"]:
                _LOGGER.info("    --- SMS that was sent (verbatim) ---")
                for text_line in str(live["sms_body"]).splitlines():
                    _LOGGER.info("    | %s", text_line)
                _LOGGER.info("    --- end of SMS ---")
            if live["correlation_id"]:
                _LOGGER.info("    correlation id : %s", live["correlation_id"])
            continue

        # no dispatcher (notifications disabled): keep the old snapshot lines
        call = incident.get("demo_emergency_call") or {}
        if call.get("demo_mode") or call.get("attempted"):
            _LOGGER.info(
                "    demo emergency call: %s -> %s%s",
                call.get("status"),
                call.get("recipient") or "(unconfigured)",
                f" ({call['reason']})" if call.get("reason") else "",
            )
        sms = incident.get("demo_emergency_sms") or {}
        if sms.get("demo_mode") or sms.get("attempted"):
            _LOGGER.info(
                "    demo emergency sms : %s -> %s%s",
                sms.get("status"),
                sms.get("recipient") or "(unconfigured)",
                f" ({sms['reason']})" if sms.get("reason") else "",
            )
    if run.incidents_file:
        _LOGGER.info("incidents written to %s", run.incidents_file)


def _print_incident(incident: dict) -> None:
    """``--json`` output: one JSON object per incident, on stdout, unbuffered.

    Nothing else is ever written to stdout, so ``main.py --json`` can be piped
    directly into a consumer.  Logs go to stderr.
    """
    print(dumps(incident), flush=True)


def _list_cameras(config: Any) -> int:
    from ai_engine.location import LocationManager

    manager = LocationManager.from_config(config.location)
    if not len(manager):
        print(f"no cameras registered ({config.location.cameras_file} not found or empty)")
        print("create cameras.yaml next to config.yaml, or register cameras at runtime")
        return EXIT_OK
    print(f"{'camera_id':<16}{'name':<28}{'lat':>11}{'lon':>12}  location")
    print("-" * 88)
    for camera_id, camera in sorted(manager.cameras.items()):
        lat = f"{camera.latitude:.6f}" if camera.latitude is not None else "-"
        lon = f"{camera.longitude:.6f}" if camera.longitude is not None else "-"
        print(f"{camera_id:<16}{camera.name[:27]:<28}{lat:>11}{lon:>12}  {camera.location_name or '-'}")
    if manager.default_camera_id:
        print(f"\ndefault camera: {manager.default_camera_id}")
    return EXIT_OK


def _check_source(source: str, config: Any) -> int:
    try:
        reader = VideoReader(source, config.video)
        reader.open()
    except VideoSourceError as exc:
        print(f"SOURCE NOT USABLE: {exc}")
        return EXIT_ERROR
    print("source is usable:")
    print(dumps(reader.stats()))
    reader.close()
    return EXIT_OK


def _gui_available() -> bool:
    """Can this OpenCV build actually open a window?

    Probed once, cheaply: a headless/server build of OpenCV raises the moment
    ``imshow`` is called, which would otherwise happen once per frame.
    """
    try:
        import cv2
        import numpy as np

        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow("safevision-gui-probe", probe)
        cv2.destroyWindow("safevision-gui-probe")
        return True
    except Exception:  # noqa: BLE001 - no GUI backend, or no display
        return False


def _check_emergency(config: Any) -> int:
    """Report the demonstration-call configuration without placing a call.

    Prints the recipient masked - this is a pre-demo sanity check, and it must
    never be the thing that leaks the number to a shared terminal or a
    screenshot of the console.
    """
    dispatcher = DemoCallDispatcher(config.emergency)
    try:
        info = dispatcher.describe()
        raw = dispatcher.raw_recipient()
        info["recipient_masked"] = mask_phone(raw)
        info[".env"] = str(env_file_path() or "(none found)")
        print("DEMO EMERGENCY CALL - configuration check")
        print("-" * 62)
        print(f"  DEMO_MODE          : {info['demo_mode']}")
        print(f"  feature enabled    : {info['enabled']}")
        print(f"  provider           : {info['provider']['provider']}")
        print(f"  provider usable    : {info['provider']['usable']}")
        if info.get("provider_problem"):
            print(f"    -> {info['provider_problem']}")
        print(f"  recipient env var  : {info['recipient_env_var']}")
        print(f"  recipient          : {info['recipient_masked']} (masked)")
        if info.get("recipient_problem"):
            print(f"    -> {info['recipient_problem']}")
        print(f"  real call possible : {info['will_place_a_real_call']}")
        if info["demo_mode"] and info["recipient_configured"] and info["recipient_valid"]:
            print(
                f"  action             : ONE automatic call per CONFIRMED incident "
                f"to {info['recipient_masked']}"
            )
        elif not info["demo_mode"]:
            print("  action             : none (demo mode is off)")
        else:
            print("  action             : none (a safety interlock will block the call)")
        print(f"  call log           : {info['call_log']}")
        print(f"  .env               : {info['.env']}")
        print()
        print(f"  {info['disclaimer']}")
        if not info["will_place_a_real_call"] and info["demo_mode"]:
            print()
            print("  No telephony credentials -> the simulation provider is used and the")
            print("  call status lifecycle is shown without placing a real call.")
        return EXIT_OK
    finally:
        dispatcher.close(timeout=0.1)


def _check_model(config: Any) -> int:
    detector = YoloDetector(config.model)
    try:
        detector.load()
    except ModelNotAvailableError as exc:
        print(f"MODEL NOT AVAILABLE: {exc}")
        return EXIT_ERROR
    print(f"model '{detector.model_name}' loaded successfully")
    print(f"  device: {config.model.device}   imgsz: {config.model.imgsz}   conf: {config.model.conf}")
    if detector._names:
        print(f"  classes: {len(detector._names)} -> {sorted(detector._names.values())[:12]}...")
    detector.close()
    return EXIT_OK


class _DropNoisyDeprecations(logging.Filter):
    """Drop the per-inference "'half' is deprecated" record from Torch.

    Belt and braces: it is neither a normal ``logging`` record nor a plain
    ``warnings`` record, so both the logging filter below and the
    ``warnings.showwarning`` override are installed. It arrives once per
    inference, which corrupts the in-place progress line and buries the workflow
    report. Anything genuinely actionable still reaches the console.
    """

    NOISE = ("'half' is deprecated", "'float' is deprecated", "quantize")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not crash logging
            return True
        return not any(token in message for token in self.NOISE)

    def install(self) -> None:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        root_logger = logging.getLogger()
        root_logger.addFilter(self)
        for handler in root_logger.handlers:
            handler.addFilter(self)

        original = warnings.showwarning

        def _show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if any(token in text for token in self.NOISE):
                return
            original(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _show


class _ConsoleProgress:
    """A live one-line status readout, so a terminal run is never silent.

    Without this, a plain ``python main.py --source ...`` prints only the start-up
    lines and the final summary, which looks identical to a hung process. With
    it, every run visibly shows frame count, throughput, the verification state
    and the score - you can *see* detection happening even when no GUI window is
    available.
    """

    #: redraw at most this often, so the line does not flood a slow terminal
    INTERVAL_SECONDS = 0.5

    def __init__(self, name: str = "SafeVision", enabled: bool = True,
                 reporter: Any = None) -> None:
        self.name = name
        self.enabled = enabled
        self.reporter = reporter
        self._start = time.perf_counter()
        self._last = 0.0
        self._frames = 0
        self.closed = False

    def push(self, frame_result: Any) -> None:
        if not self.enabled:
            return
        self._frames += 1
        now = time.perf_counter()
        if now - self._last < self.INTERVAL_SECONDS:
            return
        self._last = now
        elapsed = max(1e-6, now - self._start)
        verification = getattr(frame_result, "verification", None)
        state = getattr(verification, "state", "?")
        try:
            objects = sum(frame_result.detections.count_by_class().values())
        except Exception:  # noqa: BLE001 - a readout must never break a run
            objects = 0
        perf = getattr(frame_result, "performance", None)
        infer_ms = getattr(perf, "inference_ms", 0.0) or 0.0
        line = (f"frame {self._frames:5d} | {self._frames / elapsed:4.1f} fps "
                f"| {str(state):<18} | score {getattr(frame_result, 'accident_score', 0.0):.3f} "
                f"| objects {objects:2d} | infer {infer_ms:5.1f} ms")
        if self.reporter is not None:
            self.reporter.progress(line)
        else:
            print(f"\r   {line}", end="", flush=True)
        self.closed = False

    def close(self) -> None:
        if self.enabled and self._frames and not self.closed:
            if self.reporter is not None:
                self.reporter.clear_progress()
            else:
                print("", flush=True)
            self.closed = True


class _PreviewWindow:
    """Minimal OpenCV preview window (Phase 19).

    If the OpenCV build has no GUI backend the window cannot open. That is
    detected up front here so the run can fall back to writing an annotated
    video - otherwise ``--show`` silently produces nothing at all and the user
    is left with no visuals and no explanation.
    """

    def __init__(self, name: str = "SafeVision") -> None:
        self.name = name
        self._last_key: Optional[int] = None
        self.available = _gui_available()
        self.opened = False
        if not self.available:
            _LOGGER.warning(
                "no OpenCV GUI backend is available in this build, so a preview "
                "window cannot be opened; an annotated video will be written instead"
            )

    def push(self, frame_result: Any) -> None:
        import cv2

        canvas = frame_result.annotated_frame
        if canvas is None:
            return
        width, height = frame_result.frame_size
        max_width = 1280
        if width > max_width:
            scale = max_width / float(width)
            canvas = cv2.resize(canvas, (max_width, int(height * scale)))
        try:
            cv2.imshow(self.name, canvas)
            self.opened = True
            self._last_key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:  # no GUI backend available
            _LOGGER.warning("preview window unavailable (%s); continuing without display", exc)
            self.close()

    def push(self, frame_result: Any) -> None:
        import cv2

        canvas = frame_result.annotated_frame
        if canvas is None:
            return
        width, height = frame_result.frame_size
        max_width = 1280
        if width > max_width:
            scale = max_width / float(width)
            canvas = cv2.resize(canvas, (max_width, int(height * scale)))
        try:
            cv2.imshow(self.name, canvas)
            self._last_key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:  # no GUI backend available
            _LOGGER.warning("preview window unavailable (%s); continuing without display", exc)
            self.close()
        if self._last_key in (27, ord("q")):  # ESC or q
            _LOGGER.info("closing the preview window")
            raise KeyboardInterrupt

    def close(self) -> None:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":
    raise SystemExit(main())
