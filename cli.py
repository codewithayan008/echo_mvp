"""
cli.py
------
Command-line entrypoint for running ECHO's guided assessment on a
Raspberry Pi (or any machine with a working microphone).

Usage:
    python cli.py calibrate --user alice          # record one calibration session
    python cli.py build-baseline --user alice      # fit baseline from collected sessions
    python cli.py assess --user alice              # run + score one assessment session
    python cli.py status --user alice               # show baseline/session status

If no microphone is detected, capture automatically falls back to
synthetic audio so the CLI still works for development/demo purposes
(a clear [SYNTHETIC] notice is printed in that case).
"""

import argparse
import sys

from echo.orchestrator import EchoSession
from echo.audio_capture import device_available
from echo.protocol import ASSESSMENT_PROTOCOL, MIN_CALIBRATION_SESSIONS
from echo.questionnaire import FOLLOW_UP_QUESTIONS


def _print_protocol_instructions():
    print("\nECHO guided assessment — follow each prompt:")
    for i, task in enumerate(ASSESSMENT_PROTOCOL, 1):
        print(f"  {i}. {task.label}: {task.instruction}")
    print()


def cmd_calibrate(args):
    session = EchoSession(args.user, data_dir=args.data_dir)
    if not device_available():
        print("[SYNTHETIC] No microphone detected — using simulated audio for this run.")
    _print_protocol_instructions()
    session.run_calibration_session()
    sessions = session.storage.get_sessions(args.user, session_type="calibration")
    n = len(sessions)
    print(f"Calibration session recorded. ({n}/{MIN_CALIBRATION_SESSIONS} minimum for a usable baseline)")
    if n >= MIN_CALIBRATION_SESSIONS:
        print("You have enough sessions to build a baseline: run `python cli.py build-baseline`.")


def cmd_build_baseline(args):
    session = EchoSession(args.user, data_dir=args.data_dir)
    model = session.build_baseline()
    if not model.is_usable():
        print(f"Only {model.n_sessions} calibration sessions found "
              f"(need {MIN_CALIBRATION_SESSIONS}+). Run more `calibrate` sessions first.")
        sys.exit(1)
    print(f"Baseline built from {model.n_sessions} sessions (confidence {model.confidence():.0%}).")


def cmd_assess(args):
    session = EchoSession(args.user, data_dir=args.data_dir)
    if not device_available():
        print("[SYNTHETIC] No microphone detected — using simulated audio for this run.")
    _print_protocol_instructions()
    try:
        result = session.run_assessment_session()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    dev = result["deviation"]
    print(f"\nOverall deviation score: {dev['overall_score']:.1f} / 100")
    print(f"Trend (EWMA): {result['ewma_score']:.1f}")
    if dev["flagged"]:
        print("\n>> A meaningful change from your personal baseline was detected.")
        for task_key, tr in dev["tasks"].items():
            if tr["flagged"]:
                top = ", ".join(f"{c['feature']} (z={c['z_score']})" for c in tr["contributing_features"][:3])
                print(f"   - {task_key}: {tr['composite_score']:.1f} | top signals: {top or 'n/a'}")
        if result["questionnaire_needed"]:
            _run_questionnaire(session, result["session_id"])
    else:
        print("This session is within your personal baseline.")


def _run_questionnaire(session, session_id):
    print("\nA few quick questions to add context to this change:")
    responses = {}
    for q in FOLLOW_UP_QUESTIONS:
        if q.kind == "bool":
            ans = input(f"  {q.text} [y/n]: ").strip().lower()
            responses[q.key] = ans.startswith("y")
        elif q.kind == "scale_0_4":
            ans = input(f"  {q.text} [0-4]: ").strip()
            try:
                responses[q.key] = max(0, min(4, int(ans)))
            except ValueError:
                responses[q.key] = 0
        else:
            responses[q.key] = input(f"  {q.text}: ").strip()
    session.submit_questionnaire(session_id, responses)
    print("Thanks — this has been logged alongside the session.")


def cmd_status(args):
    session = EchoSession(args.user, data_dir=args.data_dir)
    calib = session.storage.get_sessions(args.user, session_type="calibration")
    baseline = session.load_baseline()
    print(f"User: {args.user}")
    print(f"Calibration sessions: {len(calib)} (need {MIN_CALIBRATION_SESSIONS}+ for a usable baseline)")
    if baseline and baseline.is_usable():
        print(f"Baseline: ready ({baseline.n_sessions} sessions, confidence {baseline.confidence():.0%})")
    else:
        print("Baseline: not yet built / not usable")
    assessments = session.storage.get_sessions(args.user, session_type="assessment")
    print(f"Assessment sessions recorded: {len(assessments)}")


def main():
    parser = argparse.ArgumentParser(description="ECHO — personal acoustic baseline & deviation monitoring")
    parser.add_argument("--data-dir", default="./data")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [("calibrate", cmd_calibrate), ("build-baseline", cmd_build_baseline),
                      ("assess", cmd_assess), ("status", cmd_status)]:
        p = sub.add_parser(name)
        p.add_argument("--user", required=True, help="User ID (keeps each person's baseline separate)")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
