"""
demo.py
-------
End-to-end demonstration of the ECHO pipeline without any physical
microphone: uses echo.synthetic to generate plausible respiratory audio,
runs it through the real feature extraction / baseline / deviation-scoring
code paths, and populates the same SQLite database the dashboard reads.

Run:  python demo.py
Then: python dashboard/app.py   (or see README)
"""

import shutil
import numpy as np

from echo.orchestrator import EchoSession
from echo.protocol import MIN_CALIBRATION_SESSIONS

USER_ID = "demo_user"
DATA_DIR = "./data"


def main():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    rng = np.random.default_rng(42)
    session = EchoSession(USER_ID, data_dir=DATA_DIR)

    print(f"== ECHO demo: building personal baseline for '{USER_ID}' ==")
    n_calib = MIN_CALIBRATION_SESSIONS + 2
    for i in range(n_calib):
        # Calibration sessions: small natural day-to-day variation, no congestion.
        profile = {
            "congestion": 0.0,
            "f0_shift": rng.normal(0, 2.5),
            "cough_intensity": 1.0 + rng.normal(0, 0.05),
            "n_coughs": 3,
        }
        session.run_calibration_session(synthetic_profile=profile, rng=rng)
        print(f"  calibration session {i+1}/{n_calib} recorded")

    baseline = session.build_baseline()
    print(f"Baseline built: {baseline.n_sessions} sessions, confidence={baseline.confidence():.2f}\n")

    print("== Running 'healthy' follow-up assessment sessions (should NOT alert) ==")
    for i in range(3):
        profile = {
            "congestion": 0.0,
            "f0_shift": rng.normal(0, 2.5),
            "cough_intensity": 1.0 + rng.normal(0, 0.05),
            "n_coughs": 3,
        }
        result = session.run_assessment_session(synthetic_profile=profile, rng=rng)
        dev = result["deviation"]
        print(f"  session {i+1}: overall_score={dev['overall_score']:.1f} "
              f"flagged={dev['flagged']} ewma={result['ewma_score']:.1f}")

    print("\n== Running a deliberately CHANGED assessment session (should alert) ==")
    changed_profile = {
        "congestion": 0.65,       # simulated chest congestion / rattle
        "f0_shift": -12,          # slightly lower / strained voice
        "cough_intensity": 1.4,   # sharper, more forceful cough
        "n_coughs": 5,            # more coughing bouts
    }
    result = session.run_assessment_session(synthetic_profile=changed_profile, rng=rng)
    dev = result["deviation"]
    print(f"  session: overall_score={dev['overall_score']:.1f} flagged={dev['flagged']} "
          f"ewma={result['ewma_score']:.1f}")
    print("  Per-task breakdown:")
    for task_key, task_result in dev["tasks"].items():
        print(f"    - {task_key}: composite={task_result['composite_score']:.1f} "
              f"flagged={task_result['flagged']}")
        for c in task_result["contributing_features"]:
            print(f"        contributing: {c['feature']} (z={c['z_score']})")

    if result["questionnaire_needed"]:
        print("\n  -> ECHO would now trigger the contextual follow-up questionnaire.")
        # Simulate a plausible answer set and attach it.
        responses = {
            "new_cough": True,
            "breathlessness": True,
            "fatigue_level": 2,
            "fever_chills": False,
            "recent_illness_or_irritant": True,
            "notes": "Feeling like I'm coming down with a cold.",
        }
        session.submit_questionnaire(result["session_id"], responses)
        from echo.questionnaire import summarize_responses
        print(f"  Questionnaire summary: {summarize_responses(responses)}")

    print(f"\nDone. Data written to {DATA_DIR}/echo.db -- start the dashboard to visualize it.")


if __name__ == "__main__":
    main()
