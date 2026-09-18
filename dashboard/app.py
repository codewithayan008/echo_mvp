"""
dashboard/app.py
-----------------
Lightweight local web dashboard for ECHO. Reads only from the local
SQLite database populated by demo.py / cli.py -- no external network
calls, no cloud dependency. Intended to run on the Raspberry Pi itself
(or any machine on the same local network) at http://<pi-ip>:5000.

Shows exactly ECHO's MVP scope:
  - baseline status / confidence
  - longitudinal deviation score trend (with EWMA)
  - most recent session's per-task, per-signal-group breakdown
  - session log with questionnaire annotations
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, jsonify, request, redirect, url_for

from echo.storage import Storage
from echo.baseline import BaselineModel
from echo.deviation import COMPOSITE_ALERT_THRESHOLD
from echo.questionnaire import FOLLOW_UP_QUESTIONS, summarize_responses

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "echo.db")

app = Flask(__name__)


@app.template_filter("dateformat")
def dateformat_filter(ts):
    import datetime
    if ts is None:
        return ""
    return datetime.datetime.fromtimestamp(ts).strftime("%b %d, %Y %H:%M")


def get_storage():
    return Storage(DB_PATH)


@app.route("/")
def index():
    user_id = request.args.get("user", "demo_user")
    storage = get_storage()

    baseline_path = storage.get_baseline_path(user_id)
    baseline = BaselineModel.load(baseline_path) if baseline_path and os.path.exists(baseline_path) else None

    assessments = storage.get_sessions(user_id, session_type="assessment")
    calibrations = storage.get_sessions(user_id, session_type="calibration")

    chart_points = [
        {
            "date": _fmt_date(s["created_at"]),
            "overall": round(s["deviation_json"]["overall_score"], 1) if s["deviation_json"] else None,
            "ewma": round(s["ewma_score"], 1) if s["ewma_score"] is not None else None,
            "flagged": bool(s["deviation_json"]["flagged"]) if s["deviation_json"] else False,
        }
        for s in assessments
    ]

    latest = assessments[-1] if assessments else None
    latest_groups = None
    latest_tasks = None
    if latest and latest["deviation_json"]:
        latest_tasks = latest["deviation_json"]["tasks"]
        latest_groups = _aggregate_groups(latest_tasks)

    return render_template(
        "index.html",
        user_id=user_id,
        baseline=baseline,
        n_calibration=len(calibrations),
        chart_points=chart_points,
        assessments=list(reversed(assessments)),
        latest=latest,
        latest_groups=latest_groups,
        latest_tasks=latest_tasks,
        alert_threshold=COMPOSITE_ALERT_THRESHOLD,
    )


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    user_id = request.args.get("user", "demo_user")
    storage = get_storage()
    sessions = storage.get_sessions(user_id)
    session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        return "Session not found", 404

    questionnaire_summary = None
    if session["questionnaire_json"]:
        questionnaire_summary = summarize_responses(session["questionnaire_json"])

    return render_template(
        "session.html",
        session=session,
        user_id=user_id,
        questionnaire_summary=questionnaire_summary,
        questions=FOLLOW_UP_QUESTIONS,
    )


@app.route("/api/trend")
def api_trend():
    user_id = request.args.get("user", "demo_user")
    storage = get_storage()
    assessments = storage.get_sessions(user_id, session_type="assessment")
    return jsonify([
        {
            "date": _fmt_date(s["created_at"]),
            "overall": s["deviation_json"]["overall_score"] if s["deviation_json"] else None,
            "ewma": s["ewma_score"],
            "flagged": s["deviation_json"]["flagged"] if s["deviation_json"] else False,
        }
        for s in assessments
    ])


def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%b %d, %H:%M")


def _aggregate_groups(tasks: dict) -> dict:
    """Average each signal-group score across all tasks in a session, for a
    single 'what kind of change is this' summary view."""
    import numpy as np
    groups = {}
    for task_result in tasks.values():
        for g, score in task_result["group_scores"].items():
            groups.setdefault(g, []).append(score)
    return {g: round(float(np.mean(v)), 1) for g, v in groups.items()}


if __name__ == "__main__":
    print(f"ECHO dashboard reading from: {DB_PATH}")
    print("Open http://127.0.0.1:5000 in your browser (or http://<pi-ip>:5000 on your local network)")
    app.run(host="0.0.0.0", port=5000, debug=True)
