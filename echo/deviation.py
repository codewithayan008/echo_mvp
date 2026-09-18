"""
echo.deviation
--------------
Compares a newly captured assessment session against the person's own
baseline and produces a MULTI-SIGNAL deviation score -- not a single
brittle threshold.

Design:
  1. For every feature, compute a robust z-score against the baseline's
     median/MAD (robust so a single noisy calibration sample doesn't
     dominate).
  2. Group features into interpretable signal groups (timing, energy,
     spectral, cepstral/timbre, pitch).
  3. Aggregate each group into a group score, and combine groups into one
     composite 0-100 deviation score.
  4. Surface the top contributing features so an alert is explainable,
     not a black box.
  5. Maintain a longitudinal EWMA trend so gradual drift across sessions
     is caught even when no single session crosses the alert threshold.

This module never attempts to name a disease or condition -- its only
output is "how different is this from this person's own normal", plus
which acoustic dimensions are driving that difference.
"""

from __future__ import annotations
import numpy as np

# Which raw feature prefixes belong to which interpretable signal group.
FEATURE_GROUPS = {
    "timing": ["duration_s", "active_ratio"],
    "energy": ["rms_energy_mean", "rms_energy_std", "cough_peak_energy",
               "cough_decay_rate", "cough_event_count"],
    "spectral": ["spectral_centroid_mean", "spectral_centroid_std",
                 "spectral_bandwidth_mean", "spectral_rolloff_mean",
                 "spectral_flatness_mean", "zero_crossing_rate_mean"],
    "cepstral": None,  # matched by prefix "mfcc"
    "pitch": ["pitch_mean_hz", "pitch_std_hz", "voiced_ratio", "phonation_jitter"],
}

# Relative weight of each group in the composite score. Cepstral (timbre)
# and spectral changes tend to be the most informative acoustic signals for
# respiratory-sound quality changes, so they're weighted slightly higher.
GROUP_WEIGHTS = {
    "timing": 0.15,
    "energy": 0.20,
    "spectral": 0.25,
    "cepstral": 0.25,
    "pitch": 0.15,
}

Z_ALERT_THRESHOLD = 2.5     # per-feature "notable" threshold
COMPOSITE_ALERT_THRESHOLD = 38.0  # 0-100 scale


def _group_of(feature_name: str) -> str | None:
    if feature_name.startswith("mfcc"):
        return "cepstral"
    for group, members in FEATURE_GROUPS.items():
        if members and feature_name in members:
            return group
    return None


def robust_zscores(session_feats: dict, task_stats: dict) -> dict:
    """Per-feature robust z-scores: (x - median) / (1.4826 * MAD)."""
    z = {}
    for fname, value in session_feats.items():
        stat = task_stats.get(fname)
        if stat is None:
            continue
        z[fname] = (value - stat["median"]) / (1.4826 * stat["mad"])
    return z


def score_task(session_feats: dict, task_stats: dict) -> dict:
    """Score a single task (e.g. 'voluntary_cough') against its baseline stats."""
    z = robust_zscores(session_feats, task_stats)

    group_z = {g: [] for g in GROUP_WEIGHTS}
    for fname, zval in z.items():
        g = _group_of(fname)
        if g:
            group_z[g].append((fname, zval))

    group_scores = {}
    for g, items in group_z.items():
        if not items:
            group_scores[g] = 0.0
            continue
        mags = np.array([abs(v) for _, v in items])
        # Soft-cap individual feature z-scores, then use the MEDIAN (not RMS)
        # across the group. Median aggregation means a single noisy/outlier
        # feature can't single-handedly drag a whole signal group into an
        # alert -- a real deviation should show up across several related
        # features, not just one.
        capped = np.clip(mags, 0, 5)
        group_scores[g] = float(np.median(capped)) if len(capped) >= 3 else float(np.mean(capped))

    # Composite: weighted combination, scaled onto a 0-100 interpretable range.
    weighted = sum(GROUP_WEIGHTS[g] * group_scores.get(g, 0.0) for g in GROUP_WEIGHTS)
    composite = float(np.clip(weighted / 2.5 * 100.0, 0, 100))  # median z of ~2.5 => full scale

    contributing = sorted(
        [(fname, zval) for fname, zval in z.items() if abs(zval) >= Z_ALERT_THRESHOLD],
        key=lambda t: -abs(t[1]),
    )[:6]

    return {
        "group_scores": group_scores,
        "composite_score": composite,
        "z_scores": z,
        "contributing_features": [
            {"feature": f, "z_score": round(v, 2)} for f, v in contributing
        ],
        "flagged": composite >= COMPOSITE_ALERT_THRESHOLD,
    }


def score_session(session_feats_by_task: dict, baseline_stats: dict) -> dict:
    """Score every task in a session, plus an overall session-level composite."""
    task_results = {}
    for task_key, feats in session_feats_by_task.items():
        task_stats = baseline_stats.get(task_key)
        if not task_stats:
            continue
        task_results[task_key] = score_task(feats, task_stats)

    if not task_results:
        return {"overall_score": 0.0, "flagged": False, "tasks": {}}

    overall = float(np.mean([r["composite_score"] for r in task_results.values()]))
    n_tasks_flagged = sum(1 for r in task_results.values() if r["flagged"])
    # Session-level alert requires either a clearly elevated overall score,
    # or corroborating deviation across more than one task -- this is the
    # "multi-signal" guard against a single task's noise triggering an alert.
    flagged = overall >= COMPOSITE_ALERT_THRESHOLD or n_tasks_flagged >= 2

    return {
        "overall_score": overall,
        "flagged": flagged,
        "tasks": task_results,
    }


def update_trend(previous_ewma: float | None, new_score: float, alpha: float = 0.35) -> float:
    """Exponentially-weighted moving average of the overall deviation score,
    used to catch gradual drift across sessions rather than only single-session spikes."""
    if previous_ewma is None:
        return new_score
    return alpha * new_score + (1 - alpha) * previous_ewma
