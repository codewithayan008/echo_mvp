"""
echo.baseline
-------------
Builds and stores a person's PERSONAL acoustic baseline: for every feature,
in every protocol task, a robust center (median) and spread (MAD) computed
across that individual's own calibration sessions.

This is the crux of ECHO's design: there is no population model, no
disease classifier, and no cross-user comparison anywhere in this file.
Every number here comes from, and is only ever compared back to, one
person's own recordings.
"""

from __future__ import annotations
import json
import time
import numpy as np

from .protocol import TASK_KEYS, MIN_CALIBRATION_SESSIONS


class BaselineModel:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at: float | None = None
        self.updated_at: float | None = None
        self.n_sessions: int = 0
        # stats[task_key][feature_name] = {"median": ..., "mad": ..., "n": ...}
        self.stats: dict[str, dict[str, dict]] = {}

    # ---------------------------------------------------------------
    def fit(self, calibration_sessions: list[dict]) -> None:
        """
        calibration_sessions: list of session dicts, each shaped as
            { task_key: {feature_name: value, ...}, ... }
        """
        self.n_sessions = len(calibration_sessions)
        self.created_at = self.created_at or time.time()
        self.updated_at = time.time()
        self.stats = {}

        for task_key in TASK_KEYS:
            task_feats = [s[task_key] for s in calibration_sessions if task_key in s]
            if not task_feats:
                continue
            feature_names = task_feats[0].keys()
            self.stats[task_key] = {}
            for fname in feature_names:
                values = np.array([tf.get(fname, np.nan) for tf in task_feats], dtype=float)
                values = values[~np.isnan(values)]
                if values.size == 0:
                    continue
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                # Floor MAD so near-constant features don't produce divide-by-zero
                # (or absurdly inflated z-score) blow-ups later when a feature
                # happens to be very stable across a small number of calibration
                # sessions. Scaled relative to the feature's own magnitude.
                mad_floor = max(1e-6, 0.10 * (abs(median) + 1e-6))
                self.stats[task_key][fname] = {
                    "median": median,
                    "mad": max(mad, mad_floor),
                    "n": int(values.size),
                }

    def is_usable(self) -> bool:
        return self.n_sessions >= MIN_CALIBRATION_SESSIONS and bool(self.stats)

    def confidence(self) -> float:
        """0-1 confidence in the baseline, based on how much calibration data exists."""
        from .protocol import RECOMMENDED_CALIBRATION_SESSIONS
        return float(min(1.0, self.n_sessions / RECOMMENDED_CALIBRATION_SESSIONS))

    # ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "n_sessions": self.n_sessions,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineModel":
        m = cls(d["user_id"])
        m.created_at = d.get("created_at")
        m.updated_at = d.get("updated_at")
        m.n_sessions = d.get("n_sessions", 0)
        m.stats = d.get("stats", {})
        return m

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))
