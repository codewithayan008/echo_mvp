"""
echo.orchestrator
-----------------
Runs a full ECHO guided assessment session end-to-end:

  capture (per task) -> feature extraction -> [calibration: feed baseline]
                                            -> [assessment: score vs baseline,
                                                update trend, maybe trigger
                                                questionnaire] -> store

This is the module a CLI (cli.py) or demo (demo.py) calls into. It is the
one place that ties the other modules together, so each of those modules
can stay small and single-purpose.
"""

from __future__ import annotations
import os
import numpy as np

from . import features as feat_mod
from . import deviation
from .audio_capture import record_clip, device_available, AudioDeviceUnavailable, DEFAULT_SAMPLE_RATE
from .baseline import BaselineModel
from .protocol import ASSESSMENT_PROTOCOL
from .storage import Storage


class EchoSession:
    def __init__(self, user_id: str, data_dir: str = "./data"):
        self.user_id = user_id
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.storage = Storage(os.path.join(data_dir, "echo.db"))
        self.baseline_path = os.path.join(data_dir, f"baseline_{user_id}.json")

    # ------------------------------------------------------------
    def _capture_all_tasks(self, synthetic_profile: dict | None = None, rng: np.random.Generator | None = None) -> dict:
        """Capture audio for every protocol task. Uses the real mic if available;
        otherwise (e.g. dev sandbox / demo) falls back to synthetic audio so the
        rest of the pipeline can still be exercised."""
        audio_by_task = {}
        use_real_mic = device_available()
        for task in ASSESSMENT_PROTOCOL:
            if use_real_mic:
                try:
                    audio_by_task[task.key] = record_clip(task.duration_s)
                    continue
                except AudioDeviceUnavailable:
                    pass
            # Fallback: synthetic (development/demo only)
            from . import synthetic
            rng = rng or np.random.default_rng()
            audio_by_task[task.key] = synthetic.generate_task_audio(
                task.key, DEFAULT_SAMPLE_RATE, rng, synthetic_profile
            )
        return audio_by_task

    def _extract_all(self, audio_by_task: dict, sr: int = DEFAULT_SAMPLE_RATE) -> dict:
        return {tk: feat_mod.extract_features(audio, sr, tk) for tk, audio in audio_by_task.items()}

    # ------------------------------------------------------------
    def run_calibration_session(self, synthetic_profile: dict | None = None,
                                 rng: np.random.Generator | None = None) -> int:
        """Capture one calibration session and store its features (no scoring yet)."""
        audio_by_task = self._capture_all_tasks(synthetic_profile, rng)
        feats = self._extract_all(audio_by_task)
        return self.storage.add_session(self.user_id, "calibration", feats)

    def build_baseline(self) -> BaselineModel:
        """Fit a baseline from all calibration sessions collected so far and persist it."""
        sessions = self.storage.get_sessions(self.user_id, session_type="calibration")
        feats_list = [s["features_json"] for s in sessions]
        model = BaselineModel(self.user_id)
        model.fit(feats_list)
        model.save(self.baseline_path)
        self.storage.set_baseline_meta(self.user_id, self.baseline_path)
        return model

    def load_baseline(self) -> BaselineModel | None:
        path = self.storage.get_baseline_path(self.user_id)
        if path and os.path.exists(path):
            return BaselineModel.load(path)
        return None

    # ------------------------------------------------------------
    def run_assessment_session(self, synthetic_profile: dict | None = None,
                                rng: np.random.Generator | None = None) -> dict:
        """Capture + score one assessment session against the stored baseline.
        Returns {'session_id', 'deviation', 'questionnaire_needed'}."""
        baseline = self.load_baseline()
        if baseline is None or not baseline.is_usable():
            raise RuntimeError(
                "No usable baseline yet. Run more calibration sessions before assessments."
            )

        audio_by_task = self._capture_all_tasks(synthetic_profile, rng)
        feats = self._extract_all(audio_by_task)

        result = deviation.score_session(feats, baseline.stats)
        prev_ewma = self.storage.latest_ewma(self.user_id)
        new_ewma = deviation.update_trend(prev_ewma, result["overall_score"])

        from . import questionnaire as q_mod
        needs_questionnaire = q_mod.should_trigger(result)

        session_id = self.storage.add_session(
            self.user_id, "assessment", feats,
            deviation_result=result, ewma_score=new_ewma,
        )

        return {
            "session_id": session_id,
            "deviation": result,
            "ewma_score": new_ewma,
            "questionnaire_needed": needs_questionnaire,
        }

    def submit_questionnaire(self, session_id: int, responses: dict) -> None:
        self.storage.attach_questionnaire(session_id, responses)
