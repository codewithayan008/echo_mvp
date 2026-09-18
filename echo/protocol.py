"""
echo.protocol
-------------
Defines ECHO's standardized guided acoustic assessment: a fixed, short
sequence of respiratory tasks performed the same way every time, so that
comparisons between sessions reflect changes in the person rather than
changes in how the recording was taken.

This is intentionally a small, fixed set of tasks (scope guard: ECHO is not
a general audio-diagnostic tool, so we do not keep adding tasks/sensors).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssessmentTask:
    key: str
    label: str
    instruction: str
    duration_s: float          # target recording duration
    min_valid_s: float         # minimum duration to count as a usable sample


ASSESSMENT_PROTOCOL: list[AssessmentTask] = [
    AssessmentTask(
        key="normal_breathing",
        label="Normal Breathing",
        instruction="Breathe normally through your mouth, at rest, for 10 seconds.",
        duration_s=10.0,
        min_valid_s=6.0,
    ),
    AssessmentTask(
        key="deep_breathing",
        label="Deep Breathing",
        instruction="Take 3 slow, deep breaths in and out through your mouth.",
        duration_s=12.0,
        min_valid_s=7.0,
    ),
    AssessmentTask(
        key="sustained_phonation",
        label="Sustained Phonation",
        instruction='Say "aaah" in one breath, as steadily as you can, for as long as is comfortable.',
        duration_s=6.0,
        min_valid_s=2.0,
    ),
    AssessmentTask(
        key="voluntary_cough",
        label="Voluntary Cough",
        instruction="Cough three times, with a short pause between each cough.",
        duration_s=8.0,
        min_valid_s=1.0,
    ),
]

TASK_KEYS = [t.key for t in ASSESSMENT_PROTOCOL]

# Minimum number of completed calibration sessions before ECHO will
# consider a personal baseline statistically usable.
MIN_CALIBRATION_SESSIONS = 5
RECOMMENDED_CALIBRATION_SESSIONS = 7
