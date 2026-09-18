"""
echo.questionnaire
------------------
A short, fixed contextual questionnaire that ECHO offers ONLY after a
deviation alert. It exists purely to attach context to an acoustic change
(e.g. "this happened right after I started a cold") -- it is not a symptom
checker, triage tool, or diagnostic interview, and ECHO does not use the
answers to name or rule out any condition. Answers are stored alongside the
session purely as annotation for the person and their dashboard.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    kind: str            # "bool" | "scale_0_4" | "text"


FOLLOW_UP_QUESTIONS: list[Question] = [
    Question("new_cough", "Have you noticed a new or worsening cough recently?", "bool"),
    Question("breathlessness", "Have you felt more breathless than usual?", "bool"),
    Question("fatigue_level", "How would you rate your fatigue today? (0 = none, 4 = severe)", "scale_0_4"),
    Question("fever_chills", "Have you had a fever or chills recently?", "bool"),
    Question("recent_illness_or_irritant",
              "Any recent cold/flu, allergies, smoke, or other irritant exposure?", "bool"),
    Question("notes", "Anything else you'd like to note about how you're feeling? (optional)", "text"),
]


def should_trigger(deviation_result: dict) -> bool:
    """Trigger the questionnaire only when ECHO's deviation detector flags the session."""
    return bool(deviation_result.get("flagged"))


def summarize_responses(responses: dict) -> str:
    """Human-readable one-line summary of questionnaire answers, for the dashboard/log."""
    if not responses:
        return "No contextual questionnaire completed."
    flags = []
    if responses.get("new_cough"):
        flags.append("new/worsening cough")
    if responses.get("breathlessness"):
        flags.append("increased breathlessness")
    if responses.get("fever_chills"):
        flags.append("fever/chills")
    if responses.get("recent_illness_or_irritant"):
        flags.append("recent illness/irritant exposure")
    fatigue = responses.get("fatigue_level")
    if fatigue is not None and fatigue >= 2:
        flags.append(f"fatigue level {fatigue}/4")
    if not flags:
        return "No notable symptoms reported alongside this change."
    return "Reported: " + ", ".join(flags) + "."
