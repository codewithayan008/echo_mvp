# ECHO — Personalized Acoustic Health Monitoring Using Edge AI

**ECHO asks "Has this person's acoustic signature changed from their own baseline?" — not "Does this sound like a disease?"**

This is a working MVP of the architecture described in the ECHO proposal: a Raspberry-Pi-ready,
privacy-first edge system that builds a personal respiratory-acoustic baseline and flags
meaningful deviations from it over time, with an interpretable, multi-signal score and an
optional contextual questionnaire.

> **Scope guardrail:** ECHO is a personalized longitudinal *change-detection* prototype. It is
> **not** a disease classifier, a general symptom checker, a hospital dashboard, or a
> multi-sensor IoT platform, and it does not diagnose, rule out, or treat any condition. Nothing
> in this codebase compares one person's audio to another person's, or to a population model.

---

## The four MVP capabilities, and where they live

| Capability | Module(s) |
|---|---|
| 1. Standardized guided acoustic assessment | `echo/protocol.py`, `echo/audio_capture.py`, `cli.py` |
| 2. Personal acoustic baseline creation | `echo/features.py`, `echo/baseline.py` |
| 3. Longitudinal acoustic deviation detection | `echo/deviation.py`, `echo/orchestrator.py` |
| 4. Visualization of changes via dashboard | `dashboard/app.py` + templates |

Plus: `echo/questionnaire.py` (contextual follow-up, only triggered on alert) and
`echo/storage.py` (local SQLite — the single place data is persisted).

---

## How it works

### 1. Guided assessment (`echo/protocol.py`)
A fixed, four-task sequence, done the same way every time so sessions are comparable:
**normal breathing → deep breathing → sustained phonation ("aaah") → voluntary cough.**

### 2. Feature extraction (`echo/dsp.py`, `echo/features.py`)
Each task recording is reduced to ~35 named features: MFCCs (timbre/resonance), spectral
centroid/bandwidth/rolloff/flatness, zero-crossing rate, RMS energy, pitch (F0) mean/std,
duration/active-time ratio, plus a few task-specific features (cough burst count & decay rate,
phonation jitter). All implemented in plain NumPy/SciPy — no `librosa`/`soundfile` dependency,
which keeps the footprint small enough for a Pi doing on-device inference.

**Raw audio is never required to leave the device**, and by default isn't persisted at all —
`storage.py` only ever writes extracted features, scores, and (optional) questionnaire answers.

### 3. Personal baseline (`echo/baseline.py`)
After ≥5 calibration sessions (configurable via `MIN_CALIBRATION_SESSIONS`), ECHO computes a
robust **median + MAD** (median absolute deviation) for every feature, per task — from that one
person's own recordings only. This is deliberately a simple, transparent statistical model, not
a black-box classifier: every number in the baseline traces back to specific prior sessions.

### 4. Multi-signal deviation detection (`echo/deviation.py`)
A new session's features are converted to robust z-scores against the baseline, then grouped
into five interpretable signal groups (**timing, energy, spectral, cepstral/timbre, pitch**),
each scored 0–100. A weighted composite score combines the groups, and an alert requires either
a clearly elevated composite **or** corroborating deviation across more than one task — a guard
against a single noisy feature or task triggering a false alert. Every alert lists its top
contributing features (name + z-score) so it's explainable, not a black box. An EWMA trend is
also tracked across sessions to catch gradual drift, not just single-session spikes.

### 5. Contextual questionnaire (`echo/questionnaire.py`)
Only shown after an alert. Six fixed questions (new cough? breathlessness? fatigue 0–4? fever?
recent illness/irritant exposure? free-text note). Answers are stored as annotation only —
ECHO never uses them to name or rule out a condition.

### 6. Dashboard (`dashboard/`)
A local Flask app (no cloud dependency) showing: baseline status/confidence, a longitudinal
deviation-score chart (raw + EWMA trend), the latest session's per-signal-group and
per-task breakdown with top contributing features, and a session log with any questionnaire
notes attached.

---

## Running the demo (no hardware required)

```bash
pip install -r requirements.txt
python demo.py              # builds a synthetic baseline, then runs healthy +
                             # one deliberately "changed" session end-to-end
python dashboard/app.py     # open http://127.0.0.1:5000
```

`demo.py` uses `echo/synthetic.py` to generate plausible (not clinically accurate) breathing,
phonation, and cough audio through the *exact same* feature-extraction and scoring code path
used on real hardware — it exists purely so the full pipeline can be exercised and demoed
without a physical microphone. It is not part of the on-device production path.

## Running on a Raspberry Pi (real microphone)

```bash
sudo apt-get install libportaudio2
pip install -r requirements.txt

python cli.py calibrate --user alice        # repeat 5-7 times, on different days
python cli.py build-baseline --user alice
python cli.py assess --user alice           # run anytime after that
python cli.py status --user alice

python dashboard/app.py                     # view trends at http://<pi-ip>:5000
```

If no microphone is detected, the CLI automatically and visibly falls back to synthetic audio
(`[SYNTHETIC]` notice) so the workflow can still be developed/tested off-Pi.

---

## Project layout

```
echo/
  dsp.py            low-level signal processing (framing, MFCC, spectral, pitch) — numpy/scipy only
  features.py        raw audio -> named feature dict
  protocol.py          the 4-task guided assessment definition
  baseline.py            personal baseline model (median/MAD per feature per task)
  deviation.py             multi-signal deviation scoring + EWMA trend
  questionnaire.py           post-alert contextual questions
  storage.py                  local SQLite persistence (features/scores only)
  audio_capture.py             real mic capture (sounddevice/PortAudio)
  synthetic.py                  synthetic audio generator, for demo/dev only
  orchestrator.py                ties it all together into calibrate/assess workflows
dashboard/
  app.py             Flask dashboard (reads local SQLite only)
  templates/, static/  UI
cli.py              command-line entrypoint for the Pi
demo.py             end-to-end demo using synthetic audio
```

## Explicit non-goals for this MVP

- No disease/condition classification or naming
- No population-level or cross-user model or comparison
- No cloud dependency for core functioning (dashboard reads a local DB)
- No additional sensors beyond the microphone
- No general symptom-checker or diagnostic-interview behavior in the questionnaire
