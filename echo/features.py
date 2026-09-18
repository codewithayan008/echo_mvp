"""
echo.features
-------------
Turns a raw audio clip (for one protocol task) into a flat, named dictionary
of acoustic features. This is the representation ECHO actually stores and
compares over time -- raw audio itself never needs to leave the device, and
by default is not retained beyond the session (see storage.py).

Feature groups (deliberately generic / explainable, not disease-specific):
  - timing:   duration, active-time ratio
  - energy:   RMS level, envelope shape
  - spectral: centroid, bandwidth, rolloff, flatness, zero-crossing rate
  - cepstral: MFCC means + stds (timbral / resonance characteristics)
  - pitch:    F0 mean/std, voiced ratio (relevant mainly for phonation)
"""

from __future__ import annotations
import numpy as np
from . import dsp

N_MFCC = 13


def _safe(fn, default=0.0):
    try:
        v = fn()
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return default
        return float(v)
    except Exception:
        return default


def extract_features(audio: np.ndarray, sr: int, task_key: str) -> dict:
    """Extract a flat dict of {feature_name: float} for a single task recording."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        audio = np.zeros(int(sr * 0.5), dtype=np.float32)

    duration_s = len(audio) / sr
    active_s = dsp.active_duration(audio, sr)

    spec = dsp.spectral_features(audio, sr)
    mfccs = dsp.mfcc(audio, sr, n_mfcc=N_MFCC)
    f0 = dsp.frame_f0(audio, sr)
    voiced = f0[f0 > 0]

    feats = {
        "duration_s": duration_s,
        "active_ratio": _safe(lambda: active_s / duration_s if duration_s > 0 else 0.0),
        "rms_energy_mean": _safe(lambda: np.mean(spec["rms_energy"])),
        "rms_energy_std": _safe(lambda: np.std(spec["rms_energy"])),
        "spectral_centroid_mean": _safe(lambda: np.mean(spec["spectral_centroid"])),
        "spectral_centroid_std": _safe(lambda: np.std(spec["spectral_centroid"])),
        "spectral_bandwidth_mean": _safe(lambda: np.mean(spec["spectral_bandwidth"])),
        "spectral_rolloff_mean": _safe(lambda: np.mean(spec["spectral_rolloff"])),
        "spectral_flatness_mean": _safe(lambda: np.mean(spec["spectral_flatness"])),
        "zero_crossing_rate_mean": _safe(lambda: np.mean(spec["zero_crossing_rate"])),
        "pitch_mean_hz": _safe(lambda: np.mean(voiced) if voiced.size else 0.0),
        "pitch_std_hz": _safe(lambda: np.std(voiced) if voiced.size else 0.0),
        "voiced_ratio": _safe(lambda: voiced.size / len(f0) if len(f0) else 0.0),
    }

    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = _safe(lambda i=i: np.mean(mfccs[:, i]))
        feats[f"mfcc{i+1}_std"] = _safe(lambda i=i: np.std(mfccs[:, i]))

    # Task-specific extras
    if task_key == "voluntary_cough":
        env = spec["rms_energy"]
        feats["cough_peak_energy"] = _safe(lambda: np.max(env))
        feats["cough_decay_rate"] = _safe(lambda: _decay_rate(env))
        feats["cough_event_count"] = _safe(lambda: _count_energy_events(env))

    if task_key == "sustained_phonation":
        feats["phonation_jitter"] = _safe(lambda: _jitter(voiced))

    return feats


def _decay_rate(envelope: np.ndarray) -> float:
    """Rough post-peak decay slope, a proxy for how abruptly the sound cuts off."""
    if len(envelope) < 3:
        return 0.0
    peak_idx = int(np.argmax(envelope))
    tail = envelope[peak_idx:]
    if len(tail) < 2 or tail[0] <= 1e-9:
        return 0.0
    return float((tail[0] - tail[-1]) / max(len(tail), 1))


def _count_energy_events(envelope: np.ndarray, rel_thresh: float = 0.35) -> int:
    """Count distinct energy bursts above a relative threshold (e.g. 3 coughs)."""
    if envelope.max() < 1e-8:
        return 0
    thresh = rel_thresh * envelope.max()
    above = envelope > thresh
    transitions = np.diff(above.astype(int))
    return int(np.sum(transitions == 1) + (1 if above[0] else 0))


def _jitter(f0_voiced: np.ndarray) -> float:
    """Cycle-to-cycle F0 variability (proxy for vocal stability)."""
    if f0_voiced.size < 3:
        return 0.0
    periods = 1.0 / f0_voiced
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-9))
