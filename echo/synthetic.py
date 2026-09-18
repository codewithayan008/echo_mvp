"""
echo.synthetic
--------------
Synthetic respiratory-sound generator used ONLY for demoing and testing
ECHO's pipeline without physical microphone hardware. This module is not
part of the on-device production path -- on a real Raspberry Pi deployment,
audio_capture.record_clip() is used instead.

It generates plausible (not clinically accurate) breathing / phonation /
cough waveforms with a small set of controllable parameters, so the demo
can show (a) a stable personal baseline and (b) a deliberately shifted
"changed" profile for the deviation detector to catch.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt

from .audio_capture import DEFAULT_SAMPLE_RATE


def _bandpass_noise(n_samples: int, sr: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 1, n_samples)
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, noise)


def _envelope(n_samples: int, attack: float, sustain: float, release: float, sr: int) -> np.ndarray:
    a, s, r = int(attack * sr), int(sustain * sr), int(release * sr)
    a, r = max(a, 1), max(r, 1)
    env = np.ones(n_samples)
    ramp_up = np.linspace(0, 1, a)
    ramp_down = np.linspace(1, 0, r)
    env[:min(a, n_samples)] = ramp_up[:min(a, n_samples)]
    if r < n_samples:
        env[-r:] = ramp_down
    return env


def generate_breathing(duration_s: float, sr: int, rng: np.random.Generator,
                        n_cycles: int = 3, congestion: float = 0.0, noise_level: float = 0.15) -> np.ndarray:
    """Simulate cyclical inhale/exhale turbulence noise. `congestion` in [0,1]
    shifts energy toward lower-mid frequencies and adds a rattle-like texture,
    used to simulate a plausible acoustic change for the demo."""
    n_samples = int(duration_s * sr)
    audio = np.zeros(n_samples)
    cycle_len = n_samples // n_cycles
    low = 150 - 60 * congestion
    high = 1200 - 400 * congestion
    for i in range(n_cycles):
        start = i * cycle_len
        seg_len = cycle_len
        breath = _bandpass_noise(seg_len, sr, max(low, 60), max(high, low + 100), rng)
        env = _envelope(seg_len, attack=0.15 * cycle_len / sr, sustain=0.4, release=0.25 * cycle_len / sr, sr=sr)
        breath = breath * env * noise_level
        if congestion > 0:
            rattle = _bandpass_noise(seg_len, sr, 250, 700, rng) * congestion * 0.08
            rattle *= (rng.random(seg_len) > 0.985).astype(float)  # sparse crackle impulses
            breath = breath + rattle
        end = min(start + seg_len, n_samples)
        audio[start:end] += breath[: end - start]
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.6


def generate_phonation(duration_s: float, sr: int, rng: np.random.Generator,
                        f0: float = 130.0, jitter: float = 0.004, breathiness: float = 0.05) -> np.ndarray:
    """Simulate a sustained 'aaah' as a harmonic buzz plus a little breath noise."""
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr
    period_wobble = 1 + jitter * rng.normal(0, 1, n_samples).cumsum() / np.sqrt(n_samples)
    inst_f0 = f0 * period_wobble
    phase = 2 * np.pi * np.cumsum(inst_f0) / sr
    tone = sum(np.sin(k * phase) / k for k in range(1, 6))
    tone = tone / np.max(np.abs(tone) + 1e-9)
    noise = _bandpass_noise(n_samples, sr, 500, 4000, rng) * breathiness
    env = _envelope(n_samples, attack=0.05, sustain=0.7, release=0.15, sr=sr)
    audio = (tone * 0.8 + noise) * env
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.7


def generate_cough(sr: int, rng: np.random.Generator, n_coughs: int = 3,
                    intensity: float = 1.0, congestion: float = 0.0, gap_s: float = 0.9) -> np.ndarray:
    """Simulate N discrete cough bursts: a sharp broadband attack with quick decay.
    `congestion` lengthens the decay tail and adds low-frequency rattle."""
    burst_len = 0.35
    total_len = n_coughs * gap_s + burst_len
    n_samples = int(total_len * sr)
    audio = np.zeros(n_samples)
    for i in range(n_coughs):
        start_t = i * gap_s
        start = int(start_t * sr)
        blen = int(burst_len * sr)
        blen = min(blen, n_samples - start)
        if blen <= 0:
            continue
        burst = _bandpass_noise(blen, sr, 200, 3500, rng)
        release = 0.05 + 0.12 * congestion  # congested cough decays more slowly
        env = _envelope(blen, attack=0.01, sustain=0.02, release=release, sr=sr)
        burst = burst * env * intensity
        if congestion > 0:
            rattle = _bandpass_noise(blen, sr, 150, 500, rng) * congestion * 0.15
            burst = burst + rattle * env
        audio[start:start + blen] += burst
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.8


def generate_task_audio(task_key: str, sr: int, rng: np.random.Generator, profile: dict | None = None) -> np.ndarray:
    """profile: optional dict of overrides, e.g. {"congestion": 0.6, "f0_shift": -8}
    used by the demo to simulate a 'changed' session against a stable baseline."""
    profile = profile or {}
    congestion = profile.get("congestion", 0.0)

    if task_key == "normal_breathing":
        return generate_breathing(10.0, sr, rng, n_cycles=4, congestion=congestion, noise_level=0.15)
    if task_key == "deep_breathing":
        return generate_breathing(12.0, sr, rng, n_cycles=3, congestion=congestion, noise_level=0.30)
    if task_key == "sustained_phonation":
        f0 = profile.get("f0", 130.0) + profile.get("f0_shift", 0.0)
        jitter = 0.004 + 0.01 * congestion
        return generate_phonation(5.0, sr, rng, f0=f0, jitter=jitter, breathiness=0.05 + 0.1 * congestion)
    if task_key == "voluntary_cough":
        intensity = profile.get("cough_intensity", 1.0)
        n_coughs = profile.get("n_coughs", 3)
        return generate_cough(sr, rng, n_coughs=n_coughs, intensity=intensity, congestion=congestion)
    raise ValueError(f"Unknown task_key: {task_key}")


def generate_session(rng: np.random.Generator, sr: int = DEFAULT_SAMPLE_RATE, profile: dict | None = None) -> dict:
    """Generate a full synthetic session: {task_key: audio_array}."""
    from .protocol import TASK_KEYS
    return {tk: generate_task_audio(tk, sr, rng, profile) for tk in TASK_KEYS}
