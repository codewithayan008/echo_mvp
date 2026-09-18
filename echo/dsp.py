"""
echo.dsp
--------
Minimal, dependency-light digital signal processing primitives used to turn
raw respiratory audio into acoustic features.

Deliberately implemented with numpy + scipy only (no librosa / soundfile).
This keeps ECHO's footprint small enough to comfortably run on a
Raspberry Pi doing on-device (edge) inference, and avoids pulling in a large
audio stack for a handful of well-understood transforms.
"""

from __future__ import annotations
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a float32 mono signal in [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return data, sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))


# --------------------------------------------------------------------------
# Framing / windowing
# --------------------------------------------------------------------------

def frame_signal(signal: np.ndarray, sr: int, frame_ms: float = 32.0,
                  hop_ms: float = 16.0) -> np.ndarray:
    """Split signal into overlapping (frames, frame_len) windows, Hamming-tapered."""
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = int(round(sr * hop_ms / 1000.0))
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    idx = (np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len)
    frames = signal[idx]
    window = np.hamming(frame_len)
    return frames * window[None, :]


# --------------------------------------------------------------------------
# Mel filterbank + MFCC
# --------------------------------------------------------------------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 26,
                    fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        left, center, right = max(left, 0), max(center, left + 1), max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 26,
          frame_ms: float = 32.0, hop_ms: float = 16.0) -> np.ndarray:
    """Return (n_frames, n_mfcc) MFCC matrix."""
    pre_emph = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frames = frame_signal(pre_emph, sr, frame_ms, hop_ms)
    n_fft = int(2 ** np.ceil(np.log2(frames.shape[1])))
    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    power = (1.0 / n_fft) * (mag ** 2)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel_energy = power @ fb.T
    mel_energy = np.where(mel_energy <= 1e-10, 1e-10, mel_energy)
    log_mel = np.log(mel_energy)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs


# --------------------------------------------------------------------------
# Spectral / energy features (frame-wise)
# --------------------------------------------------------------------------

def spectral_features(signal: np.ndarray, sr: int, frame_ms: float = 32.0,
                       hop_ms: float = 16.0) -> dict:
    frames = frame_signal(signal, sr, frame_ms, hop_ms)
    n_fft = int(2 ** np.ceil(np.log2(frames.shape[1])))
    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = mag ** 2
    total_power = power.sum(axis=1) + 1e-12

    centroid = (power * freqs[None, :]).sum(axis=1) / total_power
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * power).sum(axis=1) / total_power)

    cumsum = np.cumsum(power, axis=1)
    rolloff_thresh = 0.85 * total_power
    rolloff_idx = np.array([np.searchsorted(cumsum[i], rolloff_thresh[i]) for i in range(len(cumsum))])
    rolloff_idx = np.clip(rolloff_idx, 0, len(freqs) - 1)
    rolloff = freqs[rolloff_idx]

    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
        "rms_energy": rms,
        "spectral_flatness": flatness,
    }


# --------------------------------------------------------------------------
# Pitch (F0) via autocorrelation
# --------------------------------------------------------------------------

def frame_f0(signal: np.ndarray, sr: int, frame_ms: float = 40.0, hop_ms: float = 20.0,
             fmin: float = 60.0, fmax: float = 400.0) -> np.ndarray:
    """Estimate per-frame fundamental frequency via autocorrelation. 0 => unvoiced."""
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = int(round(sr * hop_ms / 1000.0))
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    f0s = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * hop_len: i * hop_len + frame_len]
        frame = frame - frame.mean()
        energy = np.sum(frame ** 2)
        if energy < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        corr = corr / (corr[0] + 1e-12)
        search = corr[lag_min:lag_max]
        if len(search) == 0:
            continue
        peak_lag = np.argmax(search) + lag_min
        peak_val = corr[peak_lag]
        if peak_val > 0.3:  # voicing confidence threshold
            f0s[i] = sr / peak_lag
    return f0s


# --------------------------------------------------------------------------
# Simple energy-based voice activity / duration detection
# --------------------------------------------------------------------------

def active_duration(signal: np.ndarray, sr: int, frame_ms: float = 20.0,
                     hop_ms: float = 10.0, rel_thresh: float = 0.08) -> float:
    """Estimate the duration (seconds) of the acoustically active portion of a clip
    using an adaptive RMS-energy threshold. Used for cough/breath timing features."""
    frames = frame_signal(signal, sr, frame_ms, hop_ms)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    if rms.max() < 1e-6:
        return 0.0
    thresh = rel_thresh * rms.max()
    active = rms > thresh
    if not active.any():
        return 0.0
    hop_len_s = hop_ms / 1000.0
    return float(active.sum() * hop_len_s)
