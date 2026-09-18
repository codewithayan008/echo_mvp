"""
echo.audio_capture
-------------------
Microphone capture for the Raspberry Pi. Uses `sounddevice` (PortAudio)
when a real microphone is present. All capture happens locally; nothing
here ever opens a network connection.

In environments without an audio device (e.g. this development sandbox,
or CI), capture gracefully raises AudioDeviceUnavailable so callers (see
synthetic.py / demo.py) can fall back to simulated data for development
and testing without hardware.
"""

from __future__ import annotations
import numpy as np

DEFAULT_SAMPLE_RATE = 16000


class AudioDeviceUnavailable(RuntimeError):
    pass


def record_clip(duration_s: float, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Record `duration_s` seconds of mono audio from the default input device.
    Raises AudioDeviceUnavailable if no microphone / PortAudio backend is present.
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise AudioDeviceUnavailable(f"sounddevice/PortAudio not available: {e}")

    try:
        audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        raise AudioDeviceUnavailable(f"Microphone capture failed: {e}")


def device_available() -> bool:
    try:
        import sounddevice as sd
        return len(sd.query_devices()) > 0
    except Exception:
        return False
