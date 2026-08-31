"""Audio ingest: turn any user-supplied media file into 16 kHz mono float32.

Uses the static ffmpeg binary shipped by imageio-ffmpeg, so no system-wide
ffmpeg install is required. Falls back to PyAV (a faster-whisper dependency)
if the binary is unavailable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_duration(path: Path) -> float:
    """Best-effort media duration in seconds (0.0 if it cannot be determined)."""
    exe = _ffmpeg_exe()
    if exe:
        # ffmpeg writes stream info to stderr; parsing it avoids needing ffprobe,
        # which imageio-ffmpeg does not ship.
        proc = subprocess.run(
            [exe, "-i", str(path)], capture_output=True, text=True, errors="ignore"
        )
        for line in proc.stderr.splitlines():
            if "Duration:" in line:
                stamp = line.split("Duration:")[1].split(",")[0].strip()
                try:
                    h, m, s = stamp.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(s)
                except ValueError:
                    break
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return container.duration / 1_000_000
    except Exception:
        pass
    return 0.0


def load_audio(path: Path, *, normalize: bool = True) -> np.ndarray:
    """Decode `path` to a mono float32 waveform at 16 kHz in [-1, 1]."""
    exe = _ffmpeg_exe()
    audio: np.ndarray | None = None

    if exe:
        cmd = [
            exe,
            "-nostdin",
            "-threads", "0",
            "-i", str(path),
            "-vn",                      # ignore any video stream
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",                 # mono
            "-ar", str(SAMPLE_RATE),
            "-loglevel", "error",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
        elif not proc.stdout:
            detail = proc.stderr.decode("utf-8", "ignore").strip()[-400:]
            raise RuntimeError(f"Could not decode audio from this file. ffmpeg said: {detail}")

    if audio is None:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("The file contains no decodable audio.")

    if normalize:
        audio = _normalize(audio)
    return audio


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise to a consistent level.

    Whisper is trained on reasonably loud speech; very quiet recordings measurably
    increase both hallucinations and language-ID errors. A high percentile is used
    instead of the raw max so a single click does not flatten the whole signal.
    """
    peak = float(np.percentile(np.abs(audio), 99.9))
    if peak < 1e-4:
        return audio
    gain = min(0.95 / peak, 12.0)  # cap the boost so noise floors stay put
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def slice_audio(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(start_s * SAMPLE_RATE))
    b = min(len(audio), int(end_s * SAMPLE_RATE))
    return audio[a:b] if b > a else np.zeros(0, dtype=np.float32)
