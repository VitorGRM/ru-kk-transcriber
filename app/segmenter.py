"""Voice-activity segmentation tuned for code-switching.

The unit of work is a stretch of speech bounded by real pauses, because a pause
is exactly where a speaker is likely to change language. Packing chunks to fill
Whisper's 30 s window would merge Russian and Kazakh into one chunk and force a
single language token onto both, so chunks are deliberately left short and
boundaries at long pauses are never crossed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE

MIN_CHUNK_S = 1.0      # below this there is too little signal to identify a language
MAX_CHUNK_S = 26.0     # Whisper's window is 30 s; leave room for padding
DEFAULT_BOUNDARY_GAP_S = 0.60   # a longer pause is treated as a possible switch point
DECODE_PAD_S = 0.15    # extra audio around a chunk so words are not clipped


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    short: bool = False        # too brief for language ID to stand on its own

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech(audio: np.ndarray, *, aggressiveness: float = 0.5
                  ) -> list[tuple[float, float]]:
    """Speech regions as (start_s, end_s).

    The VAD is run at fine granularity — short minimum silence, small padding —
    so that grouping decisions are made here rather than being pre-empted.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=aggressiveness,
        min_speech_duration_ms=150,
        max_speech_duration_s=MAX_CHUNK_S,
        min_silence_duration_ms=200,
        speech_pad_ms=60,
    )
    stamps = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def _split_long(start: float, end: float, audio: np.ndarray) -> list[tuple[float, float]]:
    """Cut an over-long region at its quietest interior point, recursively."""
    if end - start <= MAX_CHUNK_S:
        return [(start, end)]

    # Look for the minimum-energy 100 ms window in the middle half of the region,
    # so the cut lands in a breath rather than mid-word.
    lo = int((start + (end - start) * 0.25) * SAMPLE_RATE)
    hi = int((start + (end - start) * 0.75) * SAMPLE_RATE)
    win = int(0.10 * SAMPLE_RATE)
    seg = audio[lo:hi]
    if seg.size <= win:
        mid = (start + end) / 2
        return _split_long(start, mid, audio) + _split_long(mid, end, audio)

    energy = np.convolve(seg.astype(np.float32) ** 2, np.ones(win) / win, mode="valid")
    cut = (lo + int(np.argmin(energy)) + win // 2) / SAMPLE_RATE
    cut = min(max(cut, start + 1.0), end - 1.0)
    return _split_long(start, cut, audio) + _split_long(cut, end, audio)


def build_chunks(audio: np.ndarray, *, aggressiveness: float = 0.5,
                 boundary_gap_s: float = DEFAULT_BOUNDARY_GAP_S) -> list[Chunk]:
    """Group speech regions into chunks that never span a long pause."""
    total = len(audio) / SAMPLE_RATE
    regions = detect_speech(audio, aggressiveness=aggressiveness)

    if not regions:
        # Heavy noise, a music bed, or a very quiet source: fall back to a fixed
        # grid rather than returning an empty transcript.
        step = 20.0
        regions = [(t, min(t + step, total)) for t in np.arange(0.0, total, step)]

    # Join neighbours only across short gaps — those are pauses inside one
    # utterance. Anything longer stays a boundary.
    groups: list[list[float]] = []
    for start, end in regions:
        if (groups
                and start - groups[-1][1] < boundary_gap_s
                and end - groups[-1][0] <= MAX_CHUNK_S):
            groups[-1][1] = end
        else:
            groups.append([start, end])

    spans: list[tuple[float, float]] = []
    for start, end in groups:
        spans.extend(_split_long(start, end, audio))

    # A fragment too short to identify on its own is merged into a neighbour when
    # one is close by; otherwise it is kept and flagged so the router can lean on
    # script evidence and the surrounding context instead.
    out: list[list[float] | None] = [list(s) for s in spans]
    i = 0
    while i < len(out):
        span = out[i]
        if span is not None and span[1] - span[0] < MIN_CHUNK_S:
            prev_gap = span[0] - out[i - 1][1] if i > 0 and out[i - 1] else 1e9
            next_gap = out[i + 1][0] - span[1] if i + 1 < len(out) and out[i + 1] else 1e9
            if prev_gap <= next_gap and prev_gap < boundary_gap_s * 2:
                out[i - 1][1] = span[1]
                out[i] = None
            elif next_gap < boundary_gap_s * 2:
                out[i + 1][0] = span[0]
                out[i] = None
        i += 1

    chunks = [
        Chunk(index=i, start=max(0.0, s), end=min(total, e),
              short=(e - s) < MIN_CHUNK_S * 2)
        for i, (s, e) in enumerate([sp for sp in out if sp is not None])
        if e - s > 0.15
    ]
    for i, c in enumerate(chunks):
        c.index = i
    return chunks


def padded_slice(audio: np.ndarray, chunk: Chunk) -> np.ndarray:
    """Chunk audio with a little context so leading/trailing phonemes survive."""
    a = max(0, int((chunk.start - DECODE_PAD_S) * SAMPLE_RATE))
    b = min(len(audio), int((chunk.end + DECODE_PAD_S) * SAMPLE_RATE))
    return audio[a:b]
