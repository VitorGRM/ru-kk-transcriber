"""Export a transcript to the usual interchange formats."""
from __future__ import annotations

import json

from .config import LANGUAGE_NAMES


def _timestamp(seconds: float, comma: bool = True) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_text(segments: list[dict], *, paragraphs: bool = True,
            gap: float = 1.6) -> str:
    """Plain running text. Blank lines are inserted at long pauses."""
    if not segments:
        return ""
    if not paragraphs:
        return " ".join(s["text"].strip() for s in segments if s["text"].strip())

    out: list[str] = []
    buf: list[str] = []
    prev_end = None
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        if prev_end is not None and s["start"] - prev_end > gap and buf:
            out.append(" ".join(buf))
            buf = []
        buf.append(text)
        prev_end = s["end"]
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


def to_tagged_text(segments: list[dict]) -> str:
    """Running text annotated wherever the language changes."""
    out: list[str] = []
    current = None
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        if s["language"] != current:
            current = s["language"]
            out.append(f"\n[{LANGUAGE_NAMES.get(current, current).upper()}]")
        out.append(text)
    return " ".join(out).strip()


def to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        text = s["text"].strip()
        if not text:
            continue
        blocks.append(f"{i}\n{_timestamp(s['start'])} --> {_timestamp(s['end'])}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        lines.append(f"{_timestamp(s['start'], False)} --> {_timestamp(s['end'], False)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


EXPORTERS = {
    "txt": ("text/plain; charset=utf-8", "txt"),
    "tagged": ("text/plain; charset=utf-8", "txt"),
    "srt": ("application/x-subrip; charset=utf-8", "srt"),
    "vtt": ("text/vtt; charset=utf-8", "vtt"),
    "json": ("application/json; charset=utf-8", "json"),
}


def render(fmt: str, payload: dict) -> str:
    segments = payload.get("segments", [])
    if fmt == "txt":
        return to_text(segments)
    if fmt == "tagged":
        return to_tagged_text(segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(payload)
    raise ValueError(f"Unknown export format: {fmt}")
