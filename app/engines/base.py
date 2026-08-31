"""Shared result types for every transcription backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    language: str                      # "ru" or "kk"
    confidence: float = 0.0            # 0..1, from the decoder's avg_logprob
    lang_confidence: float = 0.0       # 0..1, how sure the router is of `language`
    routing: str = ""                  # human-readable note on how the call was made
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["words"] = [asdict(w) for w in self.words]
        return d


@dataclass
class Transcript:
    segments: list[Segment]
    engine: str
    duration: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text.strip())

    def language_share(self) -> dict[str, float]:
        """Fraction of spoken time attributed to each language."""
        totals: dict[str, float] = {}
        for s in self.segments:
            totals[s.language] = totals.get(s.language, 0.0) + (s.end - s.start)
        total = sum(totals.values()) or 1.0
        return {k: v / total for k, v in sorted(totals.items(), key=lambda x: -x[1])}

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "duration": self.duration,
            "text": self.text,
            "language_share": self.language_share(),
            "segments": [s.to_dict() for s in self.segments],
            "detail": self.detail,
        }


def confidence_from_logprob(avg_logprob: float) -> float:
    """Map a mean token log-probability onto a 0..1 scale for display.

    exp() is the natural reading (it is a per-token probability); the clamp keeps
    the rare positive-ish values from exceeding 1.
    """
    import math

    return max(0.0, min(1.0, math.exp(avg_logprob)))
