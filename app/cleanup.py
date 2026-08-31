"""Post-decode text hygiene.

Whisper is known to emit fixed "filler" strings over silence, music or noise —
subtitle-site credits, sign-off phrases and so on. They are especially frequent
in Russian, where the training data was scraped from subtitled video. Left in,
they read as real speech, so they are stripped explicitly.
"""
from __future__ import annotations

import re
import unicodedata

# Substrings (lowercased, punctuation-insensitive) that mark a whole segment as
# a hallucination. Sourced from the recurring artefacts in Whisper's ru/kk output.
HALLUCINATION_MARKERS = [
    # Russian subtitle-scraper credits
    "субтитры сделал", "субтитры создавал", "субтитры делал",
    "редактор субтитров", "корректор", "dimatorzok", "субтитр",
    "перевод и субтитры", "subs.com.ru", "игорь негода",
    "продолжение следует", "продолжение в следующей серии",
    # Sign-offs / channel boilerplate
    "спасибо за просмотр", "спасибо за внимание", "подписывайтесь на канал",
    "подписывайтесь на наш канал", "ставьте лайки", "всем пока",
    "до новых встреч", "хорошего дня", "не забудьте подписаться",
    # Kazakh equivalents
    "арнаға жазылыңыз", "көргеніңіз үшін рахмет", "назарларыңызға рахмет",
    "субтитрлерді жасаған",
    # Cross-language noise tags
    "amara.org", "www.", "http",
]

# Non-speech annotations Whisper sometimes emits as bracketed tags.
TAG_RE = re.compile(r"[\[\(\{](?:музыка|music|аплодисменты|смех|шум|аудио|"
                    r"музыка играет|звук|тыныштық|музыка ойнап тұр)[^\]\)\}]*[\]\)\}]",
                    re.IGNORECASE)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_key(text: str) -> str:
    """Normalised form used for comparisons: lowercase, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()


def is_hallucination(text: str, *, avg_logprob: float = 0.0,
                     compression_ratio: float = 1.0,
                     no_speech_prob: float = 0.0) -> bool:
    """Decide whether a decoded segment should be discarded outright."""
    key = normalize_key(text)
    if not key:
        return True

    if any(marker in key for marker in HALLUCINATION_MARKERS):
        return True

    # A high compression ratio means the text is mostly repetition
    # ("да да да да да ..."), which is Whisper looping rather than speech.
    if compression_ratio > 2.6 and avg_logprob < -0.6:
        return True

    # The decoder itself flagged the audio as non-speech while returning
    # low-confidence text.
    if no_speech_prob > 0.85 and avg_logprob < -0.7:
        return True

    # A single token repeated to fill the window.
    words = key.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return True

    return False


def strip_tags(text: str) -> str:
    return _WS_RE.sub(" ", TAG_RE.sub(" ", text)).strip()


def collapse_repeats(text: str, max_run: int = 3) -> str:
    """Trim runaway token loops inside an otherwise-valid segment."""
    words = text.split()
    if len(words) < max_run + 1:
        return text
    out: list[str] = []
    run = 1
    for word in words:
        if out and normalize_key(word) == normalize_key(out[-1]):
            run += 1
            if run > max_run:
                continue
        else:
            run = 1
        out.append(word)
    return " ".join(out)


def tidy(text: str) -> str:
    """Whitespace and punctuation normalisation applied to every kept segment."""
    text = strip_tags(text).strip()
    text = collapse_repeats(text)
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)     # no space before punctuation
    text = re.sub(r"([,.!?;:…])(?=[^\s\d])", r"\1 ", text)  # space after it
    text = _WS_RE.sub(" ", text)
    return text.strip()


def dedupe_consecutive(segments: list) -> list:
    """Drop a segment that merely repeats the previous one verbatim.

    Chunk boundaries overlap slightly, so the same sentence can be decoded twice.
    """
    out = []
    for seg in segments:
        if (out and normalize_key(seg.text) == normalize_key(out[-1].text)
                and seg.start - out[-1].end < 1.5):
            out[-1].end = max(out[-1].end, seg.end)
            continue
        out.append(seg)
    return out
