"""Local Whisper backend with per-chunk Russian/Kazakh routing.

Plain Whisper decodes a file under a single language token. On audio that
switches between Russian and Kazakh that forces one language to be wrong for
part of the recording. This backend instead segments on pauses and decides the
language for each chunk independently, so a speaker can switch mid-recording
and each stretch is decoded under the token it actually belongs to.
"""
from __future__ import annotations

import math
import re
import threading
from typing import Callable

import numpy as np

from ..cleanup import dedupe_consecutive, is_hallucination, normalize_key, tidy
from ..config import SAMPLE_RATE
from ..runtime import resolve_placement
from ..segmenter import DECODE_PAD_S, build_chunks, padded_slice
from .base import Segment, Transcript, Word, confidence_from_logprob

# The nine letters that exist in the Kazakh Cyrillic alphabet but not the Russian
# one. Kazakh running text is dense with them (~10% of all letters), so their
# presence — or total absence — in a Kazakh-forced decode is a high-precision
# signal about which language was actually spoken.
KAZAKH_ONLY = set("әғқңөұүһі"   # ә ғ қ ң ө ұ ү һ і
                  "ӘҒҚҢӨҰҮҺІ")  # uppercase
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# --- Router calibration ---------------------------------------------------
# Whisper saw far less Kazakh than Russian in training, so it is systematically
# less confident on Kazakh even when Kazakh is correct. Without a correction the
# log-probability comparison would collapse onto Russian for nearly every chunk.
KK_LOGPROB_BIAS = 0.15
LID_WEIGHT = 0.5          # how much the encoder's language posterior counts
KK_LETTER_STRONG = 0.040  # Kazakh-letter density that confirms Kazakh
KK_LETTER_ABSENT = 0.005  # ...and below which a "Kazakh" decode is really Russian
SCRIPT_BONUS = 0.60       # score nudge applied on those two script verdicts

# Context from the previous chunk is fed in as a prompt to keep terminology stable,
# but when Whisper cannot make sense of the audio it will sometimes copy that prompt
# out verbatim instead of transcribing. Such a decode looks confident and is entirely
# fabricated, so it is detected and thrown away rather than scored.
ECHO_THRESHOLD = 0.50
ECHO_PENALTY = 3.0

OTHER_LANG = {"ru": "kk", "kk": "ru"}
MODEL_CHOICES = ["large-v3", "large-v3-turbo", "medium", "small", "base"]
ROUTING_MODES = ["fast", "balanced", "maximum"]


def echo_ratio(text: str, context: str) -> float:
    """How much of `text` was simply copied out of the prompt it was given.

    Measured as the share of the decode's 4-grams that already appear in the
    context; short decodes fall back to plain containment.
    """
    if not context or not text:
        return 0.0
    t, c = normalize_key(text).split(), normalize_key(context).split()
    if not t or not c:
        return 0.0

    n = 4
    if len(t) < n:
        return 1.0 if " ".join(t) in " ".join(c) else 0.0

    seen = {tuple(c[i:i + n]) for i in range(max(0, len(c) - n + 1))}
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return sum(1 for g in grams if g in seen) / len(grams)


def kazakh_letter_ratio(text: str) -> float:
    """Share of Cyrillic characters that are Kazakh-specific."""
    cyrillic = CYRILLIC_RE.findall(text)
    if len(cyrillic) < 12:          # too little text to judge
        return -1.0
    return sum(1 for ch in cyrillic if ch in KAZAKH_ONLY) / len(cyrillic)


class CodeSwitchTranscriber:
    """Wraps a faster-whisper model and adds language routing on top of it."""

    def __init__(self, model_name: str = "large-v3", device: str = "auto",
                 compute_type: str = "auto", cpu_threads: int = 0,
                 download_root: str | None = None):
        self.model_name = model_name
        self.requested_device = device
        self.requested_compute = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self.model = None
        self.placement_note = ""
        self.device = ""
        self.compute_type = ""
        self._lock = threading.Lock()

    # -- model lifecycle ---------------------------------------------------
    def load(self, progress: Callable[[str], None] | None = None) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            from faster_whisper import WhisperModel

            device, compute, note = resolve_placement(
                self.requested_device, self.requested_compute, self.model_name
            )
            if progress:
                progress(f"Loading {self.model_name} ({device}/{compute})…")

            try:
                self.model = WhisperModel(
                    self.model_name, device=device, compute_type=compute,
                    cpu_threads=self.cpu_threads or 0,
                    download_root=self.download_root,
                )
            except Exception as exc:
                # A GPU can be present but too small for the chosen model, or the
                # CUDA libraries can be missing. Degrading to CPU beats failing.
                if device != "cpu":
                    note = (f"Could not start on {device} ({type(exc).__name__}: {exc}). "
                            "Fell back to CPU with int8.")
                    if progress:
                        progress(note)
                    device, compute = "cpu", "int8"
                    self.model = WhisperModel(
                        self.model_name, device=device, compute_type=compute,
                        cpu_threads=self.cpu_threads or 0,
                        download_root=self.download_root,
                    )
                else:
                    raise

            self.device, self.compute_type, self.placement_note = device, compute, note

    def unload(self) -> None:
        self.model = None

    # -- language identification ------------------------------------------
    def _language_posterior(self, chunk_audio: np.ndarray
                            ) -> tuple[dict[str, float], str, float]:
        """Return ({'ru': p, 'kk': p}, top_language_overall, its probability).

        The pair is renormalised over Russian and Kazakh alone. The unrestricted
        top language is returned as well, purely so a chunk that is really some
        third language can be flagged to the user.
        """
        try:
            top_lang, top_prob, all_probs = self.model.detect_language(audio=chunk_audio)
        except Exception:
            return {"ru": 0.5, "kk": 0.5}, "ru", 0.0

        probs = dict(all_probs)
        p_ru = float(probs.get("ru", 0.0))
        p_kk = float(probs.get("kk", 0.0))
        total = p_ru + p_kk
        if total <= 1e-9:
            pair = {"ru": 0.5, "kk": 0.5}
        else:
            pair = {"ru": p_ru / total, "kk": p_kk / total}
        return pair, top_lang, float(top_prob)

    # -- decoding ----------------------------------------------------------
    def _decode(self, chunk_audio: np.ndarray, language: str, *,
                beam_size: int, hotwords: str, context: str) -> dict:
        """Force-decode one chunk under `language` and summarise the result."""
        segments, info = self.model.transcribe(
            chunk_audio,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=beam_size,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            # Context is supplied explicitly per chunk instead of being carried by
            # the decoder, which stops one bad chunk from poisoning the rest.
            condition_on_previous_text=False,
            initial_prompt=context or None,
            hotwords=hotwords or None,
            word_timestamps=True,
            vad_filter=False,          # already segmented upstream
            suppress_blank=True,
        )
        segs = list(segments)

        text = tidy(" ".join(s.text for s in segs))
        if not segs:
            return {"language": language, "text": "", "score": -9.9, "logprob": -9.9,
                    "segments": [], "compression": 1.0, "no_speech": 1.0,
                    "echo": 0.0, "context": context}

        weights = [max(0.05, s.end - s.start) for s in segs]
        total_w = sum(weights)
        logprob = sum(s.avg_logprob * w for s, w in zip(segs, weights)) / total_w
        compression = max(s.compression_ratio for s in segs)
        no_speech = min(s.no_speech_prob for s in segs)

        return {"language": language, "text": text, "logprob": logprob,
                "segments": segs, "compression": compression, "no_speech": no_speech,
                "score": logprob, "echo": echo_ratio(text, context), "context": context}

    def _score_candidate(self, cand: dict, lid: dict[str, float]) -> tuple[float, str]:
        """Combine decoder confidence, language posterior and script evidence."""
        lang = cand["language"]
        notes: list[str] = []

        score = cand["logprob"]
        if lang == "kk":
            score += KK_LOGPROB_BIAS

        p = max(1e-4, lid.get(lang, 0.5))
        score += LID_WEIGHT * math.log(p)

        if lang == "kk":
            ratio = kazakh_letter_ratio(cand["text"])
            if ratio >= KK_LETTER_STRONG:
                score += SCRIPT_BONUS
                notes.append(f"Kazakh-only letters at {ratio:.1%}")
            elif 0 <= ratio <= KK_LETTER_ABSENT:
                # A Kazakh-forced decode that contains no Kazakh letters has really
                # produced Russian text under a Kazakh token.
                score -= SCRIPT_BONUS
                notes.append("no Kazakh-only letters present")

        if cand.get("echo", 0.0) >= ECHO_THRESHOLD:
            score -= ECHO_PENALTY
            notes.append(f"repeated the previous segment ({cand['echo']:.0%} overlap)")

        if not cand["text"].strip():
            score -= 5.0
            notes.append("empty decode")
        if cand["compression"] > 2.6:
            score -= 1.0
            notes.append("repetitive output")

        return score, "; ".join(notes)

    # -- public API --------------------------------------------------------
    def transcribe(self, audio: np.ndarray, *, mode: str = "balanced",
                   beam_size: int = 5, hotwords: str = "",
                   vad_aggressiveness: float = 0.5, boundary_gap_s: float = 0.60,
                   ambiguity_threshold: float = 0.90,
                   use_context: bool = True,
                   progress: Callable[[dict], None] | None = None) -> Transcript:
        self.load(lambda m: progress and progress({"stage": "load", "message": m}))

        chunks = build_chunks(audio, aggressiveness=vad_aggressiveness,
                              boundary_gap_s=boundary_gap_s)
        total_duration = len(audio) / SAMPLE_RATE
        if progress:
            progress({"stage": "segmented", "chunks": len(chunks),
                      "duration": total_duration})

        out: list[Segment] = []
        context: dict[str, str] = {"ru": "", "kk": ""}
        dual_count = 0
        seg_id = 0

        previous_lang: str | None = None

        for chunk in chunks:
            piece = padded_slice(audio, chunk)
            offset = max(0.0, chunk.start - DECODE_PAD_S)
            if piece.size < int(0.15 * SAMPLE_RATE):
                continue

            lid, top_lang, top_prob = self._language_posterior(piece)

            # A fragment of a second or two carries little language evidence, and a
            # speaker almost never switches language for a single short phrase, so
            # the previous chunk's language is used as a prior.
            if chunk.short and previous_lang:
                lid = dict(lid)
                lid[previous_lang] = min(0.99, lid[previous_lang] + 0.15)
                other = OTHER_LANG[previous_lang]
                lid[other] = max(0.01, 1.0 - lid[previous_lang])

            leader = "kk" if lid["kk"] > lid["ru"] else "ru"
            margin = abs(lid["ru"] - lid["kk"])

            # Decide whether one decode is enough or both languages must be tried.
            if mode == "maximum":
                candidates = ["ru", "kk"]
            elif mode == "fast":
                candidates = [leader]
            else:
                candidates = [leader] if margin >= ambiguity_threshold else ["ru", "kk"]

            if len(candidates) > 1:
                dual_count += 1

            results = []
            for lang in candidates:
                ctx = context.get(lang, "") if use_context else ""
                results.append(self._decode(piece, lang, beam_size=beam_size,
                                            hotwords=hotwords, context=ctx))

            scored = [(self._score_candidate(c, lid), c) for c in results]
            scored.sort(key=lambda x: x[0][0], reverse=True)
            (best_score, note), best = scored[0]

            # Every reading merely parroted the prompt. Decode again with no
            # context so the chunk is transcribed from the audio alone.
            if all(c.get("echo", 0.0) >= ECHO_THRESHOLD for c in results):
                retry = [self._decode(piece, c["language"], beam_size=beam_size,
                                      hotwords=hotwords, context="")
                         for c in results]
                rescored = [(self._score_candidate(c, lid), c) for c in retry]
                rescored.sort(key=lambda x: x[0][0], reverse=True)
                scored = rescored
                (best_score, note), best = scored[0]
                note = (note + "; " if note else "") + "re-decoded without context"

            if top_lang not in ("ru", "kk") and top_prob > 0.60:
                note = ((note + "; " if note else "")
                        + f"audio may actually be '{top_lang}' ({top_prob:.0%})")

            if len(scored) > 1:
                runner_score = scored[1][0][0]
                gap = best_score - runner_score
                lang_conf = 1.0 / (1.0 + math.exp(-2.5 * gap))
                routing = (f"compared ru vs kk (margin {gap:.2f})"
                           + (f"; {note}" if note else ""))
            else:
                lang_conf = float(lid[leader])
                routing = f"language ID {lid[leader]:.0%} confident"
                if note:
                    routing += f"; {note}"

            lang = best["language"]

            for s in best["segments"]:
                text = tidy(s.text)
                if not text:
                    continue
                if is_hallucination(text, avg_logprob=s.avg_logprob,
                                    compression_ratio=s.compression_ratio,
                                    no_speech_prob=s.no_speech_prob):
                    continue
                words = [Word(start=offset + (w.start or 0.0),
                              end=offset + (w.end or 0.0),
                              word=w.word, probability=w.probability)
                         for w in (s.words or [])]
                out.append(Segment(
                    id=seg_id,
                    start=round(offset + s.start, 3),
                    end=round(offset + s.end, 3),
                    text=text,
                    language=lang,
                    confidence=round(confidence_from_logprob(s.avg_logprob), 4),
                    lang_confidence=round(lang_conf, 4),
                    routing=routing,
                    words=words,
                ))
                seg_id += 1

            previous_lang = lang
            if use_context and best["text"]:
                # Keep a short tail as style/terminology context for the next chunk
                # in the same language.
                context[lang] = best["text"][-180:]

            if progress:
                progress({"stage": "chunk", "index": chunk.index + 1,
                          "total": len(chunks), "language": lang,
                          "start": chunk.start, "end": chunk.end,
                          "text": best["text"], "routing": routing})

        out = dedupe_consecutive(out)
        for i, s in enumerate(out):
            s.id = i

        return Transcript(
            segments=out,
            engine=f"local:{self.model_name}",
            duration=total_duration,
            detail={
                "device": self.device,
                "compute_type": self.compute_type,
                "placement": self.placement_note,
                "chunks": len(chunks),
                "dual_decoded_chunks": dual_count,
                "routing_mode": mode,
                "beam_size": beam_size,
            },
        )

    def redecode(self, audio: np.ndarray, start: float, end: float, language: str,
                 *, beam_size: int = 5, hotwords: str = "") -> list[Segment]:
        """Re-transcribe one span under a language the user picked by hand."""
        self.load()
        piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        res = self._decode(piece, language, beam_size=beam_size,
                           hotwords=hotwords, context="")
        out: list[Segment] = []
        for i, s in enumerate(res["segments"]):
            text = tidy(s.text)
            if not text:
                continue
            out.append(Segment(
                id=i, start=round(start + s.start, 3), end=round(start + s.end, 3),
                text=text, language=language,
                confidence=round(confidence_from_logprob(s.avg_logprob), 4),
                lang_confidence=1.0, routing="manual override",
                words=[Word(start + (w.start or 0), start + (w.end or 0),
                            w.word, w.probability) for w in (s.words or [])],
            ))
        return out
