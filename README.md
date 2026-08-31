# Mixed Russian / Kazakh Transcriber

Speech-to-text for recordings where the speaker moves between **Russian and Kazakh**,
sometimes several times in the same conversation. Runs entirely on your own machine —
no account, no API key, no per-minute cost, nothing leaves the computer.

The transcript lands in an editable, copyable text box, and every segment shows which
language it was decoded as so you can correct the rare mistake by hand.

---

## Why not just run Whisper?

Whisper decodes a file under **one** language token. Give it mixed Russian/Kazakh audio
and you have to pick a loser: choose `ru` and the Kazakh comes out as Russian-sounding
nonsense; choose `kk` and the Russian degrades. Whisper's automatic detection only looks
at the first 30 seconds, so it just picks whichever language happens to open the file.

This app decides the language **per stretch of speech** instead:

```
audio ─▶ 16 kHz mono ─▶ Silero VAD ─▶ chunks split at real pauses
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                 language ID on the chunk        (ambiguous chunks only)
                 restricted to {ru, kk}          decode as ru AND as kk,
                          │                      then score both
                          └───────────────┬───────────────┘
                                          ▼
                          best reading wins ─▶ hallucination filter ─▶ transcript
```

Three things make the routing hold up:

**Chunks never cross a long pause.** A pause is exactly where a speaker switches
language, so boundaries are placed there instead of on Whisper's blind 30-second grid.
Chunks stay short on purpose even though longer ones would be faster.

**Ambiguous chunks are decoded both ways.** When the language posterior is close, the
chunk is transcribed as Russian *and* as Kazakh, and the readings are compared on
decoder confidence, the language posterior, and the script evidence below.

**Script evidence breaks the tie.** Kazakh Cyrillic has nine letters Russian does not
have — **ә ғ қ ң ө ұ ү һ і** — and they make up roughly a fifth of running Kazakh text.
A Kazakh-forced decode dense with them is genuinely Kazakh; one containing none of them
has really produced Russian words under a Kazakh token. This is the most reliable signal
in the system and it costs nothing to compute.

Three further corrections worth knowing about:

- Whisper is systematically less confident on Kazakh than on Russian, simply because it
  saw far less of it in training. Comparing raw log-probabilities would therefore collapse
  onto Russian almost every time, so Kazakh gets a fixed calibration offset
  (`KK_LOGPROB_BIAS` in `app/engines/local_whisper.py`).
- Whisper emits stock filler over silence and noise — subtitle-site credits, sign-offs,
  `Продолжение следует...`, `Субтитры сделал DimaTorzok`. These read as real speech, so
  they are matched and dropped in `app/cleanup.py`.
- The previous chunk's text is fed in as a prompt to keep terminology stable, but when
  Whisper cannot make sense of a chunk it will sometimes copy that prompt back out
  instead of transcribing. The result looks confident and is entirely invented, so a
  decode that overlaps its own prompt too heavily is thrown away and the chunk is
  decoded again with no context.

### Does the routing actually work?

Measured on a 22-second clip alternating Russian and Kazakh every few seconds
(`ru → kk → ru → kk`), using the small model on CPU:

| | Spans routed correctly |
|---|---|
| Whisper's own language ID alone | 2 / 4 |
| This router (dual decode + script evidence) | **4 / 4** |

Whisper's encoder put the Kazakh spans at `ru=0.89, kk=0.11` and `ru=0.83, kk=0.17` — it
would have called both of them Russian. What recovered them was decoding each span both
ways and seeing that the Kazakh reading came back 25% and 22% Kazakh-only letters while
the Russian reading came back with none. Restricting to `{ru, kk}` mattered too: the
unrestricted top language for those spans was German and English.

That clip is synthetic speech, which is harder for the acoustic model than real voices —
the point it establishes is that the *routing* holds up where plain language ID does not,
not that the word-level text is perfect at that model size.

---

## Install

`ffmpeg` is **not** required — a static build ships with the dependencies.

### Windows, no Python needed

Download **`RU-KK-Transcriber-windows.zip`** from the
[latest release](../../releases/latest), unzip it, and double-click
`RU-KK-Transcriber.exe`. It opens in **its own window** — no browser, no console, and
nothing to install first.

The window is drawn by the Microsoft **WebView2** runtime, which Windows 11 and most
up-to-date Windows 10 machines already have. If yours does not, the app downloads
Microsoft's official installer (about 2 MB) on the first launch and runs it silently;
it normally needs no admin rights. If that step cannot complete, the app says so and
opens in your browser instead rather than showing you nothing.

That build runs on the **CPU**. If you have an NVIDIA GPU, the app offers a one-click
download of the CUDA runtime from inside the window.

### Windows, from source (this is the GPU path)

```bat
git clone https://github.com/VitorGRM/ru-kk-transcriber.git
cd ru-kk-transcriber
install.bat gpu       :: omit "gpu" for a CPU-only install
run.bat
```

Needs Python 3.10+ from [python.org](https://python.org) with *Add python.exe to PATH*
ticked during setup.

### Linux and macOS

```bash
./setup.sh          # CPU only
./setup.sh --gpu    # NVIDIA GPU: also installs the CUDA runtime libraries
./run.sh            # starts the app
```

The native window on Linux and macOS needs pywebview's GTK or Qt backend
(`python3-gi` and `gir1.2-webkit2-4.1` on Debian and Ubuntu). Without one the app says
so and opens in your browser instead — the Windows build is the one that is guaranteed
browser-free.

The first transcription downloads the model (~3 GB for `large-v3`) into `data/models/`.
Later runs start immediately, and the app works offline from then on. If port 8000 is
already taken the app moves to the next free one and tells you which.

### Is it really offline?

The transcription is, always: your audio is decoded on this machine and never leaves it.
Two things do need the network, both one-off and both optional:

| What | When | Size |
|---|---|---|
| The speech model | Before the first transcription | ~3 GB |
| The WebView2 runtime | First launch, only if Windows lacks it | ~2 MB |
| The CUDA runtime | Only if you ask for GPU acceleration | ~1.2 GB |

After those, you can pull the network cable and the app works exactly the same. To set
up a machine that will never have internet, copy the whole `data/models/` folder from a
machine that has already downloaded the model, or point `TA_MODEL_DIR` at it.

### Choosing how it opens

`TA_UI` decides where the interface goes:

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Native window, falling back to the browser if that fails |
| `window` | Native window only; report an error rather than opening a browser |
| `browser` | Hand the address to your default browser |
| `none` | Serve only, open nothing |

## What it accepts

Any audio or video file, since decoding goes through the bundled ffmpeg. Verified:
`.ogg` (both Vorbis and Opus, so **WhatsApp voice notes work**), `.opus`, `.mp3`, `.wav`,
`.m4a`, `.flac`, `.mp4`, `.webm`, `.mkv`. Video files have their audio track extracted.

Anything is resampled to 16 kHz mono and peak-normalised first — quiet recordings
measurably worsen both hallucinations and language ID, so levelling them helps.

---

## Hardware

Quantisation is chosen automatically from the VRAM the app finds.

| Machine | What it picks | Speed on `large-v3` |
|---|---|---|
| NVIDIA GPU, 4 GB VRAM | `cuda` / `int8_float16` | roughly 8–15× real time |
| NVIDIA GPU, 8 GB+ | `cuda` / `float16` | roughly 15–30× real time |
| No GPU | `cpu` / `int8` | around real time, or slower |

On a 4 GB card the full `large-v3` fits in `int8_float16` alongside beam search — weights
drop from about 3.1 GB to 1.6 GB while activations stay in half precision, so the accuracy
cost is negligible. If the GPU turns out to be unusable, the app falls back to CPU and
says so rather than failing.

**`large-v3-turbo` is offered but is not the default.** Its distilled 4-layer decoder is
much faster, and it gives up more on low-resource languages than on well-resourced ones —
which in this app means Kazakh specifically. Use it for a quick first pass; use `large-v3`
for the real transcript.

---

## Using it

**Language routing** is the setting that matters:

| Mode | What it does | When |
|---|---|---|
| Fast | One pass per chunk, trusting language ID | A quick draft, or audio that is mostly one language |
| **Balanced** | Decodes ambiguous chunks both ways | **Default. Best accuracy for the time.** |
| Maximum | Decodes every chunk both ways | Heavily mixed speech, or when the result has to be right |

Maximum roughly doubles the work. On a GPU that is usually a fine trade.

Other controls:

- **Custom vocabulary** — names, places and jargon the model should expect
  (`Алматы`, `Нұрсұлтан`, `ЖСН`). This helps more than any other setting on proper nouns.
- **Language switch sensitivity** — how long a pause has to be before it counts as a
  possible switch point. Lower it if your speakers switch mid-sentence.
- **Carry context between segments** — keeps terminology and punctuation consistent
  across chunks. Turn it off if you see text repeating.

### Fixing what it gets wrong

The **Segments & languages** tab lists every segment with its language and a confidence
figure, and **Show only uncertain segments** narrows it to the ones worth checking.
Click a timestamp to hear it. If the language is wrong, click the language badge — that
segment is transcribed again in the other language and the text box updates.

Segment text is editable in place, and so is the whole transcript in the **Text** tab.
Edits are kept in every export.

### Exports

Plain text · text with language tags · SRT · WebVTT · JSON (timestamps, per-segment
language, confidence, word-level timings).

Plain-text exports use exactly what is in the text box, so manual edits are never lost.
SRT, VTT and JSON are built from the segments, since they need timestamps.

---

## Layout

```
app/
  main.py                    FastAPI routes, job registry, SSE progress
  audio.py                   decode anything to 16 kHz mono via bundled ffmpeg
  segmenter.py               VAD chunking that respects pause boundaries
  runtime.py                 GPU detection, device and quantisation choice
  cleanup.py                 hallucination filter and text tidying
  formats.py                 txt / tagged / srt / vtt / json exports
  engines/local_whisper.py   the language router — the core of the app
  desktop.py                 the native window, and the browser fallback
  webview2.py                detects and installs the WebView2 runtime
  static/                    interface (English)
data/
  uploads/                   uploaded media
  models/                    downloaded Whisper weights
  logs/app.log               what the window-less build would have printed

launcher.py                  entry point for the packaged Windows build
transcriber.spec             PyInstaller build definition
install.bat / run.bat        Windows setup and start
setup.sh / run.sh            Linux and macOS setup and start
.github/workflows/           builds and smoke-tests the Windows executable
```

## Tuning

The router's constants sit at the top of `app/engines/local_whisper.py`:

| Constant | Meaning |
|---|---|
| `KK_LOGPROB_BIAS` | Correction for Whisper's lower confidence on Kazakh |
| `LID_WEIGHT` | How much the encoder's language posterior counts against decode confidence |
| `KK_LETTER_STRONG` | Kazakh-letter density that confirms Kazakh |
| `KK_LETTER_ABSENT` | Density below which a "Kazakh" decode is really Russian |
| `SCRIPT_BONUS` | Size of the nudge those two verdicts apply |

These are defaults chosen from how the two languages are written, not values fitted to
your recordings. If routing leans the wrong way on your material, raise
`KK_LOGPROB_BIAS` to favour Kazakh or lower it to favour Russian.

## Building the Windows executable yourself

The `.exe` is built on a Windows runner by
[`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml) — PyInstaller
cannot cross-compile, so it cannot be produced from Linux or macOS. Push a `v*` tag, or
run the workflow by hand from the Actions tab. Before publishing, the workflow launches
the packaged app and checks three things: that it answers on `/api/system`, that a real
upload reaches the data folder beside the executable, and that the build still carries a
working window backend. That last check matters most — PyInstaller can quietly drop
pywebview's interop assemblies, and the result would be an app that silently falls back
to a browser on the user's machine. A build that does is failed rather than released.

To build on your own Windows machine:

```bat
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --clean transcriber.spec
```

It is a one-folder build on purpose: a single-file build would unpack several hundred
megabytes on every launch.

## Privacy

Files stay in `data/uploads/` on this machine. There is still a local server behind the
window — that is how the interface talks to the transcriber — but it binds to
`127.0.0.1`, so nothing else on the network can reach it. Delete `data/` to remove
everything, including the transcripts, the downloaded model and the window's own cache.
