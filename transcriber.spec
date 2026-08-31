# PyInstaller build definition. Used by .github/workflows/build-windows.yml;
# run locally with:  pyinstaller transcriber.spec
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# These packages carry data or native libraries that PyInstaller cannot infer:
# the Silero VAD model, the CTranslate2 and ONNX Runtime DLLs, the bundled
# ffmpeg binary, and the tokenizer's Rust extension.
# numpy is here because its 2.x layout loads several submodules lazily
# (numpy._core._exceptions among them) and static analysis does not see them.
for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "av",
            "imageio_ffmpeg", "tokenizers", "huggingface_hub", "numpy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# FastAPI imports python-multipart dynamically and raises at *route definition*
# time when it is absent, so a build without it dies on import of app.main
# rather than failing later on an upload. Static analysis cannot see it.
hiddenimports += ["multipart", "python_multipart"]

datas += [("app/static", "app/static")]

# uvicorn resolves these by name at runtime, so they are invisible to static analysis.
hiddenimports += [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
    "app.main", "app.config", "app.audio", "app.segmenter", "app.runtime",
    "app.cleanup", "app.formats", "app.engines.local_whisper", "app.engines.base",
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "PIL", "pytest", "torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="RU-KK-Transcriber",
    console=True,           # the window doubles as the log and the stop button
    upx=False,
)

# A one-folder build: it starts in seconds, where a single-file build would have to
# unpack hundreds of megabytes on every launch.
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="RU-KK-Transcriber")
