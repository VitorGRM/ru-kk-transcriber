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

# The desktop window. pywebview ships the WebView2 interop assemblies as package
# data, and reaches its backend through pythonnet, whose "clr" module is created
# at import time and so is invisible to static analysis. These are Windows-only
# packages, hence the tolerant loop: a build on another platform still works and
# simply falls back to the browser.
for pkg in ("webview", "pythonnet", "clr_loader"):
    try:
        d, b, h = collect_all(pkg)
    except Exception as exc:
        print(f"spec: skipping {pkg} ({exc})")
        continue
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "clr",
    "webview.platforms.edgechromium",
    "webview.platforms.winforms",
    "app.desktop", "app.webview2",
]

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
    # No console: the app is a window, not a terminal session. Everything that
    # would have been printed goes to data/logs/app.log, and a failed start
    # raises a message box rather than vanishing silently.
    console=False,
    upx=False,
)

# A one-folder build: it starts in seconds, where a single-file build would have to
# unpack hundreds of megabytes on every launch.
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="RU-KK-Transcriber")
