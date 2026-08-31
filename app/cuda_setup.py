"""On-demand CUDA runtime installer for the packaged Windows build.

CTranslate2 needs cuBLAS and cuDNN to run on a GPU. Those libraries are 1.8 GB
unpacked — far too much to ship inside the executable, and pure waste for anyone
transcribing on a CPU. So the executable stays small and fetches them the first
time a user with an NVIDIA GPU asks for acceleration.

The libraries are ordinary PyPI wheels, which are zip files: they are downloaded,
the DLLs are extracted next to the executable, and the folder is added to the DLL
search path. No pip, no CUDA toolkit, no admin rights.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .config import DATA_DIR

# CTranslate2 4.x is built against CUDA 12 and cuDNN 9;; staying inside those
# major versions is what keeps the DLLs loadable.
PACKAGES = {
    "nvidia-cublas-cu12": {
        "min": (12, 4), "max": (13, 0),
        # nvblas is a BLAS interception shim that CTranslate2 never calls.
        "skip": {"nvblas64_12.dll"},
    },
    "nvidia-cudnn-cu12": {
        "min": (9, 0), "max": (10, 0),
        "skip": set(),
    },
}

USER_AGENT = "ru-kk-transcriber/1.0 (+https://github.com/VitorGRM/ru-kk-transcriber)"


def cuda_dir() -> Path:
    return DATA_DIR / "cuda"


def supported() -> bool:
    """The in-process installer only works on Windows.

    Linux resolves shared libraries through LD_LIBRARY_PATH, which cannot be
    changed once the process is running, so there `setup.sh --gpu` is the answer.
    """
    return sys.platform == "win32"


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _resolve_wheel(package: str) -> tuple[str, str, int]:
    """Pick the newest compatible Windows wheel. Returns (version, url, size)."""
    spec = PACKAGES[package]
    req = urllib.request.Request(f"https://pypi.org/pypi/{package}/json",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)

    best = None
    for version, files in data["releases"].items():
        vt = _version_tuple(version)
        if not (spec["min"] <= vt < spec["max"]):
            continue
        for f in files:
            if "win_amd64" in f["filename"] and not f.get("yanked"):
                if best is None or vt > best[0]:
                    best = (vt, version, f["url"], f["size"])
    if best is None:
        raise RuntimeError(
            f"No {package} wheel found for CUDA "
            f"{spec['min'][0]}.x on Windows. Check your internet connection."
        )
    return best[1], best[2], best[3]


def _download(url: str, dest: Path, total: int,
              on_bytes: Callable[[int], None] | None = None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    done = 0
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        while chunk := resp.read(1024 * 512):
            fh.write(chunk)
            done += len(chunk)
            if on_bytes:
                on_bytes(done)


def _extract_dlls(wheel: Path, target: Path, skip: set[str]) -> list[str]:
    written = []
    with zipfile.ZipFile(wheel) as zf:
        for member in zf.namelist():
            name = member.rsplit("/", 1)[-1]
            if not name.lower().endswith(".dll") or name in skip:
                continue
            with zf.open(member) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            written.append(name)
    return written


def installed_dlls() -> list[str]:
    d = cuda_dir()
    return sorted(p.name for p in d.glob("*.dll")) if d.is_dir() else []


def is_installed() -> bool:
    """True when the libraries CTranslate2 actually opens are present."""
    names = {n.lower() for n in installed_dlls()}
    return ("cublas64_12.dll" in names
            and "cublaslt64_12.dll" in names
            and "cudnn64_9.dll" in names)


def enable_dll_path() -> bool:
    """Put the downloaded libraries on the DLL search path for this process.

    Must run before CTranslate2 opens a CUDA model; it is safe to call repeatedly.
    """
    if not supported() or not is_installed():
        return False
    d = str(cuda_dir())
    try:
        os.add_dll_directory(d)
    except (OSError, AttributeError):
        return False
    # Belt and braces: some loaders consult PATH rather than the added directory.
    if d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return True


def status() -> dict:
    from .runtime import gpu_info

    gpu = gpu_info()
    dlls = installed_dlls()
    size = sum((cuda_dir() / n).stat().st_size for n in dlls) if dlls else 0
    return {
        "supported": supported(),
        "gpu_present": bool(gpu["available"]),
        "gpu_name": gpu["name"],
        "vram_mb": gpu["vram_mb"],
        "installed": is_installed(),
        "dll_count": len(dlls),
        "bytes_on_disk": size,
        "target": str(cuda_dir()),
        "estimated_download_mb": 1230,
    }


def install(progress: Callable[[dict], None] | None = None) -> dict:
    """Download and unpack the CUDA runtime. Blocking; run it on a worker thread."""
    if not supported():
        raise RuntimeError(
            "The automatic installer is Windows-only. On Linux or macOS run "
            "./setup.sh --gpu instead."
        )

    def emit(**kw):
        if progress:
            progress(kw)

    target = cuda_dir()
    target.mkdir(parents=True, exist_ok=True)

    resolved = []
    for package in PACKAGES:
        emit(stage="resolve", package=package,
             message=f"Looking up the newest compatible {package}…")
        version, url, size = _resolve_wheel(package)
        resolved.append((package, version, url, size))

    grand_total = sum(s for _, _, _, s in resolved)
    carried = 0
    all_written: list[str] = []

    with tempfile.TemporaryDirectory(prefix="cuda-dl-") as tmp:
        for package, version, url, size in resolved:
            wheel = Path(tmp) / f"{package}-{version}.whl"
            emit(stage="download", package=package, version=version,
                 message=f"Downloading {package} {version} "
                         f"({size / 1048576:.0f} MB)…",
                 percent=round(carried / grand_total * 90, 1))

            def on_bytes(done: int, _c=carried, _s=size):
                emit(stage="download", package=package,
                     message=f"Downloading {package} — "
                             f"{done / 1048576:.0f} of {_s / 1048576:.0f} MB",
                     percent=round((_c + done) / grand_total * 90, 1))

            _download(url, wheel, size, on_bytes)
            carried += size

            emit(stage="extract", package=package,
                 message=f"Unpacking {package}…",
                 percent=round(carried / grand_total * 90, 1))
            written = _extract_dlls(wheel, target, PACKAGES[package]["skip"])
            all_written += written
            # Free the wheel before fetching the next one; together they are 1.2 GB.
            wheel.unlink(missing_ok=True)

    ok = is_installed()
    if not ok:
        raise RuntimeError(
            "The download finished but the expected libraries are missing. "
            f"Extracted: {', '.join(sorted(all_written)) or 'nothing'}"
        )

    enable_dll_path()
    emit(stage="done", message="GPU acceleration is ready.", percent=100)
    return {"installed": True, "dlls": sorted(all_written),
            "bytes_on_disk": sum((target / n).stat().st_size for n in all_written)}
