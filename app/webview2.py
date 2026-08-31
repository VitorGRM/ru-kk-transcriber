"""Detect — and if necessary install — the Microsoft WebView2 runtime.

The desktop window is drawn by WebView2, the same rendering engine Edge uses.
Windows 11 ships it, and so do most up-to-date Windows 10 machines, but it is
not guaranteed to be there. When it is missing the app fetches Microsoft's
official bootstrapper (about 2 MB) and runs it silently, which is what turns
"you need a browser" into "you need nothing".

The bootstrapper installs per-user when it cannot write machine-wide, so this
normally completes without an admin prompt.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

# The Evergreen Runtime's product code under EdgeUpdate. Microsoft documents
# these exact registry locations as the supported detection method.
_CLIENT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_KEYS = (
    ("HKLM", rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_CLIENT_GUID}"),
    ("HKLM", rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_CLIENT_GUID}"),
    ("HKCU", rf"Software\Microsoft\EdgeUpdate\Clients\{_CLIENT_GUID}"),
)

# Permanent Microsoft short link to MicrosoftEdgeWebview2Setup.exe.
BOOTSTRAPPER_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

USER_AGENT = "ru-kk-transcriber/1.0 (+https://github.com/VitorGRM/ru-kk-transcriber)"


def supported() -> bool:
    return sys.platform == "win32"


def installed_version() -> str | None:
    """The installed runtime version, or None when it is absent."""
    if not supported():
        return None

    import winreg

    roots = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}
    for root, path in _KEYS:
        try:
            with winreg.OpenKey(roots[root], path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
        except OSError:
            continue
        # EdgeUpdate leaves the key behind after an uninstall with pv wiped to
        # 0.0.0.0, so a present key is not on its own proof of a usable runtime.
        if version and version != "0.0.0.0":
            return str(version)
    return None


def is_installed() -> bool:
    return installed_version() is not None


def install(progress: Callable[[str], None] | None = None) -> str:
    """Download and run the bootstrapper. Blocking; returns the new version.

    Raises RuntimeError when the runtime is still missing afterwards, so the
    caller can fall back to the browser rather than opening an empty window.
    """
    def say(msg: str) -> None:
        if progress:
            progress(msg)

    if not supported():
        raise RuntimeError("The WebView2 runtime exists only on Windows.")

    with tempfile.TemporaryDirectory(prefix="webview2-") as tmp:
        setup = Path(tmp) / "MicrosoftEdgeWebview2Setup.exe"
        say("Downloading the Microsoft WebView2 runtime (about 2 MB)…")
        req = urllib.request.Request(BOOTSTRAPPER_URL,
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, setup.open("wb") as fh:
            while chunk := resp.read(1024 * 256):
                fh.write(chunk)

        say("Installing it — this takes a minute and needs no restart…")
        # /silent keeps the UI out of the way; the bootstrapper itself pulls the
        # full runtime down, so this step is the one that needs the network.
        proc = subprocess.run(
            [str(setup), "/silent", "/install"],
            capture_output=True, text=True, timeout=900,
        )

    version = installed_version()
    if version is None:
        detail = (proc.stderr or proc.stdout or "").strip()[-300:]
        raise RuntimeError(
            f"The WebView2 installer exited with code {proc.returncode} but the "
            f"runtime is still not registered. {detail}".strip()
        )
    say(f"WebView2 {version} is ready.")
    return version


def ensure(progress: Callable[[str], None] | None = None) -> str | None:
    """Return the runtime version, installing it first if it is missing."""
    if not supported():
        return None
    version = installed_version()
    if version is not None:
        return version
    return install(progress)
