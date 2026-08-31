"""The native desktop window.

The user interface is the same HTML the server has always served; what changes
here is where it is drawn. Instead of handing the URL to whatever browser the
machine happens to have, the page is rendered inside a WebView2 window owned by
this process — no address bar, no tabs, no second application to install.

The server still listens on 127.0.0.1 because that is how the window talks to
it, but it is bound to the loopback interface and nothing else on the network
can reach it.
"""
from __future__ import annotations

import ctypes
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

from . import webview2
from .config import DATA_DIR, DOWNLOAD_DIR, HOST, LOG_DIR, PORT, UI_MODE

WINDOW_TITLE = "Mixed Russian / Kazakh Transcriber"
MIN_SIZE = (900, 620)
DEFAULT_SIZE = (1180, 860)


class Api:
    """The bridge the page calls as `window.pywebview.api`.

    Only one thing needs crossing: saving an export. A browser would do that
    through a download; a desktop window does it through the system's own Save
    dialog, which is both less surprising and free of the download-blocking
    heuristics WebView2 inherits from Edge.
    """

    def __init__(self) -> None:
        self._window = None

    def bind(self, window) -> None:
        self._window = window

    def save_text(self, filename: str, text: str) -> str | None:
        """Ask where to put `text` and write it. Returns the path, or None if
        the user cancelled."""
        import webview

        # Never let the page choose a path — only a leaf name.
        name = Path(str(filename)).name or "transcript.txt"

        # pywebview 6 replaced the SAVE_DIALOG constant with the FileDialog enum
        # and warns on the old name, but 5.x has only the constant.
        try:
            save_dialog = webview.FileDialog.SAVE
        except AttributeError:
            save_dialog = webview.SAVE_DIALOG

        chosen = self._window.create_file_dialog(
            save_dialog,
            directory=str(DOWNLOAD_DIR),
            save_filename=name,
        )
        # The return type has moved between pywebview releases: a bare string in
        # some, a one-element sequence in others, and None on cancel in both.
        if not chosen:
            return None
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0]

        dest = Path(chosen)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return str(dest)


def wait_for_server(host: str, port: int, timeout: float = 60.0) -> bool:
    """Block until the server accepts connections, or the timeout runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def serve_in_background(app, host: str, port: int) -> threading.Thread:
    """Run uvicorn on a daemon thread so the window can own the main one.

    pywebview has to be started from the main thread on every platform, and a
    daemon thread means closing the window ends the process without a
    shutdown handshake.
    """
    import uvicorn

    def run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=run, name="uvicorn", daemon=True)
    thread.start()
    return thread


def open_window(url: str, on_status: Callable[[str], None] | None = None) -> bool:
    """Open the UI in a native window. Returns False if that was not possible.

    A False return is not an error — it means the caller should fall back to the
    browser, which still works.
    """
    def say(msg: str) -> None:
        if on_status:
            on_status(msg)

    try:
        import webview
    except Exception as exc:
        say(f"The desktop window component is unavailable ({exc}).")
        return False

    if webview2.supported() and not webview2.is_installed():
        # This build has no console, so an unannounced two-minute download would
        # look exactly like an app that failed to start. Say so first.
        notify(
            "One-time setup",
            "Windows is missing the display component this app draws its window "
            "with (Microsoft WebView2).\n\n"
            "It will be downloaded from Microsoft now — about 2 MB, a minute or "
            "so, and only this once. The app opens by itself when it is done.",
        )
        try:
            webview2.ensure(say)
        except Exception as exc:
            say(f"Could not set up the WebView2 runtime: {exc}")
            return False

    api = Api()

    # WebView2 keeps a browser profile on disk. Pointing it at our own data
    # folder keeps a build installed under Program Files working, where the
    # default location beside the executable would not be writable.
    storage = DATA_DIR / "window"
    storage.mkdir(parents=True, exist_ok=True)

    try:
        window = webview.create_window(
            WINDOW_TITLE,
            url,
            js_api=api,
            width=DEFAULT_SIZE[0],
            height=DEFAULT_SIZE[1],
            min_size=MIN_SIZE,
            text_select=True,          # the transcript is meant to be selected
        )
        api.bind(window)

        # Belt and braces: the page saves through the dialog above, but if a
        # future export ever goes out as a plain download this keeps it working.
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
        except (AttributeError, TypeError):
            pass

        # private_mode=False makes the profile persistent, so the window does not
        # rebuild its cache on every launch.
        kwargs = {"storage_path": str(storage), "private_mode": False}
        if sys.platform == "win32":
            # Without this pywebview may settle for the legacy MSHTML backend,
            # which is Internet Explorer and cannot render this page.
            kwargs["gui"] = "edgechromium"
        webview.start(**kwargs)
    except Exception as exc:
        say(f"The desktop window could not be opened: {type(exc).__name__}: {exc}")
        return False
    return True


def probe() -> dict:
    """Report whether a native window is actually possible here.

    Importing the platform backend is the only honest test: in a packaged build
    the question is not "was pywebview installed" but "did PyInstaller carry its
    interop assemblies and pythonnet across", and only an import answers that.
    Nothing here opens a window, so it is safe to call from a request handler.
    """
    info = {
        "mode": UI_MODE,
        "pywebview": None,
        "backend": None,
        "webview2_version": None,
        "native_window": False,
        "reason": "",
    }

    try:
        import webview  # noqa: F401
    except Exception as exc:
        info["reason"] = f"pywebview is unavailable: {type(exc).__name__}: {exc}"
        return info

    try:
        from importlib.metadata import version

        info["pywebview"] = version("pywebview")
    except Exception:
        info["pywebview"] = "unknown"

    if sys.platform == "win32":
        candidates = [("edgechromium", "webview.platforms.edgechromium")]
        info["webview2_version"] = webview2.installed_version()
    else:
        candidates = [("gtk", "webview.platforms.gtk"), ("qt", "webview.platforms.qt")]

    import importlib

    failures = []
    for name, module in candidates:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        info["backend"] = name
        info["native_window"] = True
        return info

    info["reason"] = "; ".join(failures)
    return info


def redirect_output_to_log() -> None:
    """Send stdout and stderr to data/logs/app.log when there is no console.

    A windowed build — the packaged .exe, or run.bat's pythonw — leaves both set
    to None, and then the first traceback raises a second, more confusing
    exception on the way out. Redirecting keeps the log and stops that.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Line buffering so the log is readable while the app is still running.
    stream = open(LOG_DIR / "app.log", "a", encoding="utf-8",
                  buffering=1, errors="replace")
    stream.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout = stream
    sys.stderr = stream


def _message_box(title: str, message: str, icon: int) -> None:
    """Best-effort message box, for when there is no console to print to."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, icon)
    except Exception:
        pass


def alert(title: str, message: str) -> None:
    _message_box(title, message, 0x10)      # MB_ICONERROR


def notify(title: str, message: str) -> None:
    _message_box(title, message, 0x40)      # MB_ICONINFORMATION


def _open_browser(url: str) -> None:
    def go() -> None:
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=go, daemon=True).start()


def _wait_forever() -> None:
    """Keep the process alive while the UI lives somewhere we do not own."""
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def run(app, host: str = HOST, port: int | None = None) -> int:
    """Serve `app` and show it. Returns the process exit code.

    Shared by the packaged executable and `python -m app.main`, so the two
    behave identically.
    """
    from .main import _free_port

    redirect_output_to_log()
    port = _free_port(host, PORT if port is None else port)
    url = f"http://{host}:{port}"

    print("=" * 66)
    print(f"  {WINDOW_TITLE}")
    print("=" * 66)
    print(f"  Address:   {url}")
    print(f"  Your data: {DATA_DIR}")
    print(f"  Log:       {LOG_DIR / 'app.log'}")
    print()
    print("  The first transcription downloads the speech model (about 3 GB).")
    print("  Everything after that runs offline on this computer.")
    print("=" * 66)
    print(flush=True)

    serve_in_background(app, host, port)
    if not wait_for_server(host, port):
        message = (f"The local server did not start on {url}.\n\n"
                   f"See {LOG_DIR / 'app.log'} for the reason.")
        print(message, flush=True)
        alert("Transcriber could not start", message)
        return 1

    if UI_MODE == "none":
        print("  TA_UI=none — serving only. Press Ctrl+C to stop.", flush=True)
        _wait_forever()
        return 0

    if UI_MODE == "browser":
        _open_browser(url)
        _wait_forever()
        return 0

    problems: list[str] = []

    def note(msg: str) -> None:
        print(f"  {msg}", flush=True)
        problems.append(msg)

    if open_window(url, on_status=note):
        return 0                    # the window was closed; we are done

    if UI_MODE == "window":
        alert("The desktop window could not be opened",
              "\n".join(problems) or "No further detail was reported.")
        return 1

    # UI_MODE == "auto": a window would have been nicer, but an interface in the
    # browser beats no interface at all.
    print("  Falling back to the browser.", flush=True)
    notify("Opening in your browser instead",
          "The desktop window could not be opened, so the app will open in your "
          "browser instead.\n\n" + "\n".join(problems))
    _open_browser(url)
    _wait_forever()
    return 0
