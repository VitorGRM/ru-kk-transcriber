"""Entry point for the packaged Windows build.

Starts the local server on a background thread and draws the interface in a
native desktop window. There is no browser in the picture and, once the speech
model has been fetched, no network either.

The packaged build has no console, so everything that would have been printed
goes to data/logs/app.log instead, and anything fatal also raises a message box
— otherwise a failed start would look like nothing happening at all.

The orchestration itself lives in app.desktop.run, which `python -m app.main`
calls too, so a source checkout and the executable behave the same way.
"""
from __future__ import annotations

import multiprocessing
import sys
import traceback


def main() -> int:
    # PyInstaller re-executes the program for each child process; without this the
    # app would relaunch itself in a loop.
    multiprocessing.freeze_support()

    from app import desktop
    from app.main import app

    # desktop.run redirects stdout and stderr to the log itself, which this
    # console-less build depends on.
    return desktop.run(app)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        # This handler also covers a failure to import the app at all, so it
        # cannot assume the log redirection ever ran: without a console,
        # sys.stderr is still None here and printing would raise in its turn.
        detail = "".join(traceback.format_exception(exc))
        try:
            from app.config import LOG_DIR

            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "app.log").open("a", encoding="utf-8") as fh:
                fh.write(detail)
        except Exception:
            pass
        if sys.stderr is not None:
            sys.stderr.write(detail)

        try:
            from app.desktop import alert
        except Exception:
            import ctypes

            def alert(title, message):
                if sys.platform == "win32":
                    ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)

        alert("Transcriber stopped with an error", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
