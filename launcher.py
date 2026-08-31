"""Entry point for the packaged Windows build.

Starts the local server, opens a browser at it, and keeps the console window
around long enough to read if something goes wrong.
"""
from __future__ import annotations

import multiprocessing
import sys
import threading
import time
import webbrowser


def main() -> None:
    # PyInstaller re-executes the program for each child process; without this the
    # app would relaunch itself in a loop.
    multiprocessing.freeze_support()

    import uvicorn

    from app.config import DATA_DIR, HOST, PORT
    from app.main import _free_port, app

    port = _free_port(HOST, PORT)
    url = f"http://{HOST}:{port}"

    print("=" * 66)
    print("  Mixed Russian / Kazakh Transcriber")
    print("=" * 66)
    print(f"  Open:      {url}")
    print(f"  Your data: {DATA_DIR}")
    print()
    print("  The first transcription downloads the speech model (about 3 GB).")
    print("  Leave this window open while you work. Close it to quit.")
    print("=" * 66)
    print()

    def open_browser() -> None:
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except Exception as exc:
        print(f"\n  The server stopped with an error: {type(exc).__name__}: {exc}\n")
        input("  Press Enter to close this window. ")
        sys.exit(1)


if __name__ == "__main__":
    main()
