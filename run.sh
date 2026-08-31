#!/usr/bin/env bash
# Start the transcriber and open it in a browser.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "The virtual environment is missing. Run ./setup.sh first." >&2
  exit 1
fi

# CTranslate2 loads cuBLAS/cuDNN by name, so the pip-installed CUDA libraries
# have to be on the loader path. Harmless when they are not installed.
NV_LIBS="$(.venv/bin/python - <<'PY'
import pathlib, site
roots = set()
for base in site.getsitepackages() + [site.getusersitepackages()]:
    nv = pathlib.Path(base) / "nvidia"
    if nv.is_dir():
        roots.update(str(p) for p in nv.glob("*/lib") if p.is_dir())
print(":".join(sorted(roots)))
PY
)"
[[ -n "$NV_LIBS" ]] && export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

HOST="${TA_HOST:-127.0.0.1}"
PORT="${TA_PORT:-8000}"

# The app moves to the next free port if this one is taken, so wait for it to
# say which port it settled on before opening a browser.
LOG="$(mktemp)"
( for _ in $(seq 1 40); do
    FOUND=$(grep -oE 'http://[0-9.]+:[0-9]+' "$LOG" 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
      for opener in xdg-open open; do
        command -v "$opener" >/dev/null 2>&1 && "$opener" "$FOUND" >/dev/null 2>&1 && break
      done
      break
    fi
    sleep 0.5
  done
  rm -f "$LOG" ) &

exec .venv/bin/python -m app.main 2>&1 | tee "$LOG"
