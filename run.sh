#!/usr/bin/env bash
# Start the transcriber.
#
# The app opens its own window; on Linux that needs pywebview's GTK or Qt
# backend, and without one it falls back to the browser on its own. Set
# TA_UI=browser to skip the window entirely, or TA_UI=none to serve only.
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

exec .venv/bin/python -m app.main
