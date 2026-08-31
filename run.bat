@echo off
REM Start the transcriber and open it in your browser.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing. Run install.bat first.
  pause
  exit /b 1
)

REM CTranslate2 loads cuBLAS and cuDNN by name, so the pip-installed CUDA
REM libraries have to be on PATH. Harmless when they are not installed.
for /f "delims=" %%i in ('.venv\Scripts\python -c "import pathlib,site;print(';'.join(str(p) for b in site.getsitepackages() for p in (pathlib.Path(b)/'nvidia').glob('*/bin') if p.is_dir()))" 2^>nul') do set "NVBIN=%%i"
if defined NVBIN set "PATH=%NVBIN%;%PATH%"

.venv\Scripts\python -m app.main
pause
