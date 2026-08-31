@echo off
REM One-time setup on Windows.
REM   install.bat        CPU only
REM   install.bat gpu    also install the CUDA libraries for an NVIDIA GPU
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python 3.10 or newer from https://python.org and tick
  echo "Add python.exe to PATH" during setup, then run this again.
  pause
  exit /b 1
)

echo === Creating the virtual environment ===
python -m venv .venv
if errorlevel 1 goto fail

echo === Installing dependencies ===
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 goto fail

if /i "%~1"=="gpu" (
  echo === Installing the CUDA runtime libraries ===
  .venv\Scripts\python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
  if errorlevel 1 goto fail
)

echo.
echo Setup finished. Start the app by double-clicking run.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. The error above says why.
pause
exit /b 1
