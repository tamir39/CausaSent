@echo off
REM Helper invoked by run.bat :all — runs the FastAPI backend in its own window.
REM Lives in its own file so the escape rules of `start cmd /k "..."` don't bite.

cd /d "%~dp0\.."

REM ---- Env vars ---------------------------------------------------------------
if "%PHOBERT_CKPT%"=="" set "PHOBERT_CKPT=%CD%\checkpoints\phobert\best.pt"
if "%CAUSASENT_VNCORENLP_DIR%"=="" set "CAUSASENT_VNCORENLP_DIR=%CD%\vncorenlp"
if "%JAVA_HOME%"=="" (
  if exist "C:\Program Files\Java\jdk-11.0.13" set "JAVA_HOME=C:\Program Files\Java\jdk-11.0.13"
)
if not "%JAVA_HOME%"=="" set "PATH=%JAVA_HOME%\bin;%PATH%"

REM ---- Python detection -------------------------------------------------------
REM A fresh cmd window does NOT inherit venv activation. Search common locations.
set "PYTHON_EXE="
if not "%CAUSASENT_PYTHON%"=="" if exist "%CAUSASENT_PYTHON%" set "PYTHON_EXE=%CAUSASENT_PYTHON%"
if "%PYTHON_EXE%"=="" if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "D:\AppData\PythonMasterEnv\.venv\Scripts\python.exe" set "PYTHON_EXE=D:\AppData\PythonMasterEnv\.venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" for /f "delims=" %%i in ('where python 2^>nul') do (
  if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
)
if "%PYTHON_EXE%"=="" for /f "delims=" %%i in ('where python3 2^>nul') do (
  if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
)
if "%PYTHON_EXE%"=="" (
  echo  ERROR: cannot find python.exe.
  echo         Set CAUSASENT_PYTHON env var to your interpreter path, or
  echo         create .venv\ in the repo root, or install Python globally.
  echo.
  pause
  exit /b 1
)

echo.
echo  CausaSent API
echo  http://localhost:8000   /health   /analyze   /analyze-csv
echo  Ctrl+C to stop
echo.
echo  PYTHON_EXE         = %PYTHON_EXE%
echo  PHOBERT_CKPT       = %PHOBERT_CKPT%
echo  JAVA_HOME          = %JAVA_HOME%
echo  VnCoreNLP dir      = %CAUSASENT_VNCORENLP_DIR%
echo.
echo  Loading PhoBERT-large checkpoint... ~10-15s before /analyze ready.
echo.

if not exist "%PHOBERT_CKPT%" (
  echo  WARNING: checkpoint not found at %PHOBERT_CKPT%
  echo           /health will respond, /analyze will return 503.
  echo.
)

"%PYTHON_EXE%" -m uvicorn apps.api.main:app --reload --port 8000
set "EXITCODE=%ERRORLEVEL%"

echo.
echo  uvicorn exited with code %EXITCODE%. Press any key to close.
pause >nul
exit /b %EXITCODE%
