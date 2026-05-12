@echo off
REM ===========================================================================
REM CausaSent launcher
REM
REM Default (no arg)        → start API + Web together (full demo stack)
REM   run.bat menu           → interactive menu
REM   run.bat web            → Next.js dev server only
REM   run.bat api            → FastAPI inference backend only (needs best.pt)
REM   run.bat demo           → Gradio demo only (needs best.pt)
REM   run.bat all            → same as no-arg (API + Web)
REM ===========================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
if /i "%MODE%"=="menu" goto :menu
goto :dispatch

:menu
echo.
echo  CausaSent launcher
echo  ==================
echo   [1] all     api ^(new window^) + web ^(here^)    [default]
echo   [2] web     Next.js PWA frontend only         http://localhost:3000
echo   [3] api     FastAPI inference backend only    http://localhost:8000
echo   [4] demo    Gradio demo ^(full local^)         http://127.0.0.1:7860
echo   [q] quit
echo.
set "CHOICE="
set /p "CHOICE=Choose [1]: "
if "!CHOICE!"=="" set "MODE=all"
if /i "!CHOICE!"=="1" set "MODE=all"
if /i "!CHOICE!"=="2" set "MODE=web"
if /i "!CHOICE!"=="3" set "MODE=api"
if /i "!CHOICE!"=="4" set "MODE=demo"
if /i "!CHOICE!"=="q" exit /b 0

:dispatch
if /i "%MODE%"=="web"  goto :web
if /i "%MODE%"=="api"  goto :api
if /i "%MODE%"=="demo" goto :demo
if /i "%MODE%"=="all"  goto :all
echo [run] unknown mode: %MODE%
echo       usage: run.bat [web ^| api ^| demo ^| all ^| menu]
pause
exit /b 1

REM ---------------------------------------------------------------------------
:web
echo.
echo ============================================================
echo  Starting Next.js dev server
echo  Open in browser: http://localhost:3000
echo  Press Ctrl+C in this window to stop.
echo ============================================================
echo.
pushd apps\web
if not exist "node_modules\" (
  echo [run] node_modules not found, installing dependencies ^(1-2 min^)...
  call npm install --loglevel=error
  if errorlevel 1 (
    echo.
    echo [run] npm install FAILED. Make sure Node.js 18+ is on PATH.
    popd
    pause
    exit /b 1
  )
)
call npm run dev
set "EXITCODE=%ERRORLEVEL%"
popd
echo.
echo [run] Next.js exited with code %EXITCODE%.
pause
exit /b %EXITCODE%

REM ---------------------------------------------------------------------------
:api
call :install_api_deps
call :setup_env
if not exist "checkpoints\phobert\best.pt" (
  echo [run] WARNING: checkpoints\phobert\best.pt not found.
  echo       /health still responds; /analyze* will return 503 until weights are downloaded.
  echo.
)
echo.
echo ============================================================
echo  Starting FastAPI backend
echo  http://localhost:8000/health
echo  PHOBERT_CKPT      = %PHOBERT_CKPT%
echo  JAVA_HOME         = %JAVA_HOME%
echo  VnCoreNLP dir     = %CAUSASENT_VNCORENLP_DIR%
echo  Press Ctrl+C to stop.
echo ============================================================
echo.
python -m uvicorn apps.api.main:app --reload --port 8000
set "EXITCODE=%ERRORLEVEL%"
echo.
echo [run] uvicorn exited with code %EXITCODE%.
pause
exit /b %EXITCODE%

REM ---------------------------------------------------------------------------
:demo
if not exist "checkpoints\phobert\best.pt" (
  echo [run] ERROR: checkpoints\phobert\best.pt not found.
  echo       Download best.pt from Kaggle when training finishes.
  pause
  exit /b 1
)
call :setup_env
call :find_python
if "%PYTHON_EXE%"=="" (
  echo [run] ERROR: cannot find python.exe. Set CAUSASENT_PYTHON or create .venv.
  pause
  exit /b 1
)
echo.
echo ============================================================
echo  Starting Gradio demo on http://127.0.0.1:7860
echo  Press Ctrl+C to stop.
echo ============================================================
echo.
"%PYTHON_EXE%" -m src.demo.app --phobert-ckpt checkpoints\phobert\best.pt
set "EXITCODE=%ERRORLEVEL%"
pause
exit /b %EXITCODE%

REM ---------------------------------------------------------------------------
:all
call :install_api_deps
call :setup_env
echo.
echo ============================================================
echo  CausaSent full stack starting
echo ------------------------------------------------------------
echo  [1/2] FastAPI backend   in a NEW window  http://localhost:8000
echo                          ^(model load ~10-15s, then ready^)
echo  [2/2] Next.js frontend  in THIS window   http://localhost:3000
echo ============================================================
echo  Close either window to stop that service.
echo.
REM Delegate to scripts\_run_api.bat — keeps escape rules sane.
start "CausaSent API" "%~dp0scripts\_run_api.bat"
goto :web

REM ---------------------------------------------------------------------------
:install_api_deps
where uv >nul 2>&1
if %ERRORLEVEL%==0 (
  uv pip install -r apps\api\requirements.txt >nul 2>&1
  exit /b 0
)
call :find_python
if "%PYTHON_EXE%"=="" exit /b 0
"%PYTHON_EXE%" -m pip install -q -r apps\api\requirements.txt
exit /b 0

REM ---------------------------------------------------------------------------
:find_python
REM Sets PYTHON_EXE; searches CAUSASENT_PYTHON, .venv, known dev venv, PATH.
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
exit /b 0

REM ---------------------------------------------------------------------------
:setup_env
REM Env vars the inference pipeline needs at boot.
set "PHOBERT_CKPT=%CD%\checkpoints\phobert\best.pt"
set "CAUSASENT_VNCORENLP_DIR=%CD%\vncorenlp"
if "%JAVA_HOME%"=="" (
  if exist "C:\Program Files\Java\jdk-11.0.13" set "JAVA_HOME=C:\Program Files\Java\jdk-11.0.13"
)
if not "%JAVA_HOME%"=="" set "PATH=%JAVA_HOME%\bin;%PATH%"
exit /b 0
