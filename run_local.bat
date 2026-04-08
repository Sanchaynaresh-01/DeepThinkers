@echo off
REM run_local.bat — Start the Email Triage server and run the baseline inference script (Windows).
REM Usage: run_local.bat [task]
REM   task: all | priority-classification | category-routing | full-triage-pipeline (default: all)

setlocal enabledelayedexpansion

set "TASK=%~1"
if "%TASK%"=="" set "TASK=all"
set PORT=7860

echo [run_local] Installing dependencies...
pip install -r requirements.txt -q

echo [run_local] Starting environment server on port %PORT%...
start /B "" uvicorn server.app:app --host 0.0.0.0 --port %PORT% --workers 1

REM Wait for server to be ready
echo [run_local] Waiting for server...
set READY=0
for /L %%i in (1,1,30) do (
    if !READY!==0 (
        curl -sf "http://localhost:%PORT%/health" >nul 2>&1
        if !errorlevel!==0 (
            echo  ready!
            set READY=1
        ) else (
            <nul set /p=.
            timeout /t 1 /nobreak >nul
        )
    )
)

if !READY!==0 (
    echo.
    echo [run_local] ERROR: Server failed to start.
    exit /b 1
)

echo [run_local] Running inference (task=%TASK%)...
set "EMAIL_TRIAGE_TASK=%TASK%"
set "ENV_BASE_URL=http://localhost:%PORT%"
python inference.py

echo [run_local] Done.
endlocal
