@echo off
rem ============================================================
rem  AFK Journey autoplay launcher
rem    afkj doctor    - check environment
rem    afkj capture   - capture game screens (F9 save / F10 quit)
rem    afkj crop      - cut out template images
rem    afkj run       - start autoplay
rem
rem  Finds the real python.exe even when it is not on PATH.
rem  (ASCII only on purpose: non-ASCII breaks cmd parsing.)
rem ============================================================
setlocal enabledelayedexpansion

set "PY="

rem --- 1) py launcher ---
py -3 -c "pass" >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
    goto :found
)

rem --- 2) python on PATH, skipping the Microsoft Store stub ---
rem  Pure-batch substring test on purpose: calling find/findstr here would
rem  pick up Git Bash's Unix "find" when this script is run from Git Bash.
for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PY (
        set "CAND=%%I"
        if "!CAND:WindowsApps=!"=="!CAND!" set "PY=%%I"
    )
)
if defined PY goto :found

rem --- 3) common install locations ---
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PY if exist "%%~fD\python.exe" set "PY=%%~fD\python.exe"
)
if defined PY goto :found

for /d %%D in ("%PROGRAMFILES%\Python3*") do (
    if not defined PY if exist "%%~fD\python.exe" set "PY=%%~fD\python.exe"
)
if defined PY goto :found

for %%P in ("C:\Python312\python.exe" "C:\Python311\python.exe") do (
    if not defined PY if exist %%P set "PY=%%~fP"
)
if defined PY goto :found

echo [ERROR] Python was not found.
echo         Install it from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH" during setup.
exit /b 1

:found
"%~dp0.venv\Scripts\python.exe" -c "pass" >nul 2>&1
if not errorlevel 1 set "PY=%~dp0.venv\Scripts\python.exe"

%PY% "%~dp0afkj.py" %*
exit /b !ERRORLEVEL!
