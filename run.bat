@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if not defined RECONIA_PYTHON set "RECONIA_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%RECONIA_PYTHON%" (
    echo Chua co moi truong Python. Hay chay setup.bat truoc.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"

"%RECONIA_PYTHON%" -m demo.app --open --port 8766 %*
set "RECONIA_EXIT=%errorlevel%"
if not "%RECONIA_EXIT%"=="0" (
    echo.
    echo Khong mo duoc Reconia. Xem thong bao loi ben tren.
    pause
)
exit /b %RECONIA_EXIT%
